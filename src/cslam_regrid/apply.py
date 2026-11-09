"""Apply stage: remap CAM native (ncol) history files to lat-lon, writing
output that mimics CAM's own interpolated lat-lon history files.

Vector fields (wind/stress pairs) are remapped as vectors: (u,v) components
are rotated to Cartesian (vx,vy,vz) with the local east/north basis at the
source cells, the three components are remapped as smooth scalars, and the
result is projected onto the east/north basis of the destination cells.
This avoids the large componentwise errors near the geographic poles, where
the (u,v) basis rotates rapidly.  Pairs are auto-detected from metadata
(name pattern U*<->V* or *X<->*Y, long_name zonal/meridional, matching
dimensions) or given explicitly.

Land (CLM) native history files ('lndgrid' dimension = the full physics
grid, _FillValue over ocean, fractional landfrac on coasts) take the masked
fracarea path: source moments are weighted by landfrac, the validity mask
is taken per field / per level / per record from the fill pattern (CLM
fields differ in their valid sets, even in time), cells whose
reconstruction stencil touches an invalid cell fall back to first order,
limiter bounds use valid neighbors only, and each destination value is the
mean over the covered (land) portion of the cell.  The landfrac-weighted
land integral is conserved exactly; destination landfrac/landmask are
written to the output.
"""
from __future__ import annotations

import datetime
import os
import re

import numpy as np
import netCDF4

from . import __version__
from . import fkernels as fk
from .grids import SourceGrid
from .weights_io import read_weights

LIMITER_MODES = {"none": 0, "shape": 1, "posdef": 2}

# CAM interpolate_mod vocabulary for the interp_outputgri attribute
_OUTPUTGRID = {"cap": "equally spaced with poles",
               "offset": "equally spaced no poles",
               "gauss": "Gauss"}

# native coordinate variables replaced by the lat-lon coordinates
_DROP_NCOL_VARS = {"lat", "lon", "area", "areawt"}
# land mode: landfrac/landmask are recomputed on the destination grid
_DROP_LAND_VARS = {"lat", "lon", "area", "landfrac", "landmask"}
# CLM sub-grid (compressed 1-D) dimensions: not grid fields, skipped
_LAND_NATIVE_DIMS = ("lndgrid", "gridcell", "landunit", "column", "pft")

_ZONAL = re.compile(r"\bzonal\b", re.IGNORECASE)
_MERIDIONAL = re.compile(r"\bmeridional\b", re.IGNORECASE)


def _partner_names(name):
    """Candidate meridional partners for a zonal component name."""
    out = []
    if name.startswith("U"):
        out.append("V" + name[1:])
    if name.endswith("X"):
        out.append(name[:-1] + "Y")
    return out


def detect_vector_pairs(src_ds, names):
    """Auto-detect (zonal, meridional) pairs among remappable variables.
    A pair requires the name pattern (U*<->V* or *X<->*Y), identical
    dimensions, AND matching long_names: either 'zonal'/'meridional'
    (U/V, UBOT/VBOT, TAUX/TAUY, 'Zonal wind tendency ...') or long_names
    identical up to a standalone U<->V word ('U total physics tendency').
    Scalars like U10 ('10m wind speed') are never paired."""
    pairs, used = [], set()
    nameset = set(names)
    for name in names:
        if name in used:
            continue
        var = src_ds.variables[name]
        ln = getattr(var, "long_name", "")
        for pname in _partner_names(name):
            if pname not in nameset or pname in used:
                continue
            pvar = src_ds.variables[pname]
            if pvar.dimensions != var.dimensions:
                continue
            pln = getattr(pvar, "long_name", "")
            ok = (_ZONAL.search(ln) and _MERIDIONAL.search(pln))
            if not ok and ln and re.search(r"\bU\b", ln):
                ok = re.sub(r"\bU\b", "V", ln) == pln
            if ok:
                pairs.append((name, pname))
                used.update((name, pname))
                break
    return pairs


def uv_to_cart(u, v, lam, phi):
    """(u,v) in the local east/north basis -> Cartesian (vx,vy,vz).
    lam, phi: radians, shape (ncol,); u, v: (ncol, nslice)."""
    sl = np.sin(lam)[:, None]
    cl = np.cos(lam)[:, None]
    sp = np.sin(phi)[:, None]
    cp = np.cos(phi)[:, None]
    vx = -sl * u - sp * cl * v
    vy = cl * u - sp * sl * v
    vz = cp * v
    return vx, vy, vz


def cart_to_uv(vx, vy, vz, lam, phi):
    """Project a Cartesian vector onto the local east/north basis (this
    also discards any spurious radial component)."""
    sl = np.sin(lam)[:, None]
    cl = np.cos(lam)[:, None]
    sp = np.sin(phi)[:, None]
    cp = np.cos(phi)[:, None]
    u = -sl * vx + cl * vy
    v = -sp * cl * vx - sp * sl * vy + cp * vz
    return u, v


class RemapEngine:
    """Weight file + kernels, reusable across input files."""

    def __init__(self, weight_path, nthreads=0):
        self.w = read_weights(weight_path)
        self.weight_path = os.path.abspath(weight_path)
        self.src = SourceGrid(n=self.w["n"], geom=self.w["geom"])
        if nthreads:
            fk._lib.cr_set_num_threads(int(nthreads))

    def remap_slices(self, f, col2pij, limiter_mode):
        """f: (ncol, nslice) float64 -> (nb, nslice)."""
        w = self.w
        return fk.apply_slices(w["n"], w["area_b"].size, col2pij, w["halo"],
                               w["geom"], w["pairs"], w["area_b"],
                               limiter_mode, f)

    def remap_slices_masked(self, f, mask, lf, col2pij, limiter_mode,
                            frac_tol, fillval):
        """Masked fracarea remap: f(ncol, nslice) with per-slice validity
        mask and per-cell area fraction lf -> (values, covered fraction),
        both (nb, nslice).  Values are means over the covered portion of
        each destination cell; coverage <= frac_tol*area gets fillval."""
        w = self.w
        return fk.apply_slices_masked(w["n"], w["area_b"].size, col2pij,
                                      w["halo"], w["geom"], w["pairs"],
                                      w["area_b"], limiter_mode, lf, mask,
                                      f, frac_tol=frac_tol, fillval=fillval)

    def remap_file(self, infile, outfile, variables=None,
                   default_limiter="shape", limiter_overrides=None,
                   vectors=None, auto_vectors=True, frac_tol=1e-8,
                   verbose=True):
        lim_over = limiter_overrides or {}
        w = self.w
        nlat, nlon = w["nlat"], w["nlon"]

        src_ds = netCDF4.Dataset(infile, "r")
        try:
            if "ncol" in src_ds.dimensions:
                spatial, land = "ncol", False
            elif "lndgrid" in src_ds.dimensions:
                spatial, land = "lndgrid", True
            else:
                raise ValueError(f"{infile}: no 'ncol' or 'lndgrid' "
                                 "dimension (not a native physgrid or CLM "
                                 "history file)")
            ncol = len(src_ds.dimensions[spatial])
            if ncol != self.src.ncol:
                raise ValueError(
                    f"{infile}: {spatial}={ncol} but weight file is for "
                    f"n={w['n']} (ncol={self.src.ncol})")

            lf = colok = None
            if land:
                if "landfrac" not in src_ds.variables:
                    raise ValueError(f"{infile}: 'lndgrid' file without a "
                                     "'landfrac' variable")
                lf = np.ma.filled(src_ds.variables["landfrac"][:],
                                  0.0).astype(np.float64)
                # unmasked spval (no _FillValue attr) must read as ocean
                lf[~np.isfinite(lf) | (lf < 0.0) | (lf > 1.5)] = 0.0
                lf = np.clip(lf, 0.0, 1.0)
                lonv = src_ds.variables["lon"][:]
                latv = src_ds.variables["lat"][:]
                colok = (~np.ma.getmaskarray(lonv)
                         & ~np.ma.getmaskarray(latv) & (lf > 0.0))
                lf[~colok] = 0.0
                col2pij = self.src.col2pij_partial(
                    np.ma.filled(lonv, 0.0), np.ma.filled(latv, 0.0), colok)
            else:
                col2pij = self.src.col2pij(src_ds.variables["lon"][:],
                                           src_ds.variables["lat"][:])

            fmt = src_ds.data_model
            if fmt == "NETCDF3_CLASSIC":
                fmt = "NETCDF3_64BIT_DATA"
            # Build the output in memory and flush it once at close:
            # python-netCDF4 wraps every variable/attribute define in
            # nc_redef/nc_enddef, and each enddef on a growing netCDF3
            # file relocates the allocated ranges -- O(n^2) in variable
            # count, fatal at CLM's ~600 variables (hours instead of
            # seconds).  Costs RAM equal to the output file size.
            diskless = not fmt.startswith("NETCDF4")
            out = netCDF4.Dataset(outfile, "w", format=fmt,
                                  diskless=diskless, persist=diskless)
            try:
                self._write(src_ds, out, col2pij, variables,
                            default_limiter, lim_over, vectors,
                            auto_vectors, verbose, spatial=spatial,
                            land=land, lf=lf, colok=colok,
                            frac_tol=frac_tol)
            finally:
                out.close()
        finally:
            src_ds.close()

    # ------------------------------------------------------------------
    def _write(self, src_ds, out, col2pij, variables, default_limiter,
               lim_over, vectors, auto_vectors, verbose, spatial="ncol",
               land=False, lf=None, colok=None, frac_tol=1e-8):
        w = self.w
        nlat, nlon = w["nlat"], w["nlon"]

        # --- global attributes: copy, then set interpolation provenance
        attrs = {a: src_ds.getncattr(a) for a in src_ds.ncattrs()}
        attrs["interp_type"] = (
            "CSLAM: conservative, shape-preserving, geometrically exact "
            f"(cslam-regrid {__version__})")
        attrs["interp_outputgri"] = _OUTPUTGRID.get(w["dst_kind"],
                                                    w["dst_kind"])
        attrs["cslam_regrid_map"] = self.weight_path
        if land:
            attrs["cslam_normalization"] = (
                "fracarea: values are landfrac-weighted means over the "
                "land portion of each destination cell; the land-area "
                "integral is conserved")
            attrs["cslam_frac_threshold"] = np.float64(frac_tol)
        stamp = (f"{datetime.datetime.now().strftime('%c')}: cslam-regrid "
                 f"{__version__} remap to {nlat}x{nlon} "
                 f"({w['dst_kind']}), map {os.path.basename(self.weight_path)}")
        attrs["history"] = stamp + "\n" + str(attrs.get("history", ""))
        out.setncatts(attrs)

        # --- dimensions: all except native-grid dimensions, plus lat/lon.
        # CAM: ncol + other ncol_* grids (GLL ncol_d, ncol_fvm, ...);
        # CLM: lndgrid + the compressed sub-grid vectors (gridcell,
        # landunit, column, pft) -- not grid fields, skipped.
        if land:
            native_dims = {d for d in _LAND_NATIVE_DIMS
                           if d in src_ds.dimensions}
        else:
            native_dims = {d for d in src_ds.dimensions
                           if d.startswith("ncol")}
        for name, dim in src_ds.dimensions.items():
            if name in native_dims:
                continue
            out.createDimension(name, None if dim.isunlimited() else len(dim))
        out.createDimension("lat", nlat)
        out.createDimension("lon", nlon)

        # ===== define phase: every variable is created before the first
        # data value is written.  netCDF3 relocates all previously written
        # data on each late define; with CLM's ~550 remappable fields that
        # would dominate the runtime by orders of magnitude.
        v = out.createVariable("lat", "f8", ("lat",))
        v.long_name = "latitude"
        v.units = "degrees_north"
        v = out.createVariable("lon", "f8", ("lon",))
        v.long_name = "longitude"
        v.units = "degrees_east"
        v = out.createVariable("w", "f8", ("lat",))
        v.long_name = "latitude weights"

        # land mode: destination landfrac/landmask (the covered fraction
        # of each lat-lon cell, landfrac-weighted exact overlap)
        if land:
            v = out.createVariable("landfrac", "f8", ("lat", "lon"))
            v.long_name = "land fraction"
            v = out.createVariable("landmask", "i4", ("lat", "lon"))
            v.long_name = "land/ocean mask (0.=ocean and 1.=land)"

        # --- classify variables; define the copy-through ones
        drop_vars = _DROP_LAND_VARS if land else _DROP_NCOL_VARS
        remap_vars, skipped, copy_vars = [], [], []
        for name, var in src_ds.variables.items():
            if spatial in var.dimensions:
                if name in drop_vars:
                    continue
                if var.dimensions[-1] != spatial or \
                        var.dtype.kind not in "fd":
                    skipped.append(name)
                    continue
                if variables is None or name in variables:
                    remap_vars.append(name)
            else:
                unknown = [d for d in var.dimensions
                           if d not in out.dimensions]
                if unknown:
                    skipped.append(name)
                    continue
                nv = out.createVariable(name, var.dtype, var.dimensions)
                nv.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
                copy_vars.append((nv, var))
        if variables is not None:
            missing = set(variables) - set(remap_vars)
            if missing:
                raise ValueError(f"requested variables not remappable or "
                                 f"not found: {sorted(missing)}")
        if skipped and verbose:
            print(f"  skipped (not on the physics grid or non-float): "
                  f"{' '.join(sorted(skipped))}")

        # --- vector pairs: explicit first, then auto-detected
        # (not supported on masked/land files: CLM history has no true
        # vector pairs on the grid)
        pairs = []
        paired = set()
        if land:
            if vectors:
                raise ValueError("vector pairs are not supported for "
                                 "masked (land) files")
        else:
            for ua, va in (vectors or []):
                if ua not in remap_vars or va not in remap_vars:
                    raise ValueError(f"vector pair {ua}:{va} not among the "
                                     "remappable variables")
                if src_ds.variables[ua].dimensions != \
                        src_ds.variables[va].dimensions:
                    raise ValueError(
                        f"vector pair {ua}:{va} dimension mismatch")
                pairs.append((ua, va))
                paired.update((ua, va))
            if auto_vectors:
                for ua, va in detect_vector_pairs(src_ds, remap_vars):
                    if ua not in paired and va not in paired:
                        pairs.append((ua, va))
                        paired.update((ua, va))
        if pairs and verbose:
            print("  vector pairs: "
                  + " ".join(f"{a}:{b}" for a, b in pairs))

        vec_jobs = [self._define_vector(src_ds, out, ua, va, lim_over)
                    for ua, va in pairs]

        # --- define the remapped scalar variables
        scal_jobs = []
        for name in remap_vars:
            if name in paired:
                continue
            var = src_ds.variables[name]
            dims = var.dimensions[:-1] + ("lat", "lon")
            fill = getattr(var, "_FillValue", None)
            if land and fill is None:
                fill = netCDF4.default_fillvals[
                    f"{var.dtype.kind}{var.dtype.itemsize}"]
            nv = out.createVariable(name, var.dtype, dims,
                                    fill_value=fill)
            nv.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                          if a != "_FillValue"})
            mode_name = lim_over.get(name, default_limiter)
            nv.cslam_limiter = mode_name
            scal_jobs.append((name, var, nv, mode_name, fill))

        # ===== write phase =====
        out.variables["lat"][:] = w["lat"]
        out.variables["lon"][:] = w["lon"]
        out.variables["w"][:] = w["w"]
        if land:
            _, frac = self.remap_slices_masked(
                np.ones(self.src.ncol), colok.astype(np.int32), lf,
                col2pij, 0, frac_tol, np.nan)
            frac2d = frac[:, 0].reshape(nlat, nlon)
            out.variables["landfrac"][:] = frac2d
            out.variables["landmask"][:] = \
                (frac2d > frac_tol).astype(np.int32)
        for nv, var in copy_vars:
            if var.shape:
                nv[:] = var[:]
            else:
                nv.assignValue(var[:])

        if pairs:
            lam_a = np.deg2rad(np.asarray(src_ds.variables["lon"][:],
                                          dtype=np.float64))
            phi_a = np.deg2rad(np.asarray(src_ds.variables["lat"][:],
                                          dtype=np.float64))
            lam_b = np.tile(np.deg2rad(w["lon"]), nlat)
            phi_b = np.repeat(np.deg2rad(w["lat"]), nlon)
        for (ua, va), (nvs, mode_name) in zip(pairs, vec_jobs):
            self._write_vector(src_ds, nvs, col2pij, ua, va, mode_name,
                               lam_a, phi_a, lam_b, phi_b, nlat, nlon,
                               verbose)

        for name, var, nv, mode_name, fill in scal_jobs:
            mode = LIMITER_MODES[mode_name]
            if land:
                def slab(a, _mode=mode, _fill=float(fill)):
                    return self._remap_land_slab(a, colok, lf, col2pij,
                                                 _mode, frac_tol, _fill,
                                                 nlat, nlon)
            else:
                def slab(a, _mode=mode):
                    data = np.asarray(a, dtype=np.float64)
                    flat = data.reshape(-1, self.src.ncol).T
                    res = self.remap_slices(flat, col2pij, _mode)
                    return res.T.reshape(data.shape[:-1] + (nlat, nlon))

            if var.dimensions and var.dimensions[0] == "time":
                for t in range(var.shape[0]):
                    nv[t] = slab(var[t])
            else:
                nv[:] = slab(var[:])
            if verbose:
                print(f"  {name} ({mode_name})")

    # ------------------------------------------------------------------
    def _remap_land_slab(self, a, colok, lf, col2pij, mode, frac_tol,
                         fillval, nlat, nlon):
        """One land slab (..., ncol) -> (..., nlat, nlon).  The validity
        mask comes from the slab's own fill pattern (per level/record;
        CLM valid sets are field- and time-dependent), intersected with
        the columns that have coordinates; fracarea normalization."""
        a = np.ma.masked_invalid(a, copy=False)
        valid = ~np.ma.getmaskarray(a)
        data = np.ma.filled(a, 0.0).astype(np.float64)
        shape = data.shape
        flat = data.reshape(-1, self.src.ncol).T
        vflat = (valid.reshape(-1, self.src.ncol).T
                 & colok[:, None]).astype(np.int32)
        res, _ = self.remap_slices_masked(flat, vflat, lf, col2pij, mode,
                                          frac_tol, fillval)
        return res.T.reshape(shape[:-1] + (nlat, nlon))

    # ------------------------------------------------------------------
    def _define_vector(self, src_ds, out, ua, va, lim_over):
        """Define phase for one (zonal, meridional) pair."""
        dims = src_ds.variables[ua].dimensions[:-1] + ("lat", "lon")
        # vectors default to unlimited: wind components are not bounded
        # quantities and componentwise Cartesian bounds are frame-dependent
        mode_name = lim_over.get(ua, lim_over.get(va, "none"))
        nvs = []
        for name in (ua, va):
            var = src_ds.variables[name]
            fill = getattr(var, "_FillValue", None)
            nv = out.createVariable(var.name, var.dtype, dims,
                                    fill_value=fill)
            nv.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                          if a != "_FillValue"})
            nv.cslam_limiter = mode_name
            nv.cslam_vector_pair = f"{ua}:{va}"
            nvs.append(nv)
        return nvs, mode_name

    def _write_vector(self, src_ds, nvs, col2pij, ua, va, mode_name,
                      lam_a, phi_a, lam_b, phi_b, nlat, nlon, verbose):
        """Remap one (zonal, meridional) pair via Cartesian components."""
        varu = src_ds.variables[ua]
        varv = src_ds.variables[va]
        mode = LIMITER_MODES[mode_name]

        def do_slab(u, v):
            shp = u.shape
            uf = u.reshape(-1, self.src.ncol).T
            vf = v.reshape(-1, self.src.ncol).T
            nsl = uf.shape[1]
            vx, vy, vz = uv_to_cart(uf, vf, lam_a, phi_a)
            r = self.remap_slices(np.concatenate([vx, vy, vz], axis=1),
                                  col2pij, mode)
            ub, vb = cart_to_uv(r[:, :nsl], r[:, nsl:2 * nsl],
                                r[:, 2 * nsl:], lam_b, phi_b)
            return (ub.T.reshape(shp[:-1] + (nlat, nlon)),
                    vb.T.reshape(shp[:-1] + (nlat, nlon)))

        if varu.dimensions and varu.dimensions[0] == "time":
            for t in range(varu.shape[0]):
                ub, vb = do_slab(np.asarray(varu[t], dtype=np.float64),
                                 np.asarray(varv[t], dtype=np.float64))
                nvs[0][t] = ub
                nvs[1][t] = vb
        else:
            ub, vb = do_slab(np.asarray(varu[:], dtype=np.float64),
                             np.asarray(varv[:], dtype=np.float64))
            nvs[0][:] = ub
            nvs[1][:] = vb
        if verbose:
            print(f"  {ua}+{va} (vector, {mode_name})")
