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

    def remap_file(self, infile, outfile, variables=None,
                   default_limiter="shape", limiter_overrides=None,
                   vectors=None, auto_vectors=True, verbose=True):
        lim_over = limiter_overrides or {}
        w = self.w
        nlat, nlon = w["nlat"], w["nlon"]

        src_ds = netCDF4.Dataset(infile, "r")
        try:
            if "ncol" not in src_ds.dimensions:
                raise ValueError(f"{infile}: no 'ncol' dimension "
                                 "(not a native physgrid history file)")
            ncol = len(src_ds.dimensions["ncol"])
            if ncol != self.src.ncol:
                raise ValueError(
                    f"{infile}: ncol={ncol} but weight file is for "
                    f"n={w['n']} (ncol={self.src.ncol})")
            col2pij = self.src.col2pij(src_ds.variables["lon"][:],
                                       src_ds.variables["lat"][:])

            fmt = src_ds.data_model
            if fmt == "NETCDF3_CLASSIC":
                fmt = "NETCDF3_64BIT_DATA"
            out = netCDF4.Dataset(outfile, "w", format=fmt)
            try:
                self._write(src_ds, out, col2pij, variables,
                            default_limiter, lim_over, vectors,
                            auto_vectors, verbose)
            finally:
                out.close()
        finally:
            src_ds.close()

    # ------------------------------------------------------------------
    def _write(self, src_ds, out, col2pij, variables, default_limiter,
               lim_over, vectors, auto_vectors, verbose):
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
        stamp = (f"{datetime.datetime.now().strftime('%c')}: cslam-regrid "
                 f"{__version__} remap to {nlat}x{nlon} "
                 f"({w['dst_kind']}), map {os.path.basename(self.weight_path)}")
        attrs["history"] = stamp + "\n" + str(attrs.get("history", ""))
        out.setncatts(attrs)

        # --- dimensions: all except ncol and other native-grid dimensions
        # (ncol_d GLL, ncol_fvm, ...: their variables cannot share the
        # lat-lon grid and are skipped), plus lat/lon
        native_dims = {d for d in src_ds.dimensions if d.startswith("ncol")}
        for name, dim in src_ds.dimensions.items():
            if name in native_dims:
                continue
            out.createDimension(name, None if dim.isunlimited() else len(dim))
        out.createDimension("lat", nlat)
        out.createDimension("lon", nlon)

        v = out.createVariable("lat", "f8", ("lat",))
        v.long_name = "latitude"
        v.units = "degrees_north"
        v[:] = w["lat"]
        v = out.createVariable("lon", "f8", ("lon",))
        v.long_name = "longitude"
        v.units = "degrees_east"
        v[:] = w["lon"]
        v = out.createVariable("w", "f8", ("lat",))
        v.long_name = "latitude weights"
        v[:] = w["w"]

        # --- classify variables
        remap_vars, skipped = [], []
        for name, var in src_ds.variables.items():
            if "ncol" in var.dimensions:
                if name in _DROP_NCOL_VARS:
                    continue
                if var.dimensions[-1] != "ncol" or \
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
                if var.shape:
                    nv[:] = var[:]
                else:
                    nv.assignValue(var[:])
        if variables is not None:
            missing = set(variables) - set(remap_vars)
            if missing:
                raise ValueError(f"requested variables not remappable or "
                                 f"not found: {sorted(missing)}")
        if skipped and verbose:
            print(f"  skipped (not on the physics grid or non-float): "
                  f"{' '.join(sorted(skipped))}")

        # --- vector pairs: explicit first, then auto-detected
        pairs = []
        paired = set()
        for ua, va in (vectors or []):
            if ua not in remap_vars or va not in remap_vars:
                raise ValueError(f"vector pair {ua}:{va} not among the "
                                 "remappable variables")
            if src_ds.variables[ua].dimensions != \
                    src_ds.variables[va].dimensions:
                raise ValueError(f"vector pair {ua}:{va} dimension mismatch")
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

        lam_a = np.deg2rad(np.asarray(src_ds.variables["lon"][:],
                                      dtype=np.float64))
        phi_a = np.deg2rad(np.asarray(src_ds.variables["lat"][:],
                                      dtype=np.float64))
        lam_b = np.tile(np.deg2rad(w["lon"]), nlat)
        phi_b = np.repeat(np.deg2rad(w["lat"]), nlon)

        for ua, va in pairs:
            self._write_vector(src_ds, out, col2pij, ua, va, lim_over,
                               lam_a, phi_a, lam_b, phi_b, nlat, nlon,
                               verbose)

        # --- remapped scalar variables
        for name in remap_vars:
            if name in paired:
                continue
            var = src_ds.variables[name]
            dims = var.dimensions[:-1] + ("lat", "lon")
            fill = getattr(var, "_FillValue", None)
            nv = out.createVariable(name, var.dtype, dims,
                                    fill_value=fill)
            nv.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                          if a != "_FillValue"})
            mode_name = lim_over.get(name, default_limiter)
            nv.cslam_limiter = mode_name
            mode = LIMITER_MODES[mode_name]

            tdim = var.dimensions[0] if var.dimensions and \
                var.dimensions[0] == "time" else None
            if tdim:
                nt = var.shape[0]
                for t in range(nt):
                    data = np.asarray(var[t], dtype=np.float64)
                    flat = data.reshape(-1, self.src.ncol).T
                    res = self.remap_slices(flat, col2pij, mode)
                    res = res.T.reshape(data.shape[:-1] + (nlat, nlon))
                    nv[t] = res
            else:
                data = np.asarray(var[:], dtype=np.float64)
                flat = data.reshape(-1, self.src.ncol).T
                res = self.remap_slices(flat, col2pij, mode)
                nv[:] = res.T.reshape(data.shape[:-1] + (nlat, nlon))
            if verbose:
                print(f"  {name} ({mode_name})")

    # ------------------------------------------------------------------
    def _write_vector(self, src_ds, out, col2pij, ua, va, lim_over,
                      lam_a, phi_a, lam_b, phi_b, nlat, nlon, verbose):
        """Remap one (zonal, meridional) pair via Cartesian components."""
        varu = src_ds.variables[ua]
        varv = src_ds.variables[va]
        dims = varu.dimensions[:-1] + ("lat", "lon")
        # vectors default to unlimited: wind components are not bounded
        # quantities and componentwise Cartesian bounds are frame-dependent
        mode_name = lim_over.get(ua, lim_over.get(va, "none"))
        mode = LIMITER_MODES[mode_name]

        nvs = []
        for var in (varu, varv):
            fill = getattr(var, "_FillValue", None)
            nv = out.createVariable(var.name, var.dtype, dims,
                                    fill_value=fill)
            nv.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                          if a != "_FillValue"})
            nv.cslam_limiter = mode_name
            nv.cslam_vector_pair = f"{ua}:{va}"
            nvs.append(nv)

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
