"""Apply stage: remap CAM native (ncol) history files to lat-lon, writing
output that mimics CAM's own interpolated lat-lon history files."""
from __future__ import annotations

import datetime
import os

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
_DROP_NCOL_VARS = {"lat", "lon", "area"}


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
                   verbose=True):
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
                            default_limiter, lim_over, verbose)
            finally:
                out.close()
        finally:
            src_ds.close()

    # ------------------------------------------------------------------
    def _write(self, src_ds, out, col2pij, variables, default_limiter,
               lim_over, verbose):
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

        # --- dimensions: all except ncol, plus lat/lon
        for name, dim in src_ds.dimensions.items():
            if name == "ncol":
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

        # --- remapped variables
        for name in remap_vars:
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
