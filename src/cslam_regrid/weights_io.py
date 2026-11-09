"""Self-describing netCDF weight file: everything the apply stage needs.

Format version 1.  The weight file is pure exact geometry and is shared by
atmosphere (CAM) and land (CLM) files alike: masking/fracarea for land data
is dynamic at apply time (CLM valid sets are field- and time-dependent), so
grid_imask_a/frac_b stay at their unmasked values.  Array index conventions
are Fortran 1-based, as consumed by libcslamregrid; dtypes are preserved
exactly (int32 / float64).
"""
from __future__ import annotations

import datetime
import getpass
import socket

import numpy as np
import netCDF4

from . import __version__
from .grids import DestGrid, SourceGrid

FORMAT_VERSION = 1


def write_weights(path, src: SourceGrid, dst: DestGrid, halo, pairs, diags,
                  ne=None, history=""):
    n = src.n
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.title = "cslam-regrid mapping weights"
        ds.format_version = FORMAT_VERSION
        ds.cslam_regrid_version = __version__
        ds.created = datetime.datetime.now().isoformat()
        ds.created_by = getpass.getuser()
        ds.host = socket.gethostname()
        ds.method = ("CSLAM quadratic reconstruction (irecons=6), "
                     "shape-preserving limiter available at apply time, "
                     "geometrically exact overlap moments (GECoRe method)")
        ds.source_grid = f"ne{ne}pg{n // ne if ne else 3} (n={n})" if ne \
            else f"gnomonic cubed-sphere physics grid, n={n}"
        ds.dest_grid = f"{dst.kind} {dst.nlat}x{dst.nlon}"
        ds.dst_area_abserr = diags.get("dst_area_abserr", -1.0)
        ds.src_moment_relerr = diags.get("src_moment_relerr", -1.0)
        ds.history = history

        ds.src_n = np.int32(n)
        if ne:
            ds.src_ne = np.int32(ne)
        ds.dst_kind = dst.kind

        # --- dimensions
        npair = pairs["dst"].size
        dims = dict(n=n, np1=n + 1, na=6 * n * n, nb=dst.ncell,
                    nlat=dst.nlat, nlon=dst.nlon, nlatp1=dst.nlat + 1,
                    nlonp1=dst.nlon + 1, n_s=npair, nmom=6, three=3, ns=3,
                    four=4, five=5, seven=7, nhi=halo["hi_dst"].shape[1],
                    nli=halo["li_dst"].shape[1])
        for name, size in dims.items():
            ds.createDimension(name, size)

        def put(name, data, dimnames, dtype=None):
            data = np.asarray(data)
            v = ds.createVariable(name, dtype or data.dtype, dimnames,
                                  zlib=True, complevel=1)
            v[:] = data
            return v

        g = src.geom
        put("xe", g["xe"], ("np1",))
        put("xc", g["xc"], ("n",))
        put("area_a", g["area"], ("n", "n"))
        put("centroid", g["centroid"], ("five", "n", "n"))
        put("metrics", g["metrics"], ("three", "n", "n"))
        put("metrics_int", g["metrics_int"], ("three", "n", "n"))
        put("stretch", g["stretch"], ("seven", "n", "n"))
        put("vtxw", g["vtxw"], ("four", "five", "n", "n"))

        put("hi_dst", halo["hi_dst"], ("three", "nhi"))
        put("hi_src", halo["hi_src"], ("three", "three", "nhi"))
        put("hi_w", halo["hi_w"], ("ns", "nhi"))
        put("li_dst", halo["li_dst"], ("three", "nli"))
        put("li_src", halo["li_src"], ("three", "nli"))

        put("pr_src", pairs["src"], ("three", "n_s"))
        put("pr_dst", pairs["dst"], ("n_s",))
        put("S", pairs["S"], ("nmom", "n_s"))

        v = put("lat", np.rad2deg(dst.lat_centers), ("nlat",))
        v.units = "degrees_north"
        v = put("lon", np.rad2deg(dst.lon_centers), ("nlon",))
        v.units = "degrees_east"
        put("lat_edges", np.rad2deg(dst.lat_edges), ("nlatp1",))
        put("lon_edges", np.rad2deg(dst.lon_edges), ("nlonp1",))
        v = put("area_b", dst.areas(), ("nb",))
        v.units = "steradian"
        v = put("w", dst.lat_weights(), ("nlat",))
        v.long_name = "latitude weights"

        # unmasked by construction: land/masked data is handled dynamically
        # at apply time (per-field validity + landfrac), weights stay pure
        # exact geometry
        v = put("grid_imask_a", np.ones(6 * n * n, dtype=np.int32), ("na",))
        v.comment = "source mask; always 1 (masking is dynamic at apply time)"
        v = ds.createVariable("frac_b", "f8", ("nb",), zlib=True, complevel=1)
        v[:] = 1.0
        v.comment = ("destination fraction; always 1 (fracarea for land "
                     "data is dynamic at apply time)")
    finally:
        ds.close()


def read_weights(path):
    ds = netCDF4.Dataset(path, "r")
    try:
        if int(ds.format_version) != FORMAT_VERSION:
            raise ValueError(f"weight file format {ds.format_version} "
                             f"unsupported (expected {FORMAT_VERSION})")
        out = dict(
            n=int(ds.src_n),
            ne=int(getattr(ds, "src_ne", 0)) or None,
            dst_kind=str(ds.dst_kind),
            attrs={a: getattr(ds, a) for a in ds.ncattrs()},
        )
        for name in ("xe", "xc", "area_a", "centroid", "metrics",
                     "metrics_int", "stretch", "vtxw", "hi_dst", "hi_src",
                     "hi_w", "li_dst", "li_src", "pr_src", "pr_dst", "S",
                     "lat", "lon", "lat_edges", "lon_edges", "area_b", "w",
                     "frac_b"):
            out[name] = ds.variables[name][:].filled() \
                if hasattr(ds.variables[name][:], "filled") \
                else np.asarray(ds.variables[name][:])
        out["geom"] = {k: out[k] for k in
                       ("xe", "xc", "centroid", "metrics", "metrics_int",
                        "stretch", "vtxw")}
        out["geom"]["area"] = out["area_a"]
        out["halo"] = {k: out[k] for k in
                       ("hi_dst", "hi_src", "hi_w", "li_dst", "li_src")}
        out["pairs"] = dict(src=out["pr_src"], dst=out["pr_dst"], S=out["S"])
        out["nlat"] = out["lat"].size
        out["nlon"] = out["lon"].size
        return out
    finally:
        ds.close()
