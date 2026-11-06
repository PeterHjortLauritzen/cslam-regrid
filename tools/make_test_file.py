#!/usr/bin/env python3
"""Create a synthetic native ne*pg3 CAM history file (dimension ncol) with
CAM-like structure and analytic fields, for testing cslam-regrid apply.

Usage: python tools/make_test_file.py out.nc [--ne 30] [--nlev 32] [--ntime 2]
"""
import argparse
import os
import sys

import numpy as np
import netCDF4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cslam_regrid import fkernels as fk           # noqa: E402
from cslam_regrid.grids import SourceGrid         # noqa: E402


def cell_latlon(src):
    """lon/lat (degrees) of every cell centroid, synthetic ncol ordering."""
    n = src.n
    cx = src.geom["centroid"][0]          # (n, n)
    cy = src.geom["centroid"][1]
    lon = np.empty(6 * n * n)
    lat = np.empty(6 * n * n)
    for p in range(1, 7):
        lo, la = fk.lonlat_from_panel(cx.reshape(-1, order="F"),
                                      cy.reshape(-1, order="F"),
                                      np.full(n * n, p, np.int32))
        s = (p - 1) * n * n
        lon[s:s + n * n] = np.rad2deg(lo)
        lat[s:s + n * n] = np.rad2deg(la)
    return lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ne", type=int, default=30)
    ap.add_argument("--nlev", type=int, default=32)
    ap.add_argument("--ntime", type=int, default=2)
    args = ap.parse_args()

    src = SourceGrid.from_ne(args.ne)
    lon, lat = cell_latlon(src)
    lonr, latr = np.deg2rad(lon), np.deg2rad(lat)
    ncol = lon.size
    nlev = args.nlev

    ds = netCDF4.Dataset(args.out, "w", format="NETCDF3_64BIT_DATA")
    ds.Conventions = "CF-1.0"
    ds.source = "CAM"
    ds.case = "cslam-regrid synthetic test"
    ds.logname = "test"
    ds.host = "synthetic"
    ds.time_period_freq = "month_1"

    ds.createDimension("ncol", ncol)
    ds.createDimension("time", None)
    ds.createDimension("nbnd", 2)
    ds.createDimension("lev", nlev)
    ds.createDimension("ilev", nlev + 1)

    def var(name, dtype, dims, **attrs):
        v = ds.createVariable(name, dtype, dims)
        for k, val in attrs.items():
            setattr(v, k, val)
        return v

    v = var("lat", "f8", ("ncol",), long_name="latitude",
            units="degrees_north"); v[:] = lat
    v = var("lon", "f8", ("ncol",), long_name="longitude",
            units="degrees_east"); v[:] = lon
    v = var("area", "f8", ("ncol",), long_name="physics grid areas",
            units="steradian")
    v[:] = np.tile(src.geom["area"].reshape(-1, order="F"), 6)

    hybm = np.linspace(0.01, 0.99, nlev)
    v = var("lev", "f8", ("lev",), long_name="hybrid level at midpoints "
            "(1000*(A+B))", units="hPa", positive="down")
    v[:] = 1000.0 * hybm
    v = var("hyam", "f8", ("lev",), long_name="hybrid A coefficient at "
            "layer midpoints", units="1"); v[:] = 0.0 * hybm
    v = var("hybm", "f8", ("lev",), long_name="hybrid B coefficient at "
            "layer midpoints", units="1"); v[:] = hybm
    hybi = np.linspace(0.0, 1.0, nlev + 1)
    v = var("ilev", "f8", ("ilev",), long_name="hybrid level at interfaces "
            "(1000*(A+B))", units="hPa", positive="down")
    v[:] = 1000.0 * hybi
    v = var("hyai", "f8", ("ilev",), units="1"); v[:] = 0.0 * hybi
    v = var("hybi", "f8", ("ilev",), units="1"); v[:] = hybi

    v = var("time", "f8", ("time",), long_name="time",
            units="days since 1979-01-01 00:00:00", calendar="noleap",
            bounds="time_bounds")
    v[:] = np.arange(args.ntime, dtype=float)
    v = var("time_bounds", "f8", ("time", "nbnd"),
            long_name="time interval endpoints")
    v[:] = np.stack([np.arange(args.ntime) - 1.0,
                     np.arange(args.ntime)], axis=1)
    v = var("date", "i4", ("time",), long_name="current date (YYYYMMDD)")
    v[:] = 19790101 + np.arange(args.ntime)
    v = var("datesec", "i4", ("time",),
            long_name="current seconds of current date")
    v[:] = np.zeros(args.ntime, np.int32)
    v = var("mdt", "i4", (), long_name="timestep", units="s")
    v.assignValue(1800)

    # analytic fields (functions of lon/lat evaluated at centroids)
    bell = np.exp(-((latr - np.deg2rad(35.26)) ** 2
                    + (lonr - np.deg2rad(45.0)) ** 2) / 0.05)
    step = (latr > np.deg2rad(70)).astype(float)

    v = var("PS", "f4", ("time", "ncol"), units="Pa",
            long_name="Surface pressure", cell_methods="time: mean")
    v = ds.variables["PS"]
    for t in range(args.ntime):
        v[t] = 1.0e5 + 1000.0 * np.cos(latr) ** 2 * np.cos(2 * lonr + 0.1 * t)

    v = var("TS", "f4", ("time", "ncol"), units="K",
            long_name="Surface temperature", cell_methods="time: mean")
    for t in range(args.ntime):
        v[t] = 288.0 - 60.0 * np.sin(latr) ** 2 + 10.0 * bell

    v = var("PRECT", "f4", ("time", "ncol"), units="m/s",
            long_name="Total precipitation rate", cell_methods="time: mean")
    for t in range(args.ntime):
        v[t] = np.maximum(0.0, 5e-8 * (bell + step - 0.3))

    v = var("T", "f4", ("time", "lev", "ncol"), units="K", mdims=np.int32(1),
            long_name="Temperature", cell_methods="time: mean")
    prof = 220.0 + 70.0 * hybm
    for t in range(args.ntime):
        v[t] = (prof[:, None]
                - 40.0 * np.sin(latr[None, :]) ** 2
                + 5.0 * np.cos(2 * lonr[None, :]) * np.cos(latr[None, :]))

    v = var("Q", "f4", ("time", "lev", "ncol"), units="kg/kg",
            mdims=np.int32(1), long_name="Specific humidity",
            cell_methods="time: mean")
    for t in range(args.ntime):
        v[t] = np.maximum(
            1e-9, 0.02 * hybm[:, None] ** 2 * np.cos(latr[None, :]) ** 4
            * (1.0 + 0.5 * np.cos(3 * lonr[None, :])) - 1e-4)

    ds.close()
    print(f"wrote {args.out}: ncol={ncol}, nlev={nlev}, ntime={args.ntime}")


if __name__ == "__main__":
    main()
