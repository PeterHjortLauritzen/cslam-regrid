"""ctypes bindings to libcslamregrid.so (the Fortran kernels)."""
from __future__ import annotations

import ctypes as ct
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBPATH = os.environ.get("CSLAM_REGRID_LIB", os.path.join(_HERE, "libcslamregrid.so"))

_c_dbl = ct.c_double
_c_int = ct.c_int
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")


def _load():
    lib = ct.CDLL(_LIBPATH)

    lib.cr_lonlat_from_panel.argtypes = [_c_int, _dp, _dp, _ip, _dp, _dp]
    lib.cr_panel_from_lonlat.argtypes = [_c_int, _dp, _dp, _dp, _dp, _ip]
    lib.cr_cart_from_panel.argtypes = [_c_int, _dp, _dp, _ip, _dp]
    lib.cr_panel_point_from_lonlat.argtypes = [_c_int, _dp, _dp, _c_int, _dp, _dp, _ip]

    for name in ("cr_seg_w6_horizontal", "cr_seg_w6_parallel_eq",
                 "cr_seg_w6_meridian_po"):
        getattr(lib, name).argtypes = [_c_dbl, _c_dbl, _c_dbl, _dp]
    lib.cr_seg_w6_parallel_po.argtypes = [_c_dbl, _c_dbl, _c_dbl, _c_dbl, _dp]
    lib.cr_cell_moments.argtypes = [_c_dbl, _c_dbl, _c_dbl, _c_dbl, _dp]

    lib.cr_grid_geometry.argtypes = [_c_int] + [_dp]*8
    lib.cr_slice_recon.argtypes = [
        _c_int, _c_int, _ip,                 # n, ncol, col2pij
        _c_int, _ip, _ip, _dp,               # nhi, hi_dst, hi_src, hi_w
        _c_int, _ip, _ip,                    # nli, li_dst, li_src
        _dp, _dp, _dp, _dp, _dp, _dp,        # geometry
        _c_int, _dp, _dp, _dp, _dp,          # limiter, f, fext, flim, rec
    ]
    lib.cr_apply_slices.argtypes = [
        _c_int, _c_int, _c_int, _c_int, _ip,  # n, ncol, nslice, nb, col2pij
        _c_int, _ip, _ip, _dp,                # nhi, hi_dst, hi_src, hi_w
        _c_int, _ip, _ip,                     # nli, li_dst, li_src
        _dp, _dp, _dp, _dp, _dp, _dp,         # geometry
        _c_int, _ip, _ip, _dp, _dp,           # npair, pr_src, pr_dst, smat, area_b
        _c_int, _dp, _dp,                     # limiter, f_in, f_out
    ]
    lib.cr_apply_slices_masked.argtypes = [
        _c_int, _c_int, _c_int, _c_int, _ip,  # n, ncol, nslice, nb, col2pij
        _c_int, _ip, _ip, _dp,                # nhi, hi_dst, hi_src, hi_w
        _c_int, _ip, _ip,                     # nli, li_dst, li_src
        _dp, _dp, _dp, _dp, _dp, _dp,         # geometry
        _c_int, _ip, _ip, _dp, _dp,           # npair, pr_src, pr_dst, smat, area_b
        _c_int, _c_dbl, _c_dbl,               # limiter, frac_tol, fillval
        _dp, _ip, _dp, _dp, _dp,              # lf, mask, f_in, f_out, frac_out
    ]

    for fn in (lib.cr_lonlat_from_panel, lib.cr_panel_from_lonlat,
               lib.cr_cart_from_panel, lib.cr_panel_point_from_lonlat,
               lib.cr_seg_w6_horizontal, lib.cr_seg_w6_parallel_eq,
               lib.cr_seg_w6_parallel_po, lib.cr_seg_w6_meridian_po,
               lib.cr_cell_moments, lib.cr_grid_geometry,
               lib.cr_slice_recon, lib.cr_apply_slices,
               lib.cr_apply_slices_masked):
        fn.restype = None
    return lib


_lib = _load()


def _asf(a, dtype=np.float64):
    return np.asfortranarray(np.atleast_1d(np.asarray(a, dtype=dtype)))


def lonlat_from_panel(x, y, panel):
    x, y = _asf(x), _asf(y)
    p = _asf(np.broadcast_to(panel, x.shape), np.int32)
    lon = np.zeros_like(x)
    lat = np.zeros_like(x)
    _lib.cr_lonlat_from_panel(x.size, x, y, p, lon, lat)
    return lon, lat


def panel_from_lonlat(lon, lat):
    lon, lat = _asf(lon), _asf(lat)
    x = np.zeros_like(lon)
    y = np.zeros_like(lon)
    p = np.zeros(lon.shape, dtype=np.int32, order="F")
    _lib.cr_panel_from_lonlat(lon.size, lon, lat, x, y, p)
    return x, y, p


def cart_from_panel(x, y, panel):
    x, y = _asf(x), _asf(y)
    p = _asf(np.broadcast_to(panel, x.shape), np.int32)
    cart = np.zeros((3, x.size), dtype=np.float64, order="F")
    _lib.cr_cart_from_panel(x.size, x, y, p, cart)
    return cart


def panel_point_from_lonlat(lon, lat, panel):
    lon, lat = _asf(lon), _asf(lat)
    x = np.zeros_like(lon)
    y = np.zeros_like(lon)
    valid = np.zeros(lon.shape, dtype=np.int32, order="F")
    _lib.cr_panel_point_from_lonlat(lon.size, lon, lat, int(panel), x, y, valid)
    return x, y, valid.astype(bool)


def _seg(fn, *args):
    w = np.zeros(6, dtype=np.float64, order="F")
    fn(*args, w)
    return w


def seg_w6_horizontal(c, xa, xb):
    return _seg(_lib.cr_seg_w6_horizontal, c, xa, xb)


def seg_w6_parallel_eq(t, xa, xb):
    return _seg(_lib.cr_seg_w6_parallel_eq, t, xa, xb)


def seg_w6_parallel_po(r, sbranch, xa, xb):
    return _seg(_lib.cr_seg_w6_parallel_po, r, float(sbranch), xa, xb)


def seg_w6_meridian_po(m, xa, xb):
    return _seg(_lib.cr_seg_w6_meridian_po, m, xa, xb)


def cell_moments(x1, x2, y1, y2):
    mom = np.zeros(6, dtype=np.float64, order="F")
    _lib.cr_cell_moments(x1, x2, y1, y2, mom)
    return mom


def grid_geometry(n):
    """Per-cell panel geometry (identical on all six panels)."""
    F = dict(dtype=np.float64, order="F")
    out = dict(
        xe=np.zeros(n + 1, **F),
        xc=np.zeros(n, **F),
        area=np.zeros((n, n), **F),
        centroid=np.zeros((5, n, n), **F),
        metrics=np.zeros((3, n, n), **F),
        metrics_int=np.zeros((3, n, n), **F),
        stretch=np.zeros((7, n, n), **F),
        vtxw=np.zeros((4, 5, n, n), **F),
    )
    _lib.cr_grid_geometry(n, out["xe"], out["xc"], out["area"],
                          out["centroid"], out["metrics"], out["metrics_int"],
                          out["stretch"], out["vtxw"])
    return out


def _fi(a):
    return np.asfortranarray(np.asarray(a, dtype=np.int32))


def _fd(a):
    return np.asfortranarray(np.asarray(a, dtype=np.float64))


def slice_recon(n, col2pij, halo, geom, limiter_mode, f_slice):
    """Reconstruction coefficients + halo arrays for one ncol slice."""
    ncol = f_slice.size
    fext = np.zeros((n + 4, n + 4, 6), dtype=np.float64, order="F")
    flim = np.zeros((n + 2, n + 2, 6), dtype=np.float64, order="F")
    rec = np.zeros((6, n, n, 6), dtype=np.float64, order="F")
    _lib.cr_slice_recon(
        n, ncol, _fi(col2pij),
        halo["hi_dst"].shape[1], _fi(halo["hi_dst"]), _fi(halo["hi_src"]),
        _fd(halo["hi_w"]),
        halo["li_dst"].shape[1], _fi(halo["li_dst"]), _fi(halo["li_src"]),
        _fd(geom["stretch"]), _fd(geom["centroid"]), _fd(geom["metrics"]),
        _fd(geom["metrics_int"]), _fd(geom["vtxw"]), _fd(geom["xe"]),
        int(limiter_mode), _fd(f_slice), fext, flim, rec)
    return fext, flim, rec


def apply_slices(n, nb, col2pij, halo, geom, pairs, area_b, limiter_mode, f_in):
    """Remap f_in(ncol, nslice) -> (nb, nslice)."""
    f_in = np.asarray(f_in, dtype=np.float64)
    if f_in.ndim == 1:
        f_in = f_in[:, None]
    f_in = np.asfortranarray(f_in)
    ncol, nslice = f_in.shape
    f_out = np.zeros((nb, nslice), dtype=np.float64, order="F")
    npair = pairs["dst"].size
    _lib.cr_apply_slices(
        n, ncol, nslice, nb, _fi(col2pij),
        halo["hi_dst"].shape[1], _fi(halo["hi_dst"]), _fi(halo["hi_src"]),
        _fd(halo["hi_w"]),
        halo["li_dst"].shape[1], _fi(halo["li_dst"]), _fi(halo["li_src"]),
        _fd(geom["stretch"]), _fd(geom["centroid"]), _fd(geom["metrics"]),
        _fd(geom["metrics_int"]), _fd(geom["vtxw"]), _fd(geom["xe"]),
        npair, _fi(pairs["src"]), _fi(pairs["dst"]), _fd(pairs["S"]),
        _fd(area_b), int(limiter_mode), f_in, f_out)
    return f_out


def apply_slices_masked(n, nb, col2pij, halo, geom, pairs, area_b,
                        limiter_mode, lf, mask, f_in, frac_tol=0.0,
                        fillval=np.nan):
    """Masked/fracarea remap: f_in(ncol, nslice) with per-slice validity
    mask(ncol, nslice) and per-cell area fraction lf(ncol) -> values that
    are means over the covered portion of each destination cell, plus the
    covered fraction frac(nb, nslice).  Cells with coverage <=
    frac_tol*area get fillval."""
    f_in = np.asarray(f_in, dtype=np.float64)
    if f_in.ndim == 1:
        f_in = f_in[:, None]
    f_in = np.asfortranarray(f_in)
    ncol, nslice = f_in.shape
    mask = np.asarray(mask, dtype=np.int32)
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = np.asfortranarray(np.broadcast_to(mask, (ncol, nslice)))
    f_out = np.zeros((nb, nslice), dtype=np.float64, order="F")
    frac_out = np.zeros((nb, nslice), dtype=np.float64, order="F")
    npair = pairs["dst"].size
    _lib.cr_apply_slices_masked(
        n, ncol, nslice, nb, _fi(col2pij),
        halo["hi_dst"].shape[1], _fi(halo["hi_dst"]), _fi(halo["hi_src"]),
        _fd(halo["hi_w"]),
        halo["li_dst"].shape[1], _fi(halo["li_dst"]), _fi(halo["li_src"]),
        _fd(geom["stretch"]), _fd(geom["centroid"]), _fd(geom["metrics"]),
        _fd(geom["metrics_int"]), _fd(geom["vtxw"]), _fd(geom["xe"]),
        npair, _fi(pairs["src"]), _fi(pairs["dst"]), _fd(pairs["S"]),
        _fd(area_b), int(limiter_mode), float(frac_tol), float(fillval),
        _fd(lf), mask, f_in, f_out, frac_out)
    return f_out, frac_out
