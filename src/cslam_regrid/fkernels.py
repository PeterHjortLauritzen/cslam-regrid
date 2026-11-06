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

    for fn in (lib.cr_lonlat_from_panel, lib.cr_panel_from_lonlat,
               lib.cr_cart_from_panel, lib.cr_panel_point_from_lonlat,
               lib.cr_seg_w6_horizontal, lib.cr_seg_w6_parallel_eq,
               lib.cr_seg_w6_parallel_po, lib.cr_seg_w6_meridian_po,
               lib.cr_cell_moments):
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
