"""Vector remapping tests: basis transforms, pole accuracy, auto-detection."""
import os

import numpy as np
import netCDF4
import pytest

from cslam_regrid import fkernels as fk
from cslam_regrid.apply import (RemapEngine, cart_to_uv, detect_vector_pairs,
                                uv_to_cart)
from cslam_regrid.grids import DestGrid, SourceGrid
from cslam_regrid.halo import build_halo
from cslam_regrid.walker import generate_weights

RNG = np.random.default_rng(23)


def test_basis_roundtrip():
    n = 5000
    lam = RNG.uniform(0, 2 * np.pi, n)
    phi = np.arcsin(RNG.uniform(-1, 1, n))
    u = RNG.uniform(-50, 50, (n, 3))
    v = RNG.uniform(-50, 50, (n, 3))
    vx, vy, vz = uv_to_cart(u, v, lam, phi)
    u2, v2 = cart_to_uv(vx, vy, vz, lam, phi)
    assert np.abs(u2 - u).max() < 1e-12
    assert np.abs(v2 - v).max() < 1e-12
    # the constructed vector is tangential: radial part must vanish
    rad = (np.cos(phi) * np.cos(lam))[:, None] * vx \
        + (np.cos(phi) * np.sin(lam))[:, None] * vy \
        + np.sin(phi)[:, None] * vz
    assert np.abs(rad).max() < 1e-12 * 50


def _centroid_lonlat(src):
    n = src.n
    cx = src.geom["centroid"][0].reshape(-1, order="F")
    cy = src.geom["centroid"][1].reshape(-1, order="F")
    lon = np.empty(6 * n * n)
    lat = np.empty(6 * n * n)
    for p in range(1, 7):
        lo, la = fk.lonlat_from_panel(cx, cy, np.full(n * n, p, np.int32))
        lon[(p - 1) * n * n:p * n * n] = lo
        lat[(p - 1) * n * n:p * n * n] = la
    return lon, lat


def _synth_col2pij(n):
    k = np.arange(6 * n * n)
    out = np.empty((3, k.size), dtype=np.int32)
    out[0] = k // (n * n) + 1
    out[1] = k % n + 1
    out[2] = (k // n) % n + 1
    return out


def test_solid_body_rotation_over_poles():
    """Rotation about an equatorial axis (flow straight over both poles):
    u = -sin(phi) cos(lam), v = sin(lam).  Componentwise remapping breaks
    down near the poles; the Cartesian vector path must not."""
    src = SourceGrid.from_ne(16)
    dst = DestGrid.create("offset", 45, 90)
    halo = build_halo(src.n)
    w = generate_weights(src, dst, check=False)
    pairs = dict(src=w["src"], dst=w["dst"], S=w["S"])
    col2pij = _synth_col2pij(src.n)

    lam_a, phi_a = _centroid_lonlat(src)
    u = -np.sin(phi_a) * np.cos(lam_a)
    v = np.sin(lam_a)

    lam_b = np.tile(dst.lon_centers, dst.nlat)
    phi_b = np.repeat(dst.lat_centers, dst.nlon)
    u_exact = -np.sin(phi_b) * np.cos(lam_b)
    v_exact = np.sin(lam_b)

    def remap(f, lim=0):
        return fk.apply_slices(src.n, dst.ncell, col2pij, halo, src.geom,
                               pairs, dst.areas(), lim, f)[:, 0]

    # componentwise (scalar) path
    u_s = remap(u)
    v_s = remap(v)
    # vector path
    vx, vy, vz = uv_to_cart(u[:, None], v[:, None], lam_a, phi_a)
    r = np.stack([remap(vx[:, 0]), remap(vy[:, 0]), remap(vz[:, 0])], axis=1)
    u_c, v_c = cart_to_uv(r[:, 0:1], r[:, 1:2], r[:, 2:3], lam_b, phi_b)
    u_c, v_c = u_c[:, 0], v_c[:, 0]

    cap = np.abs(phi_b) > np.deg2rad(78.0)
    err_s = max(np.abs(u_s - u_exact)[cap].max(),
                np.abs(v_s - v_exact)[cap].max())
    err_c = max(np.abs(u_c - u_exact)[cap].max(),
                np.abs(v_c - v_exact)[cap].max())
    # vector path must be far more accurate at the poles ...
    assert err_c < 0.25 * err_s
    assert err_c < 0.05
    # ... and no worse globally
    glob_s = np.abs(u_s - u_exact).max()
    glob_c = np.abs(u_c - u_exact).max()
    assert glob_c <= glob_s + 1e-12


def test_auto_detection_and_file_path(tmp_path):
    """End-to-end through remap_file: U/V auto-paired, U10 (a speed) not."""
    src = SourceGrid.from_ne(4)
    n = src.n
    lam_a, phi_a = _centroid_lonlat(src)

    infile = str(tmp_path / "native.nc")
    ds = netCDF4.Dataset(infile, "w", format="NETCDF3_64BIT_DATA")
    ds.createDimension("ncol", src.ncol)
    ds.createDimension("time", None)
    for name, data, attrs in (
        ("lat", np.rad2deg(phi_a), dict(units="degrees_north")),
        ("lon", np.rad2deg(lam_a), dict(units="degrees_east")),
    ):
        vv = ds.createVariable(name, "f8", ("ncol",))
        vv.setncatts(attrs)
        vv[:] = data
    tv = ds.createVariable("time", "f8", ("time",))
    tv[:] = [0.0]
    u = -np.sin(phi_a) * np.cos(lam_a)
    v = np.sin(lam_a)
    for name, data, ln in (
        ("U", u, "Zonal wind"),
        ("V", v, "Meridional wind"),
        ("U10", np.hypot(u, v), "10m wind speed"),
        ("TS", 280 + 10 * np.cos(phi_a), "Surface temperature"),
        ("UTEND_PHYSTOT", u, "U total physics tendency"),
        ("VTEND_PHYSTOT", v, "V total physics tendency"),
    ):
        vv = ds.createVariable(name, "f4", ("time", "ncol"))
        vv.long_name = ln
        vv.units = "m/s" if name != "TS" else "K"
        vv[0] = data
    assert detect_vector_pairs(
        ds, ["U", "V", "U10", "TS", "UTEND_PHYSTOT", "VTEND_PHYSTOT"]) == \
        [("U", "V"), ("UTEND_PHYSTOT", "VTEND_PHYSTOT")]
    ds.close()

    dst = DestGrid.create("offset", 18, 36)
    halo = build_halo(n)
    w = generate_weights(src, dst, check=False)
    from cslam_regrid.weights_io import write_weights
    mapfile = str(tmp_path / "map.nc")
    write_weights(mapfile, src, dst, halo,
                  dict(src=w["src"], dst=w["dst"], S=w["S"]), w["diags"],
                  ne=4)

    outfile = str(tmp_path / "latlon.nc")
    eng = RemapEngine(mapfile)
    eng.remap_file(infile, outfile, verbose=False)

    out = netCDF4.Dataset(outfile)
    assert out.variables["U"].cslam_vector_pair == "U:V"
    assert out.variables["V"].cslam_vector_pair == "U:V"
    assert out.variables["U"].cslam_limiter == "none"
    assert not hasattr(out.variables["U10"], "cslam_vector_pair")
    assert not hasattr(out.variables["TS"], "cslam_vector_pair")
    assert out.variables["TS"].cslam_limiter == "shape"
    # sanity: remapped winds are finite and of the right magnitude
    uu = np.asarray(out.variables["U"][0])
    assert np.isfinite(uu).all() and np.abs(uu).max() < 1.2
    out.close()
