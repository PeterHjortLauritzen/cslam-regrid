"""End-to-end remap tests: constant preservation, conservation, shape
preservation, positive definiteness, and convergence on a smooth field.

Source cell averages are computed with 10x10 Gauss-Legendre quadrature per
gnomonic cell (machine-exact for smooth fields at these resolutions), so the
tests exercise true finite-volume data, not point samples.
"""
import numpy as np
import pytest

from cslam_regrid import fkernels as fk
from cslam_regrid.grids import DestGrid, SourceGrid
from cslam_regrid.halo import build_halo
from cslam_regrid.walker import generate_weights

RNG = np.random.default_rng(3)


def synth_col2pij(n):
    """Synthetic ncol ordering: column k <-> (p,i,j), i fastest."""
    ncol = 6 * n * n
    k = np.arange(ncol)
    out = np.empty((3, ncol), dtype=np.int32)
    out[0] = k // (n * n) + 1
    out[1] = k % n + 1
    out[2] = (k // n) % n + 1
    return out


def smooth_field(lon, lat):
    return 2.0 + np.cos(lat) ** 2 * np.cos(2 * lon) + 0.5 * np.sin(lat)


def src_cell_averages(src, func, nq=10):
    """Exact spherical cell averages of func(lon,lat) on every panel cell,
    returned in synth_col2pij ordering."""
    n = src.n
    xe = src.geom["xe"]
    gx, gw = np.polynomial.legendre.leggauss(nq)
    x1 = xe[:-1][:, None]
    x2 = xe[1:][:, None]
    xq = 0.5 * (x1 + x2) + 0.5 * (x2 - x1) * gx       # (n, nq)
    wq = 0.5 * (x2 - x1) * gw                          # (n, nq)
    out = np.empty((6, n, n))
    for p in range(1, 7):
        for j in range(n):
            X = np.repeat(xq, nq, axis=1).reshape(n, nq, nq)       # x nodes
            Y = np.tile(xq[j], (n, nq, 1))                         # y nodes
            W = (np.repeat(wq, nq, axis=1).reshape(n, nq, nq)
                 * np.tile(wq[j], (n, nq, 1)))
            omega = (1 + X ** 2 + Y ** 2) ** -1.5
            lon, lat = fk.lonlat_from_panel(X.ravel(), Y.ravel(),
                                            np.full(X.size, p, np.int32))
            F = func(lon, lat).reshape(n, nq, nq)
            num = (F * omega * W).sum(axis=(1, 2))
            den = (omega * W).sum(axis=(1, 2))
            out[p - 1, :, j] = num / den
    # synth order: k = ((p-1)*n + (j-1))*n + (i-1): i fastest
    return np.concatenate([out[p].reshape(-1, order="F") for p in range(6)])


def dst_cell_averages(dst, func, nq=20):
    gx, gw = np.polynomial.legendre.leggauss(nq)
    lam1, lam2 = dst.lon_edges[:-1], dst.lon_edges[1:]
    mu1 = np.sin(dst.lat_edges[:-1])
    mu2 = np.sin(dst.lat_edges[1:])
    lam = 0.5 * (lam1 + lam2)[:, None] + 0.5 * (lam2 - lam1)[:, None] * gx
    mu = 0.5 * (mu1 + mu2)[:, None] + 0.5 * (mu2 - mu1)[:, None] * gx
    out = np.empty((dst.nlat, dst.nlon))
    for J in range(dst.nlat):
        LAT = np.arcsin(np.clip(mu[J], -1, 1))
        F = func(lam[:, None, :], LAT[None, :, None])  # (nlon, nq_mu, nq_lam)
        out[J] = (F * gw[None, :, None] * gw[None, None, :]).sum(axis=(1, 2)) / 4.0
    return out.ravel()  # lon fastest


def setup(ne, kind="offset", nlat=18, nlon=36):
    src = SourceGrid.from_ne(ne)
    dst = DestGrid.create(kind, nlat, nlon)
    halo = build_halo(src.n)
    w = generate_weights(src, dst, check=False)
    pairs = dict(src=w["src"], dst=w["dst"], S=w["S"])
    col2pij = synth_col2pij(src.n)
    return src, dst, halo, pairs, col2pij


def remap(src, dst, halo, pairs, col2pij, f, limiter):
    return fk.apply_slices(src.n, dst.ncell, col2pij, halo, src.geom,
                           pairs, dst.areas(), limiter, f)[:, 0]


@pytest.fixture(scope="module")
def ne4():
    return setup(4)


@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_constant_field(ne4, limiter):
    src, dst, halo, pairs, col2pij = ne4
    out = remap(src, dst, halo, pairs, col2pij, np.ones(src.ncol), limiter)
    assert np.abs(out - 1.0).max() < 1e-12


@pytest.mark.parametrize("limiter", [0, 1])
def test_conservation_random_field(ne4, limiter):
    src, dst, halo, pairs, col2pij = ne4
    f = RNG.uniform(0.0, 10.0, src.ncol)
    out = remap(src, dst, halo, pairs, col2pij, f, limiter)
    area_a = np.tile(src.geom["area"].reshape(-1, order="F"), 6)
    total_src = (f * area_a).sum()
    total_dst = (out * dst.areas()).sum()
    assert abs(total_dst - total_src) / abs(total_src) < 1e-13


def test_shape_preservation_random_field(ne4):
    src, dst, halo, pairs, col2pij = ne4
    f = RNG.uniform(-3.0, 7.0, src.ncol)
    out = remap(src, dst, halo, pairs, col2pij, f, limiter=1)
    assert out.min() >= f.min() - 1e-12
    assert out.max() <= f.max() + 1e-12


def test_unlimited_overshoots_limited_does_not(ne4):
    """A step field must overshoot without the limiter and not with it --
    guards against the limiter being silently inert."""
    src, dst, halo, pairs, col2pij = ne4
    col2 = col2pij
    lon, lat = fk.lonlat_from_panel(
        np.tile(src.geom["xc"], 6 * src.n),
        np.repeat(np.tile(src.geom["xc"], 6), src.n), np.repeat(
            np.arange(1, 7, dtype=np.int32), src.n * src.n))
    f = (lat > np.deg2rad(20.0)).astype(float)
    out_un = remap(src, dst, halo, pairs, col2, f, limiter=0)
    out_sh = remap(src, dst, halo, pairs, col2, f, limiter=1)
    assert out_un.min() < -1e-4 or out_un.max() > 1 + 1e-4
    assert out_sh.min() >= -1e-12 and out_sh.max() <= 1 + 1e-12


def test_posdef_random_field(ne4):
    src, dst, halo, pairs, col2pij = ne4
    f = np.maximum(RNG.uniform(-2.0, 5.0, src.ncol), 0.0)
    out0 = remap(src, dst, halo, pairs, col2pij, f, limiter=0)
    out2 = remap(src, dst, halo, pairs, col2pij, f, limiter=2)
    assert out0.min() < -1e-8          # unlimited goes negative
    assert out2.min() >= -1e-13        # posdef does not
    area_a = np.tile(src.geom["area"].reshape(-1, order="F"), 6)
    assert abs((out2 * dst.areas()).sum() - (f * area_a).sum()) \
        / (f * area_a).sum() < 1e-13


def test_smooth_field_accuracy_and_convergence():
    dst = DestGrid.create("offset", 24, 48)
    fb = dst_cell_averages(dst, smooth_field)
    errs = {}
    for ne in (4, 8):
        src = SourceGrid.from_ne(ne)
        halo = build_halo(src.n)
        w = generate_weights(src, dst, check=False)
        pairs = dict(src=w["src"], dst=w["dst"], S=w["S"])
        col2pij = synth_col2pij(src.n)
        f = src_cell_averages(src, smooth_field)
        for lim in (0, 1):
            out = remap(src, dst, halo, pairs, col2pij, f, lim)
            errs[(ne, lim)] = np.abs(out - fb).max()
    # third-order scheme: doubling ne should cut linf by ~8; demand >= 5
    assert errs[(8, 0)] < errs[(4, 0)] / 5.0
    assert errs[(8, 1)] < errs[(4, 1)] / 4.0
    # limiter should not destroy smooth-field accuracy
    assert errs[(8, 1)] < 4.0 * errs[(8, 0)]
    assert errs[(8, 0)] < 2e-4
