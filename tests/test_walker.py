"""Walker tests: exactness invariants and numpy/Fortran cross-checks."""
import numpy as np
import pytest

from cslam_regrid import fkernels as fk
from cslam_regrid import walker
from cslam_regrid.grids import DestGrid, SourceGrid

RNG = np.random.default_rng(11)


# ---- numpy antiderivatives must match the Fortran reference bit-for-bit(ish)

def test_antiderivatives_match_fortran():
    for _ in range(200):
        xa, xb = np.sort(RNG.uniform(-0.98, 0.98, 2))
        c = RNG.uniform(-0.95, 0.95)
        w_np = walker.a_horizontal(c, [xb])[:, 0] - walker.a_horizontal(c, [xa])[:, 0]
        assert np.abs(w_np - fk.seg_w6_horizontal(c, xa, xb)).max() < 1e-15

        t = RNG.uniform(-0.9, 0.9)
        w_np = walker.a_parallel_eq(t, [xb])[:, 0] - walker.a_parallel_eq(t, [xa])[:, 0]
        assert np.abs(w_np - fk.seg_w6_parallel_eq(t, xa, xb)).max() < 1e-15

        r = RNG.uniform(0.2, 1.3)
        x1, x2 = np.sort(RNG.uniform(-0.95, 0.95, 2) * r)
        s = RNG.choice([-1.0, 1.0])
        w_np = walker.a_parallel_po(r, s, [x2])[:, 0] - walker.a_parallel_po(r, s, [x1])[:, 0]
        assert np.abs(w_np - fk.seg_w6_parallel_po(r, s, x1, x2)).max() < 1e-14

        m = RNG.uniform(0.05, 4.0) * RNG.choice([-1, 1])
        w_np = walker.a_meridian_po(m, [xb])[:, 0] - walker.a_meridian_po(m, [xa])[:, 0]
        assert np.abs(w_np - fk.seg_w6_meridian_po(m, xa, xb)).max() < 1e-14


# ---- exactness invariants over a matrix of grid pairings
# (includes: odd and even n; pole-cap cells; Gaussian latitudes; exact
# circle/grid-line tangencies - 1-deg-type edges vs integer-degree xe)

CASES = [
    (4, "offset", 18, 36),
    (4, "cap", 19, 36),
    (4, "gauss", 16, 32),
    (5, "offset", 18, 36),   # n=15 odd: pole inside a source cell
    (5, "cap", 19, 36),
    (5, "gauss", 16, 32),
    (4, "offset", 90, 180),  # 2-deg: lon edges coincide with source x-lines
]


@pytest.mark.parametrize("ne,kind,nlat,nlon", CASES)
def test_invariants(ne, kind, nlat, nlon):
    src = SourceGrid.from_ne(ne)
    dst = DestGrid.create(kind, nlat, nlon)
    w = walker.generate_weights(src, dst)
    assert w["diags"]["dst_area_abserr"] < 1e-13          # vs sphere = 4*pi
    assert w["diags"]["src_moment_relerr"] < 1e-12
    # every destination cell covered, every source cell used
    assert np.unique(w["dst"]).size == dst.ncell
    lin = ((w["src"][0].astype(np.int64) - 1) * src.n
           + (w["src"][2] - 1)) * src.n + (w["src"][1] - 1)
    assert np.unique(lin).size == src.ncol
