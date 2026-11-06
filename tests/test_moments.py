"""Exact line-integral tests: Green's identities and mpmath cross-checks.

Region moments are assembled as integrals of the dx-potentials along the
region's lower/upper boundary curves:  w = W(upper) - W(lower)  for a
y-simple region x in [xa,xb], y_bot(x) <= y <= y_top(x).
"""
import mpmath as mp
import numpy as np
import pytest

from cslam_regrid import fkernels as fk

mp.mp.dps = 30

CHI = [
    lambda x, y: 1,
    lambda x, y: x,
    lambda x, y: y,
    lambda x, y: x * x,
    lambda x, y: y * y,
    lambda x, y: x * y,
]


def mp_region_moments(xa, xb, ybot, ytop, breaks=()):
    """High-precision reference: integral of chi_k * (1+x^2+y^2)^(-3/2).
    Supply interior kink locations of ybot/ytop via `breaks` (mp.quad
    assumes a smooth integrand between the listed points)."""
    pts = [xa, *breaks, xb]
    out = []
    for chi in CHI:
        f = lambda x: mp.quad(
            lambda y: chi(x, y) * (1 + x * x + y * y) ** mp.mpf("-1.5"),
            [ybot(x), ytop(x)],
        )
        out.append(float(mp.quad(f, pts)))
    return np.array(out)


def test_sphere_area():
    mom = fk.cell_moments(-1.0, 1.0, -1.0, 1.0)
    assert abs(6 * mom[0] - 4 * np.pi) < 1e-13
    # full-panel symmetry: centroids and cross moment vanish
    assert abs(mom[1]) < 1e-15 and abs(mom[2]) < 1e-15 and abs(mom[5]) < 1e-15


def test_cell_moment_additivity():
    a = fk.cell_moments(-0.3, 0.5, 0.1, 0.9)
    b = fk.cell_moments(-0.3, 0.1, 0.1, 0.9)
    c = fk.cell_moments(0.1, 0.5, 0.1, 0.9)
    assert abs(b[0] + c[0] - a[0]) < 1e-15
    for k in range(1, 6):
        assert abs(b[0] * b[k] + c[0] * c[k] - a[0] * a[k]) < 1e-15


def test_rectangle_via_horizontal_segments():
    x1, x2, y1, y2 = -0.42, 0.77, -0.15, 0.6
    w = fk.seg_w6_horizontal(y2, x1, x2) - fk.seg_w6_horizontal(y1, x1, x2)
    mom = fk.cell_moments(x1, x2, y1, y2)
    raw = mom.copy()
    raw[1:] *= mom[0]
    assert np.abs(w - raw).max() < 1e-15


def test_rectangle_vs_mpmath():
    x1, x2, y1, y2 = 0.05, 0.9, -0.8, -0.2
    w = fk.seg_w6_horizontal(y2, x1, x2) - fk.seg_w6_horizontal(y1, x1, x2)
    ref = mp_region_moments(x1, x2, lambda x: y1, lambda x: y2)
    assert np.abs(w - ref).max() < 1e-14


def test_parallel_eq_region_vs_mpmath():
    """Region between y = c and the parallel y = t*sqrt(1+x^2) above it."""
    t, c, xa, xb = 0.35, 0.2, -0.6, 0.85
    assert t * np.sqrt(1 + min(xa, xb) ** 2) > c
    w = fk.seg_w6_parallel_eq(t, xa, xb) - fk.seg_w6_horizontal(c, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: c, lambda x: t * mp.sqrt(1 + x * x))
    assert np.abs(w - ref).max() < 1e-14


def test_parallel_eq_region_negative_t():
    """Southern-hemisphere band: parallel below a horizontal line."""
    t, c, xa, xb = -0.55, -0.1, -0.4, 0.3
    w = fk.seg_w6_horizontal(c, xa, xb) - fk.seg_w6_parallel_eq(t, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: t * mp.sqrt(1 + x * x), lambda x: c)
    assert np.abs(w - ref).max() < 1e-14


def test_parallel_po_region_vs_mpmath():
    """Region between y = c and the upper circle arc y = +sqrt(r^2-x^2)."""
    r, c, xa, xb = 0.8, 0.25, -0.5, 0.4
    assert np.sqrt(r**2 - max(abs(xa), abs(xb)) ** 2) > c
    w = fk.seg_w6_parallel_po(r, +1.0, xa, xb) - fk.seg_w6_horizontal(c, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: c, lambda x: mp.sqrt(r * r - x * x))
    assert np.abs(w - ref).max() < 1e-14


def test_parallel_po_lower_branch_vs_mpmath():
    """Region between the lower arc y = -sqrt(r^2-x^2) and y = c above it."""
    r, c, xa, xb = 0.9, -0.1, -0.55, 0.3
    w = fk.seg_w6_horizontal(c, xa, xb) - fk.seg_w6_parallel_po(r, -1.0, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: -mp.sqrt(r * r - x * x), lambda x: c)
    assert np.abs(w - ref).max() < 1e-14


def test_parallel_po_full_cap():
    """Polar cap: region inside the full circle r (= lat ring on a polar
    panel, entirely within the panel for r <= 1).  Contour = upper arc
    (left->right along +x is the *lower* boundary sense check) and lower arc.
    Cap area on the sphere = 2*pi*(1 - sin(lat)) / ... computed exactly."""
    r = 0.65
    # y-simple: ybot = -arc, ytop = +arc over x in [-r, r]
    w = fk.seg_w6_parallel_po(r, +1.0, -r, r) - fk.seg_w6_parallel_po(r, -1.0, -r, r)
    lat = np.arctan(1.0 / r)  # colatitude angle of the ring: tan(colat) = r
    cap_area = 2 * np.pi * (1 - np.sin(lat))
    assert abs(w[0] - cap_area) < 1e-14
    # symmetry: first moments and cross moment vanish
    assert abs(w[1]) < 1e-15 and abs(w[2]) < 1e-15 and abs(w[5]) < 1e-15


def test_meridian_po_region_vs_mpmath():
    """Region between the radial line y = m*x and y = c above it."""
    m, c, xa, xb = 0.7, 0.8, 0.1, 0.75
    assert m * xb < c
    w = fk.seg_w6_horizontal(c, xa, xb) - fk.seg_w6_meridian_po(m, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: m * x, lambda x: c)
    assert np.abs(w - ref).max() < 1e-14


def test_meridian_po_negative_slope_vs_mpmath():
    m, c, xa, xb = -1.4, -0.9, -0.6, -0.1
    w = fk.seg_w6_meridian_po(m, xa, xb) - fk.seg_w6_horizontal(c, xa, xb)
    ref = mp_region_moments(xa, xb, lambda x: c, lambda x: m * x)
    assert np.abs(w - ref).max() < 1e-14


def test_wedge_rect_circle_composite():
    """Composite region: inside circle r, above radial line m*x, below y=c,
    x in [xa,xb] -- mimics a real polar-panel overlap with 3 curve types."""
    r, m, c, xa, xb = 1.1, 0.3, 0.9, 0.25, 0.7

    def ytop(x):
        return min(mp.sqrt(r * r - x * x), c)

    def ybot(x):
        return m * x

    # split point where circle meets y=c: x* = sqrt(r^2-c^2)
    xs = float(np.sqrt(r**2 - c**2))
    assert xa < xs < xb
    w = (
        fk.seg_w6_horizontal(c, xa, xs)
        + fk.seg_w6_parallel_po(r, +1.0, xs, xb)
        - fk.seg_w6_meridian_po(m, xa, xb)
    )
    ref = mp_region_moments(xa, xb, ybot, ytop, breaks=(xs,))
    assert np.abs(w - ref).max() < 1e-14
