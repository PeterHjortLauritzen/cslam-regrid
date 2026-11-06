"""Panel transform tests: roundtrips, Cartesian consistency, seam geometry."""
import numpy as np
import pytest

from cslam_regrid import fkernels as fk

RNG = np.random.default_rng(7)


def random_lonlat(n):
    lon = RNG.uniform(0, 2 * np.pi, n)
    lat = np.arcsin(RNG.uniform(-1, 1, n))
    return lon, lat


def test_roundtrip_lonlat_panel_lonlat():
    lon, lat = random_lonlat(20000)
    x, y, p = fk.panel_from_lonlat(lon, lat)
    assert p.min() >= 1 and p.max() <= 6
    assert np.all(np.abs(x) <= 1 + 1e-14) and np.all(np.abs(y) <= 1 + 1e-14)
    lon2, lat2 = fk.lonlat_from_panel(x, y, p)
    dlon = np.abs(np.angle(np.exp(1j * (lon2 - lon))))
    assert dlon.max() < 1e-13
    assert np.abs(lat2 - lat).max() < 1e-13


def test_cart_consistency():
    lon, lat = random_lonlat(5000)
    x, y, p = fk.panel_from_lonlat(lon, lat)
    cart = fk.cart_from_panel(x, y, p)
    ref = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    assert np.abs(cart - ref).max() < 1e-14


def test_all_panels_hit():
    lon, lat = random_lonlat(5000)
    _, _, p = fk.panel_from_lonlat(lon, lat)
    assert set(np.unique(p)) == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize("panel", [1, 2, 3, 4, 5, 6])
def test_panel_point_matches_native(panel):
    """panel_point_from_lonlat agrees with panel_from_lonlat on-panel."""
    xg = RNG.uniform(-0.95, 0.95, 400)
    yg = RNG.uniform(-0.95, 0.95, 400)
    lon, lat = fk.lonlat_from_panel(xg, yg, panel)
    x2, y2, valid = fk.panel_point_from_lonlat(lon, lat, panel)
    assert valid.all()
    assert np.abs(x2 - xg).max() < 1e-13
    assert np.abs(y2 - yg).max() < 1e-13


def test_equatorial_line_continues_as_polar_line():
    """Panel 1's gnomonic line y=c (plane Z=c*X) is panel 5's line y=-1/c:
    the seam-crossing geometry behind halo filling and overlap walks."""
    c = 1.7  # beyond panel 1's top edge -> on panel 5
    X = RNG.uniform(0.05, 1.0, 200)
    Y = RNG.uniform(-0.5, 0.5, 200) * X  # keep Z = c*X dominant
    Z = c * X
    cart = np.array([X, Y, Z]) / np.sqrt(X**2 + Y**2 + Z**2)
    lon = np.arctan2(cart[1], cart[0])
    lat = np.arcsin(cart[2])
    x5, y5, p = fk.panel_from_lonlat(np.mod(lon, 2 * np.pi), lat)
    assert np.all(p == 5)
    assert np.abs(y5 - (-1.0 / c)).max() < 1e-14


def test_parallel_is_circle_on_polar_panel():
    """A latitude circle maps to x^2+y^2 = cot^2(lat) on panel 5."""
    lat = np.full(100, np.deg2rad(72.0))
    lon = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x, y, p = fk.panel_from_lonlat(lon, lat)
    assert np.all(p == 5)
    r = 1.0 / np.tan(lat[0])
    assert np.abs(np.hypot(x, y) - r).max() < 1e-14


def test_parallel_is_hyperbola_on_equatorial_panel():
    """A latitude circle maps to y = tan(lat)*sqrt(1+x^2) on panel 1."""
    lat = np.full(50, np.deg2rad(20.0))
    lon = np.linspace(-np.pi / 4 + 0.01, np.pi / 4 - 0.01, 50)
    x, y, p = fk.panel_from_lonlat(np.mod(lon, 2 * np.pi), lat)
    assert np.all(p == 1)
    assert np.abs(y - np.tan(lat[0]) * np.sqrt(1 + x**2)).max() < 1e-14


def test_meridian_is_radial_on_polar_panel():
    """A meridian maps to a line through the origin on panel 5."""
    lon = np.full(50, np.deg2rad(33.0))
    lat = np.linspace(np.deg2rad(55), np.deg2rad(89), 50)
    x, y, p = fk.panel_from_lonlat(lon, lat)
    assert np.all(p == 5)
    # lon = atan2(x, -y) => x*cos(lon) + y*sin(lon)... check collinearity:
    cross = x * (-np.cos(lon[0])) - y * np.sin(lon[0])
    # direction vector of the radial line: (sin(lon), -cos(lon))
    assert np.abs(cross).max() < 1e-14
