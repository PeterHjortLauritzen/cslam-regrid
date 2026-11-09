"""Masked (land/CLM) fracarea remapping tests.

Kernel level: constants reproduce exactly on the covered region (the
fracarea property), the landfrac-weighted integral is conserved, shape
preservation holds with bounds from valid neighbors only, invalid-cell
values cannot leak, and the all-valid/lf=1 case reduces to the unmasked
operator.  File level: a synthetic CLM-like file (fill-value coordinates
over ocean, fractional landfrac, per-field and per-level valid sets)
round-trips through remap_file with correct masks, metadata and
conservation.
"""
import numpy as np
import netCDF4
import pytest

from cslam_regrid import fkernels as fk
from cslam_regrid.apply import RemapEngine
from cslam_regrid.grids import DestGrid, SourceGrid
from cslam_regrid.halo import build_halo
from cslam_regrid.walker import generate_weights
from cslam_regrid.weights_io import write_weights

RNG = np.random.default_rng(11)


def synth_col2pij(n):
    ncol = 6 * n * n
    k = np.arange(ncol)
    out = np.empty((3, ncol), dtype=np.int32)
    out[0] = k // (n * n) + 1
    out[1] = k % n + 1
    out[2] = (k // n) % n + 1
    return out


def centroid_lonlat(src):
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


def continents(src):
    """Deterministic blobby mask (~half the sphere) + fractional lf on a
    random 30% of the valid cells."""
    lon, lat = centroid_lonlat(src)
    valid = (np.cos(lat) ** 2 * np.cos(2 * lon) + 0.5 * np.sin(lat)) > 0.15
    lf = np.where(valid, 1.0, 0.0)
    frac_cells = valid & (RNG.uniform(size=valid.size) < 0.3)
    lf[frac_cells] = RNG.uniform(0.05, 0.95, frac_cells.sum())
    return valid, lf


@pytest.fixture(scope="module")
def ne4():
    src = SourceGrid.from_ne(4)
    dst = DestGrid.create("offset", 18, 36)
    halo = build_halo(src.n)
    w = generate_weights(src, dst, check=False)
    pairs = dict(src=w["src"], dst=w["dst"], S=w["S"])
    return src, dst, halo, pairs, synth_col2pij(src.n)


def remap_masked(src, dst, halo, pairs, col2pij, f, mask, lf, limiter,
                 frac_tol=0.0):
    out, frac = fk.apply_slices_masked(
        src.n, dst.ncell, col2pij, halo, src.geom, pairs, dst.areas(),
        limiter, lf, mask.astype(np.int32), f, frac_tol=frac_tol,
        fillval=np.nan)
    return out[:, 0], frac[:, 0]


@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_constant_exact_on_covered_region(ne4, limiter):
    """fracarea normalization reproduces constants exactly on the covered
    region, regardless of mask shape or fractional lf."""
    src, dst, halo, pairs, col2pij = ne4
    valid, lf = continents(src)
    f = np.where(valid, 1.0, 1e30)   # garbage on invalid cells
    out, frac = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf,
                             limiter, frac_tol=1e-8)
    covered = ~np.isnan(out)
    assert covered.any() and not covered.all()
    assert np.abs(out[covered] - 1.0).max() < 1e-13
    assert np.isnan(out[frac <= 1e-8]).all()


@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_conservation_of_land_integral(ne4, limiter):
    src, dst, halo, pairs, col2pij = ne4
    valid, lf = continents(src)
    f = np.where(valid, RNG.uniform(0.0, 10.0, src.ncol), 0.0)
    out, frac = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf,
                             limiter, frac_tol=0.0)
    area_a = np.tile(src.geom["area"].reshape(-1, order="F"), 6)
    total_src = (f * lf * area_a)[valid].sum()
    covered = ~np.isnan(out)
    total_dst = (out[covered] * frac[covered] * dst.areas()[covered]).sum()
    assert abs(total_dst - total_src) / abs(total_src) < 1e-13


def test_shape_preservation_masked(ne4):
    """Bounds must come from valid neighbors only: with garbage on the
    invalid cells, covered output stays within the valid-cell range."""
    src, dst, halo, pairs, col2pij = ne4
    valid, lf = continents(src)
    f = np.where(valid, RNG.uniform(-3.0, 7.0, src.ncol), 1e30)
    out, _ = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf, 1)
    covered = ~np.isnan(out)
    assert out[covered].min() >= f[valid].min() - 1e-12
    assert out[covered].max() <= f[valid].max() + 1e-12


def test_posdef_masked(ne4):
    src, dst, halo, pairs, col2pij = ne4
    valid, lf = continents(src)
    f = np.where(valid, np.maximum(RNG.uniform(-2.0, 5.0, src.ncol), 0.0),
                 1e30)
    out0, _ = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf, 0)
    out2, _ = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf, 2)
    assert np.nanmin(out2) >= -1e-13


def test_invalid_values_cannot_leak(ne4):
    """Changing the values of masked-out cells must not change anything."""
    src, dst, halo, pairs, col2pij = ne4
    valid, lf = continents(src)
    f0 = np.where(valid, RNG.uniform(0.0, 5.0, src.ncol), 0.0)
    f1 = f0.copy()
    f1[~valid] = 1e30
    for lim in (0, 1):
        a, fa = remap_masked(src, dst, halo, pairs, col2pij, f0, valid, lf,
                             lim)
        b, fb = remap_masked(src, dst, halo, pairs, col2pij, f1, valid, lf,
                             lim)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(fa, fb)


def test_all_valid_reduces_to_unmasked(ne4):
    """mask all-1, lf=1: the masked path is the unmasked operator up to the
    fracarea row normalization (exact-area roundoff)."""
    src, dst, halo, pairs, col2pij = ne4
    lon, lat = centroid_lonlat(src)
    f = 2.0 + np.cos(lat) ** 2 * np.cos(2 * lon) + 0.5 * np.sin(lat)
    valid = np.ones(src.ncol, dtype=bool)
    lf = np.ones(src.ncol)
    for lim in (0, 1, 2):
        ref = fk.apply_slices(src.n, dst.ncell, col2pij, halo, src.geom,
                              pairs, dst.areas(), lim, f)[:, 0]
        out, frac = remap_masked(src, dst, halo, pairs, col2pij, f, valid,
                                 lf, lim)
        assert np.abs(out - ref).max() < 1e-10
        assert np.abs(frac - 1.0).max() < 1e-10


def test_coastal_fallback_is_first_order_not_garbage(ne4):
    """A smooth field on a masked region: coastal cells fall back to first
    order but remain accurate to O(dx); interior keeps high order."""
    src, dst, halo, pairs, col2pij = ne4
    lon, lat = centroid_lonlat(src)
    valid, lf = continents(src)
    f = np.where(valid, 2.0 + np.sin(lat), 1e30)
    out, frac = remap_masked(src, dst, halo, pairs, col2pij, f, valid, lf, 1)
    covered = ~np.isnan(out)
    lat_b = np.repeat(dst.lat_centers, dst.nlon)
    err = np.abs(out - (2.0 + np.sin(lat_b)))[covered]
    assert err.max() < 0.12      # bounded by O(dx) everywhere (ne4!)


# ---------------------------------------------------------------------------
# file level
# ---------------------------------------------------------------------------

def test_land_file_end_to_end(tmp_path):
    src = SourceGrid.from_ne(4)
    n = src.n
    ncol = src.ncol
    lon, lat = centroid_lonlat(src)
    valid, lf = continents(src)
    fillf = np.float32(1e36)
    filld = np.float64(1e36)

    infile = str(tmp_path / "native_clm.nc")
    ds = netCDF4.Dataset(infile, "w", format="NETCDF3_64BIT_DATA")
    ds.createDimension("lndgrid", ncol)
    ds.createDimension("levsoi", 2)
    ds.createDimension("gridcell", int(valid.sum()))
    ds.createDimension("time", None)

    def coordvar(name, data, **attrs):
        v = ds.createVariable(name, "f4", ("lndgrid",), fill_value=fillf)
        v.setncatts(attrs)
        a = data.astype(np.float32)
        a[~valid] = fillf
        v[:] = np.ma.masked_values(a, fillf)

    coordvar("lat", np.rad2deg(lat), units="degrees_north")
    coordvar("lon", np.rad2deg(lon), units="degrees_east")
    coordvar("landfrac", lf, long_name="land fraction")
    coordvar("area", np.ones(ncol), units="km^2")
    v = ds.createVariable("landmask", "i4", ("lndgrid",))
    v[:] = valid.astype(np.int32)
    v = ds.createVariable("nbedrock", "i4", ("lndgrid",))
    v[:] = np.ones(ncol, np.int32)          # int -> must be skipped
    v = ds.createVariable("grid1d_ixy", "f8", ("gridcell",))
    v[:] = np.arange(valid.sum())           # sub-grid dim -> skipped
    tv = ds.createVariable("time", "f8", ("time",))
    tv[:] = [0.0]

    tsa = np.where(valid, 280.0 + 10.0 * np.sin(lat), filld)
    v = ds.createVariable("TSA", "f8", ("time", "lndgrid"),
                          fill_value=filld)
    v.long_name = "2m air temperature"
    v.units = "K"
    v[0] = np.ma.masked_values(tsa, filld)

    # TSOI: level 1 has a smaller valid set than level 0 (per-level masks)
    sub = valid.copy()
    sub[np.flatnonzero(valid)[::3]] = False
    tsoi = np.full((2, ncol), fillf, np.float32)
    tsoi[0, valid] = 270.0
    tsoi[1, sub] = 260.0
    v = ds.createVariable("TSOI", "f4", ("time", "levsoi", "lndgrid"),
                          fill_value=fillf)
    v.long_name = "soil temperature"
    v[0] = np.ma.masked_values(tsoi, fillf)

    # TLAKE: sparse valid set (a few lakes)
    lakes = np.zeros(ncol, bool)
    lakes[np.flatnonzero(valid)[::40]] = True
    tl = np.full(ncol, fillf, np.float32)
    tl[lakes] = 277.0
    v = ds.createVariable("TLAKE", "f4", ("time", "lndgrid"),
                          fill_value=fillf)
    v.long_name = "lake temperature"
    v[0] = np.ma.masked_values(tl, fillf)
    ds.close()

    dst = DestGrid.create("offset", 18, 36)
    halo = build_halo(n)
    w = generate_weights(src, dst, check=False)
    mapfile = str(tmp_path / "map.nc")
    write_weights(mapfile, src, dst, halo,
                  dict(src=w["src"], dst=w["dst"], S=w["S"]), w["diags"],
                  ne=4)

    outfile = str(tmp_path / "latlon.nc")
    eng = RemapEngine(mapfile)
    eng.remap_file(infile, outfile, verbose=False)

    out = netCDF4.Dataset(outfile)
    assert "fracarea" in out.cslam_normalization
    # destination landfrac: exact land-area bookkeeping against the
    # landfrac the engine actually read (float32 in the file)
    lf = np.where(valid, lf.astype(np.float32).astype(np.float64), 0.0)
    frac2d = np.asarray(out.variables["landfrac"][:])
    area_a = np.tile(src.geom["area"].reshape(-1, order="F"), 6)
    assert abs((frac2d.ravel() * dst.areas()).sum() - (lf * area_a).sum()) \
        / (lf * area_a).sum() < 1e-12
    assert frac2d.min() >= 0.0 and frac2d.max() <= 1.0 + 1e-12
    lmask = np.asarray(out.variables["landmask"][:])
    assert set(np.unique(lmask)) <= {0, 1}

    # TSA (f8 in, f8 out): fill over ocean, shape bounds, conservation
    vtsa = out.variables["TSA"]
    assert vtsa.cslam_limiter == "shape"
    a = vtsa[0]
    m = np.ma.getmaskarray(a)
    assert m.any()
    assert (~m == (lmask == 1)).all()       # covered exactly where land
    vals = np.asarray(a[~m])
    assert vals.min() >= tsa[valid].min() - 1e-12
    assert vals.max() <= tsa[valid].max() + 1e-12
    got = (np.ma.filled(a, 0.0).ravel() * frac2d.ravel() * dst.areas()).sum()
    want = (tsa[valid] * lf[valid] * area_a[valid]).sum()
    assert abs(got - want) / abs(want) < 1e-12

    # TSOI: per-level masks; level 1 (smaller valid set) covers fewer cells
    ts = out.variables["TSOI"][0]
    m0 = ~np.ma.getmaskarray(ts[0])
    m1 = ~np.ma.getmaskarray(ts[1])
    assert m1.sum() < m0.sum()
    assert np.abs(np.asarray(ts[0][m0]) - 270.0).max() < 1e-5   # constant, f4
    assert np.abs(np.asarray(ts[1][m1]) - 260.0).max() < 1e-5

    # TLAKE: sparse; all covered values equal the lake constant
    tlk = out.variables["TLAKE"][0]
    mlk = ~np.ma.getmaskarray(tlk)
    assert 0 < mlk.sum() < m0.sum()
    assert np.abs(np.asarray(tlk[mlk]) - 277.0).max() < 1e-5

    # skipped: int, sub-grid, and replaced native vars
    for name in ("nbedrock", "grid1d_ixy", "area"):
        assert name not in out.variables
    assert "gridcell" not in out.dimensions
    assert "lndgrid" not in out.dimensions
    out.close()
