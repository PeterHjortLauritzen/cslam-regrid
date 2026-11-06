# cslam-regrid

Conservative, shape-preserving, geometrically exact remapping of CESM/CAM
native cubed-sphere physics-grid history output (regular ne·pg3) to
latitude-longitude grids, built from CSLAM technology:

- **High order**: the CAM-SE/CSLAM quadratic reconstruction (`irecons=6`),
  ported from `fvm_reconstruction_mod.F90` — 3rd-order on smooth fields.
- **Conservative**: exact overlap integrals of the reconstruction; the
  limiter rescales only the deviation from the cell mean, so global
  conservation is machine-exact *with or without* limiting.
- **Shape-preserving**: the CSLAM limiter guarantees destination values
  within the local min/max of the source neighborhood — no spurious
  negatives, no overshoots, no post-hoc clipping.
- **Geometrically exact** (GECoRe method; Ullrich, Lauritzen & Jablonowski
  2009 MWR): every overlap moment is evaluated from closed-form line
  integrals — no quadrature, no polygon approximations. Weight-generation
  invariants (destination areas, source moments) are verified at machine
  precision every time a map is written.

Everything is precomputed into a self-describing netCDF weight file; only
the (field-dependent) limiter runs at apply time, on precomputed geometry.

## Why

The CMIP7 regridding study (`cmip7-interp`) showed that no ESMF option is
simultaneously conservative, shape-preserving and high-order for ne30pg3:
`conserve` imprints polar blockiness and panel seams, `conserve2nd`
produces negative precipitation (to −1.3 mm/day along coasts) and
several-K temperature overshoots. CSLAM reconstruct-and-limit is exactly
that missing combination.

### Validation (ne30pg3 → 1°, exact cell-average inputs; `validation/`)

Precipitation surrogate, 60–90N cap (the polar-blockiness/negatives test):

| method | l2 (cap) | negatives | most negative | conservation |
|---|---|---|---|---|
| ESMF bilinear | 4.0e-3 | 0 | — | 6e-4 |
| ESMF conserve | 1.5e-2 (blocky) | 0 | — | 7e-6¹ |
| ESMF conserve2nd | 2.7e-3 | **902 cells** | −2.1e-4 | 7e-6¹ |
| cslam-none | 4.3e-5 | 415 cells | −2.8e-5 | 2e-13 |
| **cslam-shape** | 2.2e-4 | **0** | 0 | 1e-13 |
| **cslam-posdef** | **4.3e-5** | **0** | 0 | 2e-13 |

¹ ESMF conserve/conserve2nd are machine-exact on their own area metric; the
numbers here use the exact spherical areas.

Smooth fields are ~2 orders of magnitude more accurate than conserve2nd
(3rd vs 2nd order); the step (ice-edge) field is handled with zero
violations and the best error norms of any method. Full tables:
`validation/results/analytic_study.md`.

## Installation

```bash
make                      # builds libcslamregrid.so (gfortran or ifx)
pip install .             # or: PYTHONPATH=src + python -m cslam_regrid.cli
python -m pytest tests/   # 40+ tests, all exactness gates
```

On NCAR systems: `module load conda intel` (or gcc) suffices; the Python
side needs only numpy + netCDF4.

## Usage

```bash
# one-time: generate a weight file (seconds)
cslam-regrid weights --ne 30 --dst 180x360 --dst-type offset   # CMIP 1 deg
cslam-regrid weights --ne 30 --dst 257x512 --dst-type cap      # CAM-style
cslam-regrid weights --ne 30 --dst 94x192  --dst-type gauss    # Gaussian

# remap native (ncol) history files; output mimics CAM interpolated
# lat-lon history exactly (cdf5, double lat/lon, w(lat), attrs preserved)
cslam-regrid apply --map map_ne30pg3_to_offset_180x360_cslam.nc \
    run.cam.h0a.1979-*.nc --out-dir latlon/ --jobs 4 --nthreads 8 \
    --limiter shape --limiter-overrides PRECT=posdef,PRECC=posdef

cslam-regrid info  --map map.nc file.nc    # describe
cslam-regrid check --map map.nc            # re-verify exactness invariants
```

Limiter modes (per variable):
- `shape` (default): full CSLAM shape preservation — right for bounded
  fields (temperatures, fractions, mixing ratios).
- `posdef`: only enforces ≥ 0 — best accuracy for positive-definite fields
  with smooth extrema (precipitation, number concentrations): it does not
  clip legitimate maxima.
- `none`: the pure linear high-order operator (diagnostics).

Performance: a 135-level 3-D field (ne30pg3 → 257×512) remaps in ~0.15 s
on 8 cores; time-series throughput is I/O-bound.

## Notes and caveats

- Sources: any regular ne·pg3 (`--pg 2` supported); variable-resolution
  meshes are out of scope.
- Fields on other grids in the same file (e.g. GLL `ncol_d`) are skipped
  with a warning; all non-spatial variables/attributes are copied through.
- Horizontal, level-by-level only; no vertical interpolation.
- Constant fields reproduce to ~1.5e-9 (relative) in the thinnest polar
  rows — the price of assembling tiny cells from differences of exact
  line-integral potentials; invisible in float32 output and irrelevant to
  conservation (machine-exact). ESMF instead normalizes constants exactly
  but is only roundoff-conservative; we keep geometric exactness.
- v1 is unmasked (CAM). The weight-file format reserves mask/fraction
  fields for masked (CLM/fracarea) support in v2.

## Layout

```
src/fortran/        kernels: transforms, exact integrals, grid geometry,
                    reconstruction + limiter (CAM port), apply (OpenMP)
src/cslam_regrid/   Python package: grids, halo builder, exact-weight
                    walker, weight I/O, CAM-format apply engine, CLI
tests/              exactness gates (antiderivatives vs mpmath, walker
                    invariants, conservation/shape/convergence end-to-end)
validation/         the ESMF-comparison analytic study
docs/               exact-integral derivations (verified by
                    tools/derive_integrals.py)
```

## References

- Lauritzen, Nair & Ullrich (2010), JCP: CSLAM.
- Ullrich, Lauritzen & Jablonowski (2009), MWR: GECoRe exact remapping.
- CAM-SE dycore sources: `src/dynamics/se/dycore/fvm_reconstruction_mod.F90`,
  `fvm_analytic_mod.F90` (cam_development).
