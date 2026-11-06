# cslam-regrid — design document

Conservative, shape-preserving, geometrically exact remapping of CESM/CAM native
gnomonic cubed-sphere physics-grid (ne·pg3) history output to latitude-longitude
grids, using the CSLAM reconstruction technology from the CAM-SE dycore.

Status: design locked 2026-11-06 (decisions confirmed by P. Lauritzen); implementation
in progress.

---

## 1. Motivation

The CMIP7 regridding study (`/glade/campaign/cgd/amp/pel/src/cmip7-interp/`,
SUMMARY.md) quantified the available ESMF options for ne30pg3 → 1° lat-lon:

| method | conservative | shape-preserving | quality |
|---|---|---|---|
| bilinear | no (7–12 % error on masked integrals) | yes | smooth |
| conserve (1st order) | machine-exact | yes | polar blockiness, panel-seam imprint (~5× cap error) |
| conserve2nd | machine-exact | **no** | negative precip to −1.3 mm/day at coasts, 5.3 K overshoots |

No existing method is simultaneously conservative, shape-preserving, and high-order.
CSLAM's reconstruct-and-limit technology is exactly that combination:

- **High order**: quadratic polynomial reconstruction per source cell (3rd-order,
  the CAM-SE/CSLAM `irecons=6` production configuration).
- **Conservative**: overlap integrals of the reconstruction; the limiter scales only
  the deviation from the cell mean, so conservation is exact regardless of limiting.
- **Shape-preserving**: the Barth–Jespersen-type CSLAM limiter guarantees the
  reconstruction lies within the local min/max of the 3×3 neighborhood, hence every
  overlap average does too. No clipping, no post-hoc renormalization.
- **Geometrically exact**: all overlap moment integrals are evaluated with
  closed-form line integrals (the GECoRe approach, Ullrich, Lauritzen & Jablonowski
  2009 MWR), not quadrature or polygon approximations.

## 2. Scope (locked decisions)

- **Source**: regular ne·pg3 physics grids (gnomonic cubed-sphere FV, 3×3 cells per
  spectral element; globally equiangular, 3·ne × 3·ne cells per panel). ne30pg3 is
  the primary target; any regular ne works. Variable-resolution meshes: out of scope.
- **Destination**: arbitrary regular lat-lon (uniform Δλ; uniform Δφ either
  cell-centered CMIP-style or "equally spaced with poles" CAM-style) **and Gaussian
  latitudes**. Built-in presets: CMIP 1° (180×360, centers ±89.5°…), 2° (90×180),
  CAM SE default interpolated grids (e.g. 257×512 equally spaced with poles).
- **Fields**: unmasked CAM `ncol` fields in v1. The weight-file format reserves
  mask/fraction slots so masked (CLM/fracarea) support can be added in v2 without a
  format change.
- **Vertical**: horizontal-only, level-by-level on model levels. No vertical
  interpolation.
- **Limiter**: per-variable control — `shape` (default; full shape-preserving CSLAM
  limiter), `posdef` (bound from below by 0 only), `none` (pure linear high-order map).
- **Output**: mimics CAM's own interpolated lat-lon history files exactly
  (see §7) so downstream CMIP/AMWG/ADF workflows ingest it unchanged.
- **Architecture**: Python package + CLI; all numerics in Fortran
  (`iso_c_binding` shared library called via ctypes — no f2py/meson dependency).
- **Parallelism**: OpenMP over (time, level) slices inside the Fortran apply kernel;
  process-level parallelism over variables/files from Python. Single-node design.
- **Everything precomputed**: grid geometry, reconstruction geometry, and all overlap
  moments live in a weight file. Only the limiter (field-dependent by nature) runs at
  apply time — on precomputed geometric factors.

## 3. Mathematical formulation

### 3.1 Coordinates and grids

Each cube panel ν ∈ {0..5} (0–3 equatorial, 4 north, 5 south — GECoRe/HOMME
convention) carries gnomonic coordinates x = tan α, y = tan β, α,β ∈ [−π/4, π/4].
The spherical area element is dA = ω(x,y) dx dy with ω = (1 + x² + y²)^(−3/2).

The ne·pg3 physics grid is globally equiangular: N = 3·ne cells per panel edge,
cell (i,j) = [x_i, x_{i+1}] × [y_j, y_{j+1}], x_i = tan(−π/4 + (i−1)Δα),
Δα = (π/2)/N. Cell edges are straight lines in the gnomonic plane (great circles).

The destination cell (I,J) = {λ ∈ [λ_I, λ_{I+1}]} × {φ ∈ [φ_J, φ_{J+1}]}, with
exact spherical area (λ_{I+1}−λ_I)(sin φ_{J+1} − sin φ_J). In a panel's gnomonic
plane its boundaries are:

| curve | equatorial panels | polar panels |
|---|---|---|
| meridian λ=const | straight **vertical** line x = tan(λ−λ_ν) | straight **radial** line y = ∓x·tan(λ)⁻¹-form (through origin) |
| parallel φ=const | y = t·√(1+x²), t = tan φ | circle x² + y² = cot² φ |

### 3.2 Reconstruction (ported from `fvm_reconstruction_mod.F90`, irecons=6)

For each source cell with spherical cell average f̄ the reconstruction is, in
centroid form,

  f(x,y) = f̄ + r₂ δx + r₃ δy + r₄(δx² + m₁) + r₅(δy² + m₂) + r₆(δx δy + m₃),

δx = x − c_x, δy = y − c_y, with centroid c = (⟨x⟩, ⟨y⟩) and metric constants
m₁ = c_x² − ⟨x²⟩, m₂ = c_y² − ⟨y²⟩, m₃ = c_x c_y − ⟨xy⟩ (⟨·⟩ = spherical area
average over the cell, computed from closed-form antiderivatives I₀₀…I₁₁, §3.4).
By construction the spherical cell mean of f is exactly f̄ for any coefficients.

Raw derivatives use the dycore's 4th-order centered differences on the equiangular
index grid (5-point: −f₊₂+8f₊₁−8f₋₁+f₋₂ etc.; cross term from the 4 diagonal
neighbors), converted to x-derivatives at the centroid with the chain-rule factors
(`centroid_stretch(1..7)`), identically to CAM. Stencil: 13-point diamond, halo
width 2.

**Halo across panel edges**: the tool works with full-panel arrays
f(1−h : N+h, 1−h : N+h, 6), h = 2. Halo cells of a panel are filled from the
neighbor panel by the dycore's method: 1-D Lagrange interpolation (ns = 3,
quadratic, matching CAM's physics-grid configuration) of neighbor-panel cell
averages along the edge-parallel direction onto the home panel's extended
coordinate lines (`extend_panel_interpolate` logic, restructured for whole panels —
no element decomposition needed). Cube corners: the 2×2 diagonal halo block does
not exist geometrically; the reconstruction never uses it (diamond stencil), and
for the limiter's 3×3 min/max the dycore's corner-fill rule is replicated.

Panel-coordinate consistency: every source cell's reconstruction and all overlap
integrals against it are expressed in *its own* panel's (x,y) — destination-cell
boundary curves are transformed into each panel as needed — so no coefficient
rotation between panels is required at apply time (rotations appear only inside the
halo-derivative logic, as integer `rot_matrix` factors, exactly as in CAM).

### 3.3 Limiter (ported from `slope_limiter`, irecons=6 path)

Per cell and field: bounds [M⁻, M⁺] = min/max of the 3×3 patch of cell averages
(across panel edges via the halo). Candidate extrema of the quadratic over the cell
(the cell *is* its gnomonic bounding box, since edges are x=const/y=const):

1. the 4 vertices — evaluated with precomputed 5-term vertex weights,
2. the interior stationary point (if 4r₄r₅ − r₆² ≠ 0 and inside the cell),
3. up to 4 edge-line extrema (∂f/∂x = 0 on y-edges, ∂f/∂y = 0 on x-edges, if
   inside the edge span).

One scalar φ = min(1, (M± − f̄)/(extremum − f̄)) over violated bounds scales
(r₂…r₆) → φ·(r₂…r₆). Mean untouched ⇒ conservation exact.
`posdef` mode: same algorithm with bounds [0, +∞). `none`: φ ≡ 1.

Because the limited polynomial lies in [M⁻, M⁺] pointwise over the cell, every
destination average (a convex combination of pointwise values) is in-bounds:
shape preservation is inherited exactly, including for partial overlaps.

### 3.4 Geometrically exact overlap moments (GECoRe method, new derivation)

For each overlap Ω = (source cell ∩ destination cell) we need the six moments
  w_k = ∬_Ω χ_k(x,y) ω dx dy, χ = (1, x, y, x², y², xy),
so that the destination integral of the reconstruction is Σ_k w_k c_k with c the
absolute-monomial coefficients (converted from centroid form exactly as in
`reconstruction`'s final select).

Green's theorem with dx-potentials: w_k = −∮_∂Ω P_k(x, y) dx, where
∂P_k/∂y = χ_k ω — these are CSLAM's F₁…F₆ (paper eqs. 23–28, as in
`fvm_analytic_mod.F90`):

  F₁ = y/((1+x²)ρ), F₂ = xy/((1+x²)ρ), F₃ = −1/ρ,
  F₄ = x²y/((1+x²)ρ), F₅ = −y/ρ + asinh(y/√(1+x²)), F₆ = −x/ρ, ρ = √(1+x²+y²).

Per boundary-curve family, the antiderivatives ∫P_k dx are **elementary closed
forms** (verified; full derivations to be documented in `docs/exact_integrals.md`
and tested against adaptive quadrature to machine precision):

- **x = const** (source x-edges, equatorial-panel meridians): contribution ≡ 0.
- **y = const** (source y-edges): the dycore's exact antiderivatives I₀₀…I₁₁
  evaluated at segment endpoints (as `get_weights_exact` does).
- **Parallel, equatorial panel** (y = t√(1+x²)): ρ = √(1+x²)√(1+t²) factors; e.g.
  ∫F₁dx = (t/√(1+t²))·atan x, ∫F₂dx = (t/(2√(1+t²)))·ln(1+x²),
  ∫F₃dx = −asinh(x)/√(1+t²), ∫F₄dx = (t/√(1+t²))(x − atan x),
  F₅ = asinh(t) − t/√(1+t²) = const, ∫F₆dx = −√(1+x²)/√(1+t²).
- **Parallel, polar panel** (x² + y² = cot²φ): ρ = √(1+cot²φ) constant on the
  curve; trig-rational closed forms in the arc parameter (the F₅ asinh term is
  derived separately; if any term resists an elementary form it falls back to
  Gauss–Kronrod evaluated to machine ε, flagged in provenance — exactness is a
  *tested* property either way).
- **Meridian, polar panel** (radial line y = m·x): ρ = √(1+(1+m²)x²);
  standard closed forms (atan/asinh/log family).

**Overlap decomposition — global line walks (GECoRe architecture).** No 2-D
polygons are ever formed, and no connectivity assumptions are made (this matters:
for odd N = 3·ne the pole lies *inside* a source cell and a polar-cap destination
cell can be wholly interior to it). Because our basis is absolute monomials per
panel (no per-cell recentering of potentials, unlike GECoRe's centered basis),
the assembly collapses to one uniform rule. Per panel, walk every
non-vertical boundary line of either grid, split it into segments at every event
(source x/y-line crossing, meridian crossing, parallel crossing, panel boundary),
and for each segment from x_a→x_b compute W_k = ∫ F_k dx in closed form. Then:

  **each segment contributes −W_k to pair (S_above, D_above) and +W_k to
  (S_below, D_below)**,

where S_above/S_below are the source cells and D_above/D_below the destination
cells immediately above/below the segment (evaluated at y ± ε). For an interior
source grid line D_above = D_below (one destination cell, two source cells); for
a destination curve inside a source cell S_above = S_below; coincident cases
(equator on the panel y=0 grid line; polar-panel lon edges at 90°/270° lying on
the y=0 line) are handled by the same rule with all four neighbors distinct.
Vertical lines (source x-edges, equatorial-panel meridians, panel seams x=±1)
contribute exactly zero with dx-potentials and are skipped.

Curves walked per panel: the N+1 horizontal source grid lines; every parallel
arc present on the panel (equatorial: y = t√(1+x²), present where
|t|√(1+x²) ≤ 1, i.e. all parallels with |φ| ≤ 45°, partial arcs for
35.26° < |φ| < 45°; polar: circle arcs x²+y² = cot²φ); and on polar panels the
meridians (radial lines y = m·x). Crossing abscissae are closed-form
(x = ±√((y/t)²−1), x = ±√(r²−y²), radial/line intersections). Pole caps need no
special case: D_above at the pole is the cap cell. Floating-point robustness
follows GECoRe: scale-aware snapping (1e-12), tangency tolerance, zero-length
segments dropped; the ±ε cell queries are index computations, not float
comparisons, with coincidence resolved by the snapping step.

Floating-point robustness follows GECoRe: snapping tolerance (1e-12 scale-aware),
coincident-line rules (lat edge on the equator = panel y=0 line; lon edge on a
panel seam), degenerate/zero-length segments dropped. Weight generation is
OpenMP-parallel over source cells.

**Exactness invariants (tested to machine precision):**
1. Σ over source cells of w₁ per destination cell = (λ₂−λ₁)(sin φ₂ − sin φ₁).
2. Σ over destination cells of w_k per source cell = that cell's exact moment
   (I-form corner sums) for all six k.
3. A globally constant field maps to exactly 1 everywhere; each moment of a global
   quadratic in (x,y) per panel is reproduced exactly where the stencil is interior.

Unlike `dp_mapping.F90`'s area rescaling, no renormalization is applied — the
integrals are exact, and invariant (1) is a *test*, not a fix-up. (A diagnostic
reports the residual; if it exceeds 1e-13 relative, weight generation aborts.)

### 3.5 Apply

Per 2-D slice (one level, one time, one variable):
1. scatter ncol → 6 panel arrays (precomputed index map),
2. fill panel halos (precomputed interpolation weights; pure sparse ops),
3. compute centroid-form coefficients (precomputed stencil + geometric factors),
4. limit per cell (mode per variable),
5. convert to absolute monomials (precomputed per-cell constants),
6. destination value(I,J) = Σ_pairs w·c / A_dest (CSR-like precomputed sparsity).

All steps are linear except (4). With `none` the whole operator could be collapsed
to a single sparse matrix; we keep the staged form for uniformity (and the staged
form is still fast: ~0.3 M overlap pairs × 6 flops for ne30pg3 → 1°).

OpenMP parallelism over slices (collapse time × lev); the Python layer runs
variables (and files) in parallel processes. Computation in float64 throughout;
output variables written in the input's dtype (float32 for standard CAM history).

## 4. Weight file (netCDF, self-describing)

One file per (source grid, destination grid): `map_<src>_to_<dst>_cslam.nc`.
Everything the apply stage needs — no access to grid files at apply time.

Dimensions: `n_s` (overlap pairs), `n_a` (=6N² source cells), `n_b` (destination
cells), `nmom=6`, `ncoef=6`, `nlat`, `nlon`, panel geometry dims.

Groups/variables (CDL sketch):
- source grid: ne, N=3ne, panel edge angles; ncol → (panel,i,j) index map `col`;
  cell centroids (c_x, c_y), metric terms m₁..m₃, centroid_stretch(7),
  vertex limiter weights (4×5), absolute-conversion constants; halo interpolation
  weights/indices (quadratic Lagrange, per edge/corner, both rings);
  exact cell areas; `grid_imask_a` (all 1 in v1 — reserved for masked v2).
- destination grid: lat, lon (double), lat/lon bounds, exact cell areas,
  latitude weights `w` (CAM convention), grid-type tag
  (`uniform_offset | uniform_with_poles | gaussian`), `frac_b` (reserved, =1).
- map: `row` (dest index), `panel/i/j` of source cell, `S(n_s, nmom)` moments.
- global attrs: tool version, git hash, generation date/host, source/dest grid
  descriptions, exactness-invariant residuals (recorded), format version.

ESMF-style naming (`n_s`, `row`, `col`, `S`) is used where semantics match, so the
file is familiar, but it is **not** an ESMF map (S has 6 moments per pair).
Optionally `--emit-esmf-map` writes, in addition, a true ESMF/SCRIP-format
first-order map (w₁ normalized) usable by ncremap/ESMF as a by-product.

## 5. CLI

```
cslam-regrid weights --ne 30 --dst 180x360        [--dst-type cmip|cap|gauss]
                     [--dst-scrip file.nc] [--out map_ne30pg3_to_1deg_cslam.nc]
                     [--emit-esmf-map] [--nthreads N]

cslam-regrid apply   --map map_ne30pg3_to_1deg_cslam.nc in1.nc [in2.nc ...]
                     [--out-dir DIR | --out file.nc]
                     [--vars PRECT,TS,...]           (default: all ncol vars)
                     [--limiter shape|posdef|none]   (global default)
                     [--limiter-overrides PRECT=posdef,OMEGA=none|file.yaml]
                     [--nthreads N] [--jobs J]       (J files in parallel)

cslam-regrid info    --map file.nc | --file history.nc   (describe grids/vars)
cslam-regrid check   --map file.nc                        (re-verify invariants)
```

Destination presets: `180x360` / `90x180` (CMIP offset), `--dst-type cap`
(equally spaced with poles, CAM-style, e.g. `257x512`), `--dst-type gauss`
(Gaussian latitudes, `nlat` even). Arbitrary `NLATxNLON` accepted for each type.

## 6. Package layout

```
cslam-regrid/
  DESIGN.md  README.md  LICENSE  pyproject.toml  Makefile
  src/fortran/
    cr_kinds.F90            precision/kind constants
    cr_gnomonic.F90         panel transforms (lat-lon <-> x,y,panel), conventions
    cr_moments.F90          I/F antiderivatives + parallel/meridian closed forms
    cr_grid.F90             pg3 panel grid geometry, centroids, metrics, stretch
    cr_halo.F90             panel halo fill (Lagrange weights, corner rules)
    cr_recon.F90            reconstruction (ported irecons=6 path)
    cr_limiter.F90          shape/posdef/none limiter (ported slope_limiter)
    cr_overlap.F90          exact overlap moment generation (equatorial/polar/cap)
    cr_apply.F90            staged apply kernel, OpenMP
    cr_capi.F90             iso_c_binding entry points
  src/cslam_regrid/         Python package
    __init__.py  cli.py  fkernels.py (ctypes)  grids.py  weights_io.py
    history_io.py (CAM-format output writer)  apply.py  limiters.py  version.py
  tests/
    test_moments.py         antiderivatives vs mpmath/adaptive quadrature
    test_invariants.py      weight-sum identities to machine precision
    test_recon.py           quadratic reproduction, convergence order
    test_limiter.py         bounds never violated; mean preserved
    test_roundtrip.py       constant field, analytic fields (reuse cmip7-interp)
    test_output_format.py   CAM-history structural fidelity
  docs/exact_integrals.md   full derivations of all antiderivatives
  validation/               driver reproducing the cmip7-interp study + CSLAM column
```

Build: `make` → `libcslamregrid.so` (gfortran or ifx, `-O2 -fPIC -fopenmp`,
float64). Python loads via ctypes; `pip install .` for the package (the Makefile is
invoked by a thin build hook; no meson/f2py).

## 7. Output format (CAM interpolated-history fidelity)

Template: CAM SE interpolated history (verified against
`*.cam.h0a.*.nc`, cdf5): copy **all** dimensions, variables and attributes that do
not carry `ncol`, unchanged and in order (time, time_bounds, date, datesec, lev,
ilev, hyam/hybm/hyai/hybi, ndbase/…/mdt, auxiliary dims like trop_*); replace
`ncol` by (lat, lon): `double lat(lat)`, `double lon(lon)` with CAM attributes,
`double w(lat)` latitude weights (exact band integrals of cos φ, Gaussian weights
for gauss grids — CAM convention, sums to 2); each field becomes
`float VAR(time[, lev], lat, lon)` with its original attributes (incl.
`cell_methods`, `mdims`) preserved verbatim. Global attributes preserved, plus:
`interp_type = "CSLAM: conservative, shape-preserving, geometrically exact"`,
`interp_outputgri` set to the matching CAM vocabulary string, and provenance
(`cslam_regrid_version`, map file, limiter settings per variable appended to
`history`). File format matches input (cdf5/netcdf4 passthrough).

Variables on other grids (`ncol_d` GLL, etc.): copied through unchanged? No —
skipped with a clear warning (they cannot share the lat/lon dims coherently);
`--vars` can restrict explicitly. Non-ncol data variables are copied verbatim.

## 8. Validation & acceptance (gates for "production")

1. **Unit exactness**: every antiderivative vs 50-digit mpmath quadrature
   (< 1e-14 rel); weight invariants (§3.4) < 1e-13 rel for ne30pg3 → each preset.
2. **Reconstruction**: reproduces global-per-panel quadratics exactly (interior);
   3rd-order convergence on smooth fields (ne16/30/60 sequence or fixed grid vs
   refined targets).
3. **Remap quality** (cmip7-interp harness, same fields, same norms): conservation
   = roundoff; **zero** shape violations with `shape`; zero negatives with
   `posdef`; smooth-field errors ≤ conserve2nd, polar-cap l2 ≈ bilinear-smooth
   (no blockiness, no seam imprint).
4. **Format fidelity**: output passes cf-checker; structural diff vs a real CAM
   interpolated file shows only expected differences; a monthly 3-D file
   (135 lev × 48600) remaps in seconds/file on one node.
5. Idempotent, deterministic (bit-identical reruns, thread-count independent sums
   via fixed-order reductions in the per-destination accumulation).

## 9. Implementation stages

1. Fortran: gnomonic/panel module + pg3 grid geometry + exact cell moments; tests.
2. Fortran: antiderivative library (all curve families); mpmath cross-check.
3. Fortran: overlap generation (equatorial, polar, caps) + invariant checks.
4. Fortran: halo + reconstruction + limiter port; quadratic/limiter tests.
5. Fortran: apply kernel + OpenMP; C API.
6. Python: grids, weight I/O, CLI `weights`; generate ne30pg3→presets maps.
7. Python: `apply` + CAM-format writer; format-fidelity tests.
8. Validation campaign (cmip7-interp harness) → tables/figures; README.
9. Performance pass (threading, I/O chunking); packaging polish.

Reference materials: dycore sources (fvm_reconstruction_mod.F90,
fvm_analytic_mod.F90, fvm_overlap_mod.F90, dp_mapping.F90), GECoRe C++
(search/crossing formulas, robustness rules), CSLAM (Lauritzen, Nair & Ullrich
2010 JCP) and GECoRe (Ullrich, Lauritzen & Jablonowski 2009 MWR) papers.
