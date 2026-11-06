#!/usr/bin/env python3
"""Validation campaign: CSLAM vs the ESMF methods of the cmip7-interp study.

Replicates the ne30pg3 -> 1 deg / 2 deg analytic study
(/glade/campaign/cgd/amp/pel/src/cmip7-interp) with the CSLAM remapper added:

  methods:  bilinear / conserve / conserve2nd   (ESMF weight files from the
            study, applied to the same input)
            cslam-none / cslam-shape / cslam-posdef (this package)

  fields:   step70n (ice-edge surrogate), bell (panel-corner cosine bell),
            precip (positive-definite surrogate), constant

  inputs:   EXACT spherical cell averages (20x20 Gauss-Legendre per gnomonic
            cell for the smooth fields; the step's cell averages are computed
            exactly as polar-cap area fractions with the exact line-integral
            machinery itself).  References are exact destination-cell
            averages.  Note: the cmip7-interp SUMMARY used cell-center point
            samples instead, so absolute numbers differ slightly; the
            comparison here is internally consistent (finite-volume in,
            finite-volume reference).

  metrics:  cmip7-interp conventions -- normalized l1/l2/linf with global
            denominators, regional masks by cell-center latitude; global
            conservation; negatives; violations of the global source bounds.

Usage: python validation/run_validation.py  (writes validation/results/)
"""
import os
import sys
import time

import numpy as np
import netCDF4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CMIP7 = "/glade/campaign/cgd/amp/pel/src/cmip7-interp"
sys.path.insert(0, CMIP7)

import analytic_fields as af                      # noqa: E402
from cslam_regrid import fkernels as fk           # noqa: E402
from cslam_regrid.grids import DestGrid, SourceGrid  # noqa: E402
from cslam_regrid.halo import build_halo          # noqa: E402
from cslam_regrid.walker import generate_weights  # noqa: E402

FIELDS = ["step70n", "bell", "precip", "constant"]
TARGETS = {"180x360": (180, 360), "90x180": (90, 180)}
OUTDIR = os.path.join(ROOT, "validation", "results")


# ---------------------------------------------------------------------------
# exact cell averages
# ---------------------------------------------------------------------------

def src_cell_averages_smooth(src, func, nq=20):
    """(6,n,n) exact spherical cell averages of func(lon_deg, lat_deg)."""
    n = src.n
    xe = src.geom["xe"]
    gx, gw = np.polynomial.legendre.leggauss(nq)
    x1, x2 = xe[:-1][:, None], xe[1:][:, None]
    xq = 0.5 * (x1 + x2) + 0.5 * (x2 - x1) * gx
    wq = 0.5 * (x2 - x1) * gw
    out = np.empty((6, n, n))
    for p in range(1, 7):
        for j in range(n):
            X = np.repeat(xq, nq, axis=1).reshape(n, nq, nq)
            Y = np.tile(xq[j], (n, nq, 1))
            W = (np.repeat(wq, nq, axis=1).reshape(n, nq, nq)
                 * np.tile(wq[j], (n, nq, 1)))
            omega = (1 + X ** 2 + Y ** 2) ** -1.5
            lon, lat = fk.lonlat_from_panel(X.ravel(), Y.ravel(),
                                            np.full(X.size, p, np.int32))
            F = func(np.rad2deg(lon), np.rad2deg(lat)).reshape(n, nq, nq)
            out[p - 1, :, j] = ((F * omega * W).sum(axis=(1, 2))
                                / (omega * W).sum(axis=(1, 2)))
    return out


def src_cell_averages_step(src, lat0_deg=70.0):
    """Exact cell averages of the step via polar-cap area fractions,
    computed with the exact line-integral walker (2-band destination)."""
    band = DestGrid(
        kind="offset", nlat=2, nlon=1,
        lat_edges=np.array([-np.pi / 2, np.deg2rad(lat0_deg), np.pi / 2]),
        lon_edges=np.array([0.0, 2 * np.pi]),
        lat_centers=np.array([-0.1, np.deg2rad(80.0)]),
        lon_centers=np.array([np.pi]))
    w = generate_weights(src, band, check=False)
    n = src.n
    north = w["dst"] == 2
    lin = ((w["src"][0][north].astype(np.int64) - 1) * n
           + (w["src"][2][north] - 1)) * n + (w["src"][1][north] - 1)
    area_north = np.bincount(lin, weights=w["S"][0][north],
                             minlength=6 * n * n)
    area = np.tile(src.geom["area"].reshape(-1, order="F"), 6)
    frac = np.clip(area_north / area, 0.0, 1.0)
    out = np.empty((6, n, n))
    for p in range(6):
        out[p] = frac[p * n * n:(p + 1) * n * n].reshape(n, n, order="F")
    return out


def dst_cell_averages_smooth(dst, func, nq=20):
    gx, gw = np.polynomial.legendre.leggauss(nq)
    lam1, lam2 = dst.lon_edges[:-1], dst.lon_edges[1:]
    mu1, mu2 = np.sin(dst.lat_edges[:-1]), np.sin(dst.lat_edges[1:])
    lam = 0.5 * (lam1 + lam2)[:, None] + 0.5 * (lam2 - lam1)[:, None] * gx
    mu = 0.5 * (mu1 + mu2)[:, None] + 0.5 * (mu2 - mu1)[:, None] * gx
    out = np.empty((dst.nlat, dst.nlon))
    for J in range(dst.nlat):
        lat = np.rad2deg(np.arcsin(np.clip(mu[J], -1, 1)))
        F = func(np.rad2deg(lam[:, None, :]), lat[None, :, None])
        out[J] = (F * gw[None, :, None] * gw[None, None, :]).sum(axis=(1, 2)) / 4
    return out.ravel()


def dst_cell_averages_step(dst, lat0_deg=70.0):
    s0 = np.sin(np.deg2rad(lat0_deg))
    s1, s2 = np.sin(dst.lat_edges[:-1]), np.sin(dst.lat_edges[1:])
    frac = np.clip((s2 - s0), 0, None) - np.clip((s1 - s0), 0, None)
    frac = frac / (s2 - s1)
    return np.repeat(np.clip(frac, 0, 1), dst.nlon)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

class EsmfMap:
    def __init__(self, path):
        ds = netCDF4.Dataset(path)
        self.row = ds.variables["row"][:].filled() - 1
        self.col = ds.variables["col"][:].filled() - 1
        self.S = ds.variables["S"][:].filled()
        self.n_b = len(ds.dimensions["n_b"])
        self.xc_a = ds.variables["xc_a"][:].filled()
        self.yc_a = ds.variables["yc_a"][:].filled()
        self.yc_b = ds.variables["yc_b"][:].filled()
        self.area_a = ds.variables["area_a"][:].filled()
        self.area_b = ds.variables["area_b"][:].filled()
        ds.close()

    def apply(self, f):
        return np.bincount(self.row, weights=self.S * f[self.col],
                           minlength=self.n_b)


def norms(dst_val, exact, area, lat_deg):
    out = {}
    denom1 = np.sum(np.abs(exact) * area)
    denom2 = np.sqrt(np.sum(exact ** 2 * area))
    denomi = np.abs(exact).max()
    for name, (lo, hi) in {"global": (-90, 90), "60-90N": (60, 90)}.items():
        m = (lat_deg >= lo) & (lat_deg <= hi)
        e = dst_val[m] - exact[m]
        a = area[m]
        out[name] = dict(
            l1=np.sum(np.abs(e) * a) / denom1 if denom1 else np.nan,
            l2=np.sqrt(np.sum(e * e * a)) / denom2 if denom2 else np.nan,
            linf=np.abs(e).max() / denomi if denomi else np.nan)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    src = SourceGrid.from_ne(30)
    halo = build_halo(src.n)
    n = src.n

    print("computing exact source cell averages ...")
    t0 = time.time()
    favg = {name: src_cell_averages_smooth(src, getattr(af, name))
            for name in ("bell", "precip", "constant")}
    favg["step70n"] = src_cell_averages_step(src)
    print(f"  ({time.time()-t0:.0f}s)")

    rows = []
    for tname, (nlat, nlon) in TARGETS.items():
        dst = DestGrid.create("offset", nlat, nlon)
        area_b = dst.areas()
        lat_b = np.rad2deg(np.repeat(dst.lat_centers, dst.nlon))

        print(f"{tname}: cslam weights ...")
        w = generate_weights(src, dst, check=False)
        pairs = dict(src=w["src"], dst=w["dst"], S=w["S"])

        exact_b = {name: dst_cell_averages_smooth(
            dst, getattr(af, name)) for name in ("bell", "precip",
                                                 "constant")}
        exact_b["step70n"] = dst_cell_averages_step(dst)

        esmf = {m: EsmfMap(os.path.join(
            CMIP7, "output", "weights",
            f"map_ne30pg3_to_latlon_{tname}_{m}.nc"))
            for m in ("bilinear", "conserve", "conserve2nd")}

        # map our (p,i,j) cell averages into the ESMF/SCRIP ncol order
        any_map = esmf["conserve"]
        col2pij = src.col2pij(any_map.xc_a, any_map.yc_a)
        gather = ((col2pij[0].astype(np.int64) - 1) * n
                  + (col2pij[2] - 1)) * n + (col2pij[1] - 1)
        area_a = np.tile(src.geom["area"].reshape(-1, order="F"), 6)[gather]

        for fname in FIELDS:
            fsrc = np.concatenate(
                [favg[fname][p].reshape(-1, order="F") for p in range(6)]
            )[gather]
            exact = exact_b[fname]
            lo, hi = fsrc.min(), fsrc.max()

            results = {}
            for m, emap in esmf.items():
                results[m] = emap.apply(fsrc)
            for lim, mname in ((0, "cslam-none"), (1, "cslam-shape"),
                               (2, "cslam-posdef")):
                results[mname] = fk.apply_slices(
                    n, dst.ncell, col2pij, halo, src.geom, pairs, area_b,
                    lim, fsrc)[:, 0]

            for mname, val in results.items():
                nm = norms(val, exact, area_b, lat_b)
                cons = ((val * area_b).sum() - (fsrc * area_a).sum()) \
                    / np.abs(fsrc * area_a).sum()
                # count violations beyond 1e-6 of the field range (well above
                # the ~1e-9 relative wiring noise of the thinnest polar
                # cells, far below any physically meaningful violation)
                tol = 1e-6 * max(hi - lo, abs(hi), 1e-300)
                neg = val < -tol
                under = val < lo - tol
                over = val > hi + tol
                rows.append(dict(
                    target=tname, field=fname, method=mname,
                    l1=nm["global"]["l1"], l2=nm["global"]["l2"],
                    linf=nm["global"]["linf"],
                    l2_cap=nm["60-90N"]["l2"],
                    conservation=cons,
                    n_under=int(under.sum()), n_over=int(over.sum()),
                    max_under=float(np.maximum(lo - val, 0).max()),
                    max_over=float(np.maximum(val - hi, 0).max()),
                    n_neg=int(neg.sum()),
                    most_neg=float(min(val.min(), 0.0))))
            print(f"  {fname} done")

    # ---- write CSV + markdown
    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(OUTDIR, "analytic_study.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3e}"
        return str(x)

    with open(os.path.join(OUTDIR, "analytic_study.md"), "w") as f:
        f.write("# ne30pg3 -> lat-lon analytic validation "
                "(exact cell-average inputs)\n\n")
        for tname in TARGETS:
            for fname in FIELDS:
                f.write(f"\n## {tname} / {fname}\n\n")
                f.write("| method | l1 | l2 | linf | l2 (60-90N) | "
                        "conservation | n<lo / n>hi | max under/over | "
                        "n_neg | most neg |\n|---|---|---|---|---|---|---"
                        "|---|---|---|\n")
                for r in rows:
                    if r["target"] != tname or r["field"] != fname:
                        continue
                    f.write(
                        f"| {r['method']} | {fmt(r['l1'])} | {fmt(r['l2'])} "
                        f"| {fmt(r['linf'])} | {fmt(r['l2_cap'])} | "
                        f"{fmt(r['conservation'])} | {r['n_under']} / "
                        f"{r['n_over']} | {fmt(r['max_under'])} / "
                        f"{fmt(r['max_over'])} | {r['n_neg']} | "
                        f"{fmt(r['most_neg'])} |\n")
    print(f"wrote {OUTDIR}/analytic_study.{{csv,md}}")


if __name__ == "__main__":
    main()
