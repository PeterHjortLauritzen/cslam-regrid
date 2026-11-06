"""Geometrically exact overlap moments via global line walks (GECoRe method).

Per panel, every non-vertical boundary line of either grid is walked in the
panel's gnomonic plane, split into segments at every event (source grid-line
crossing, meridian crossing, parallel crossing, panel boundary), and each
segment's six potential integrals W_k = int F_k dx (closed form, see
docs/exact_integrals.md) are assigned by the uniform rule

    -W  to (source cell above, destination cell above)
    +W  to (source cell below, destination cell below)

("above/below" = limits y +- eps on the segment; for an interior source grid
line the destination cell is the same on both sides, for a destination curve
inside a source cell the source cell is; coincident cases follow the same
rule with all four neighbors distinct).  Vertical lines (source x-edges,
equatorial-panel meridians, panel seams) contribute exactly zero and are
never walked.  See DESIGN.md section 3.4.

Everything here is one-time setup; results go into the weight file.
"""
from __future__ import annotations

import numpy as np

from .grids import DestGrid, SourceGrid

TOL = 1e-13          # snapping tolerance for breakpoints (gnomonic units)
ARGTOL = 1e-10       # tangency snap: crossings with x^2 = arg < ARGTOL are
                     # tangential (grid pairings like 1-deg lat edges vs
                     # ne30pg3 produce EXACT circle/grid-line tangencies;
                     # the discarded lens area is O(arg^1.5) <~ 1e-15, and
                     # surviving near-tangencies are resolvable at the
                     # off-center evaluation points, see _eval_points)
HALFPI = np.pi / 2
TWOPI = 2 * np.pi


def _eval_points(b):
    """Per-segment side-lookup evaluation points.  Tangencies of the walked
    curves with grid lines / destination parallels occur only at x = 0 (the
    lat extremum along horizontal lines and parallels), so for a segment
    straddling 0 the lookup point moves to the half-point of the larger
    side; otherwise the midpoint is used.  (Genuine coincidences - the
    equator along y=0 - have lat constant along the line, so any point
    works and the side-aware band rules apply.)"""
    xa, xb = b[:-1], b[1:]
    xm = 0.5 * (xa + xb)
    straddle = xa * xb < 0
    alt = np.where(-xa > xb, 0.5 * xa, 0.5 * xb)
    return np.where(straddle, alt, xm)


# ---------------------------------------------------------------------------
# numpy antiderivatives (mirrors of cr_moments.F90; cross-checked in tests)
# ---------------------------------------------------------------------------

def a_horizontal(c, x):
    x = np.asarray(x, dtype=np.float64)
    rho = np.sqrt(1 + x * x + c * c)
    t2 = 1 + x * x
    q = np.sqrt(1 + c * c)
    ac = np.arccos(np.clip(x * c / np.sqrt(t2 * (1 + c * c)), -1, 1))
    return np.stack([
        np.arctan(x * c / rho),
        -np.arcsinh(c / np.sqrt(t2)),
        -np.arcsinh(x / q),
        c * np.arcsinh(x / q) + ac,
        x * np.arcsinh(c / np.sqrt(t2)) + ac,
        -rho,
    ])


def a_parallel_eq(t, x):
    x = np.asarray(x, dtype=np.float64)
    q = np.sqrt(1 + t * t)
    toq = t / q
    T = np.sqrt(1 + x * x)
    return np.stack([
        toq * np.arctan(x),
        0.5 * toq * np.log(1 + x * x),
        -np.arcsinh(x) / q,
        toq * (x - np.arctan(x)),
        (np.arcsinh(t) - toq) * x,
        -T / q,
    ])


def a_parallel_po(r, sbr, x):
    x = np.asarray(x, dtype=np.float64)
    p = np.sqrt(1 + r * r)
    s_ = np.sqrt(np.maximum(r * r - x * x, 0.0))
    th = np.arcsin(np.clip(x / r, -1, 1))
    at = np.arctan2(p * x, s_)
    xs = x * s_ + r * r * th
    a1 = sbr * (at - th / p)
    return np.stack([
        a1,
        sbr * (s_ / p - 0.5 * np.log((p + s_) / (p - s_))),
        -x / p,
        0.5 * sbr * xs / p - a1,
        -0.5 * sbr * xs / p + x * np.arcsinh(sbr * s_ / np.sqrt(1 + x * x))
        + sbr * (p * th - at),
        -x * x / (2 * p),
    ])


def a_meridian_po(m, x):
    x = np.asarray(x, dtype=np.float64)
    k2 = 1 + m * m
    k = np.sqrt(k2)
    R = np.sqrt(1 + k2 * x * x)
    ymt = np.arcsinh(m * x / np.sqrt(1 + x * x))
    atr = np.arctan(R / m)
    return np.stack([
        atr,
        (m / k) * np.arcsinh(k * x) - ymt,
        -np.arcsinh(k * x) / k,
        (m / k2) * R - atr,
        -(m / k2) * R + x * ymt - atr,
        -R / k2,
    ])


# ---------------------------------------------------------------------------
# panel-local lon/lat
# ---------------------------------------------------------------------------

def _lon(p, x, y):
    if p <= 4:
        return np.mod(np.arctan(x) + (p - 1) * HALFPI, TWOPI)
    if p == 5:
        return np.mod(np.arctan2(x, -y), TWOPI)
    return np.mod(np.arctan2(x, y), TWOPI)


def _lat(p, x, y):
    if p <= 4:
        return np.arctan(y / np.sqrt(1 + x * x))
    rho = np.sqrt(x * x + y * y)
    if p == 5:
        return HALFPI - np.arctan(rho)
    return -HALFPI + np.arctan(rho)


def _dlat_dy_sign(p, y):
    """sign of d(lat)/dy at points with ordinate y (array ok)."""
    if p <= 4:
        return np.ones_like(np.asarray(y, dtype=np.float64))
    s = -np.sign(y) if p == 5 else np.sign(y)
    return s


def _dlon_dy_sign(p, x):
    """sign of d(lon)/dy at fixed x (polar panels; 0 on equatorial)."""
    if p <= 4:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return np.sign(x) if p == 5 else -np.sign(x)


# ---------------------------------------------------------------------------
# the walker
# ---------------------------------------------------------------------------

class _Accumulator:
    def __init__(self, n, dst: DestGrid):
        self.n = n
        self.dst = dst
        self.chunks = []

    def add(self, w6, p, si, sj, dcell, sign):
        """w6: (6, nseg); si, sj, dcell: (nseg,).  Skips missing cells."""
        ok = (si >= 1) & (si <= self.n) & (sj >= 1) & (sj <= self.n)
        if not ok.all():
            w6, si, sj, dcell = w6[:, ok], si[ok], sj[ok], dcell[ok]
        if si.size:
            self.chunks.append((sign * w6, np.full(si.shape, p, dtype=np.int64),
                                si.astype(np.int64), sj.astype(np.int64),
                                dcell.astype(np.int64)))

    def collect(self):
        n, dst = self.n, self.dst
        W = np.concatenate([c[0] for c in self.chunks], axis=1)
        p = np.concatenate([c[1] for c in self.chunks])
        si = np.concatenate([c[2] for c in self.chunks])
        sj = np.concatenate([c[3] for c in self.chunks])
        dc = np.concatenate([c[4] for c in self.chunks])
        key = (((p - 1) * n + (sj - 1)) * n + (si - 1)) * dst.ncell + (dc - 1)
        order = np.argsort(key, kind="stable")
        key = key[order]
        W = W[:, order]
        uniq, start = np.unique(key, return_index=True)
        S = np.add.reduceat(W, start, axis=1)
        dcell = (uniq % dst.ncell) + 1
        cell = uniq // dst.ncell
        i = (cell % n) + 1
        j = ((cell // n) % n) + 1
        pp = (cell // (n * n)) + 1
        src = np.stack([pp, i, j]).astype(np.int32)
        return src, dcell.astype(np.int32), S


def _breakpoints(lo, hi, *event_arrays):
    """Sorted, deduped breakpoints of a walk interval [lo, hi]."""
    pts = [np.asarray([lo, hi], dtype=np.float64)]
    for ev in event_arrays:
        ev = np.asarray(ev, dtype=np.float64)
        if ev.size:
            pts.append(ev[(ev > lo + TOL) & (ev < hi - TOL)])
    b = np.sort(np.concatenate(pts))
    keep = np.concatenate(([True], np.diff(b) > TOL))
    b = b[keep]
    if b.size and hi - b[-1] < TOL and b.size > 1:
        b = np.concatenate((b[:-1], [hi]))
    return b


def _meridian_x_on_line(p, c, lam):
    """x-crossings of destination meridians with the line y = c on panel p
    (equatorial: vertical meridians never cross transversally -> handled as
    columns via lon lookup; polar: radial rays)."""
    lam = np.asarray(lam)
    if p <= 4:
        rel = np.mod(lam - (p - 1) * HALFPI + np.pi, TWOPI) - np.pi
        on = np.abs(rel) < np.pi / 4 - 1e-14
        return np.tan(rel[on])
    # polar: ray through origin; crossing with y = c at x = -c*tan(lam) (p5)
    # or x = c*tan(lam) (p6), subject to the ray-side condition r > 0.
    sgn = -1.0 if p == 5 else 1.0
    cl = np.cos(lam)
    sl = np.sin(lam)
    if abs(c) < TOL:
        # all rays pass through the origin on this line
        return np.array([0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sgn * c / cl
        x = r * sl
    ok = np.isfinite(x) & (r > TOL)
    return x[ok]


def _walk_horizontal_lines(p, src: SourceGrid, dst: DestGrid, acc: _Accumulator):
    n = src.n
    xe = src.geom["xe"]
    t_edges = np.tan(dst.lat_edges[1:-1])  # interior lat edges

    lines = [(j, xe[j]) for j in range(n + 1)]
    if p >= 5 and n % 2 == 1:
        # horizontal destination meridians (lon edges at 90/270 deg) lie on
        # y = 0, which is not a source line for odd n: walk it separately
        horiz = np.abs(np.cos(dst.lon_edges[:-1])) < 1e-14
        if horiz.any():
            lines.append((None, 0.0))

    for j, c in lines:
        # destination-curve crossings with y = c
        if p <= 4:
            # parallels: t*sqrt(1+x^2) = c
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = (c / t_edges) ** 2 - 1.0
            ok = (np.sign(t_edges) == np.sign(c)) & (arg > ARGTOL) & \
                 (np.abs(t_edges) > TOL)
            xc = np.sqrt(arg[ok])
            par_x = np.concatenate((-xc, xc))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                r_edges = np.where(np.abs(np.tan(dst.lat_edges[1:-1])) > TOL,
                                   1.0 / np.abs(np.tan(dst.lat_edges[1:-1])),
                                   np.inf)
            sgn_lat = np.sign(dst.lat_edges[1:-1])
            want = 1.0 if p == 5 else -1.0
            ok = (sgn_lat == want) & np.isfinite(r_edges) & \
                 (r_edges ** 2 - c * c > ARGTOL)
            xc = np.sqrt(r_edges[ok] ** 2 - c * c)
            par_x = np.concatenate((-xc, xc))
        mer_x = _meridian_x_on_line(p, c, dst.lon_edges[:-1])

        b = _breakpoints(-1.0, 1.0, xe, par_x, mer_x)
        if b.size < 2:
            continue
        xm = _eval_points(b)
        A = a_horizontal(c, b)
        W = A[:, 1:] - A[:, :-1]

        icol = np.searchsorted(xe, xm)                      # 1-based column
        lonm = _lon(p, xm, np.full_like(xm, c))
        latm = _lat(p, xm, np.full_like(xm, c))
        latside = _dlat_dy_sign(p, np.full_like(xm, c if c != 0 else 1e-300))

        band_above = np.where(latside >= 0,
                              dst.lat_band(latm, side="above"),
                              dst.lat_band(latm, side="below"))
        band_below = np.where(latside >= 0,
                              dst.lat_band(latm, side="below"),
                              dst.lat_band(latm, side="above"))
        if p <= 4 or abs(c) > TOL:
            # midpoints never lie on a meridian (segments split there)
            col_above = dst.lon_col(lonm, side="right")
            col_below = col_above
        else:
            # polar y=0 line: lon is exactly pi/2 (x>0) / 3pi/2 (x<0) along
            # each half, which may coincide with a destination lon edge;
            # resolve via edge-index arithmetic, never float comparison.
            lonside = _dlon_dy_sign(p, xm)
            col_above, col_below = _coincident_cols(dst, lonm, lonside)

        if j is None:
            row = np.searchsorted(xe, np.zeros_like(xm))
            acc.add(W, p, icol, row, dst.cell_id(col_above, band_above), -1.0)
            acc.add(W, p, icol, row, dst.cell_id(col_below, band_below), +1.0)
        else:
            if j + 1 <= n:
                acc.add(W, p, icol, np.full(xm.shape, j + 1, dtype=np.int64),
                        dst.cell_id(col_above, band_above), -1.0)
            if j >= 1:
                acc.add(W, p, icol, np.full(xm.shape, j, dtype=np.int64),
                        dst.cell_id(col_below, band_below), +1.0)


def _walk_parallels(p, src: SourceGrid, dst: DestGrid, acc: _Accumulator):
    n = src.n
    xe = src.geom["xe"]

    for k in range(1, dst.nlat):                 # interior lat edges
        phi = dst.lat_edges[k]
        if p <= 4:
            t = np.tan(phi)
            if abs(t) >= 1.0 - 1e-14:
                continue
            if abs(t) < TOL and n % 2 == 0:
                continue  # equator on the y=0 source line: horizontal walk
            if abs(t) > 1 / np.sqrt(2):
                X = np.sqrt(1.0 / (t * t) - 1.0)
            else:
                X = 1.0
            X = min(X, 1.0)
            # source y-line crossings: t*sqrt(1+x^2) = xe[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = (xe / t) ** 2 - 1.0 if abs(t) > TOL else \
                    np.full_like(xe, -1.0)
            ok = (np.sign(xe) == np.sign(t)) & (arg > ARGTOL)
            xc = np.sqrt(arg[ok])
            ycross = np.concatenate((-xc, xc))
            mer_x = _meridian_x_on_line_parallel_eq(p, dst.lon_edges[:-1])
            b = _breakpoints(-X, X, xe, ycross, mer_x)
            if b.size < 2:
                continue
            xm = _eval_points(b)
            ym = t * np.sqrt(1 + xm * xm)
            A = a_parallel_eq(t, b)
            W = A[:, 1:] - A[:, :-1]
            latside = 1.0
            self_add_parallel(acc, dst, p, n, xe, k, xm, ym, W, latside)
        else:
            want = 1.0 if p == 5 else -1.0
            if np.sign(phi) != want or abs(abs(phi) - HALFPI) < 1e-14:
                continue
            r = 1.0 / np.tan(abs(phi))
            if r >= np.sqrt(2.0) - 1e-14:
                continue
            for sbr in (+1.0, -1.0):
                pieces = _circle_pieces(r)
                for (xa, xb) in pieces:
                    # crossings: source y-lines, meridian rays
                    with np.errstate(invalid="ignore"):
                        arg = r * r - xe ** 2
                    ok = arg > ARGTOL
                    xc = np.sqrt(arg[ok])
                    yy = xe[ok]
                    # on branch sbr the y value must match sign: y = sbr*S
                    # crossing where sbr*sqrt(r^2-x^2) = xe[j] -> xe[j]*sbr>0
                    okb = yy * sbr > 0
                    ycross = np.concatenate((-xc[okb], xc[okb]))
                    lam = dst.lon_edges[:-1]
                    sl, cl = np.sin(lam), np.cos(lam)
                    xr = r * sl
                    yr = -r * cl if p == 5 else r * cl
                    okm = yr * sbr > TOL
                    mer_x = xr[okm]
                    b = _breakpoints(xa, xb, xe, ycross, mer_x)
                    if b.size < 2:
                        continue
                    xm = _eval_points(b)
                    ym = sbr * np.sqrt(np.maximum(r * r - xm * xm, 0))
                    A = a_parallel_po(r, sbr, b)
                    W = A[:, 1:] - A[:, :-1]
                    latside = _dlat_dy_sign(p, ym)
                    self_add_parallel(acc, dst, p, n, xe, k, xm, ym, W, latside)


def self_add_parallel(acc, dst, p, n, xe, k, xm, ym, W, latside):
    """Common contribution logic for a parallel (lat edge k) segment set."""
    icol = np.searchsorted(xe, xm)
    irow = np.searchsorted(xe, ym)
    lonm = _lon(p, xm, ym)
    col = dst.lon_col(lonm, side="right")
    band_above = np.where(np.asarray(latside) >= 0, k + 1, k)
    band_below = np.where(np.asarray(latside) >= 0, k, k + 1)
    acc.add(W, p, icol, irow, dst.cell_id(col, band_above), -1.0)
    acc.add(W, p, icol, irow, dst.cell_id(col, band_below), +1.0)


def _coincident_cols(dst: DestGrid, lon, lonside):
    """Columns above/below for segments whose constant longitude may lie
    exactly on a destination lon edge (polar y=0 lines)."""
    rel = (np.asarray(lon) - dst.lon_edges[0]) / dst.dlon
    ie = np.rint(rel)
    on_edge = np.abs(rel - ie) < 1e-9
    ie = ie.astype(np.int64)
    col_e = ie % dst.nlon + 1
    col_w = (ie - 1) % dst.nlon + 1
    plain = dst.lon_col(lon, side="right")
    col_above = np.where(on_edge, np.where(lonside > 0, col_e, col_w), plain)
    col_below = np.where(on_edge, np.where(lonside > 0, col_w, col_e), plain)
    return col_above, col_below


def _meridian_x_on_line_parallel_eq(p, lam):
    """Equatorial-panel meridian x-positions (vertical lines) used as
    breakpoints when walking a parallel across them."""
    rel = np.mod(np.asarray(lam) - (p - 1) * HALFPI + np.pi, TWOPI) - np.pi
    on = np.abs(rel) < np.pi / 4 - 1e-14
    return np.tan(rel[on])


def _circle_pieces(r):
    """x-intervals of a single-valued circle branch inside the panel square
    (|y| <= 1 clip; |x| <= 1 clip).  Returns list of (xa, xb)."""
    xmax = min(r, 1.0)
    if r * r - 1.0 <= ARGTOL:
        return [(-xmax, xmax)] if xmax > TOL else []
    x_in = np.sqrt(r * r - 1.0)          # where |y| = 1
    if x_in >= 1.0:
        return []
    return [(-xmax, -x_in), (x_in, xmax)]


def _walk_meridians_po(p, src: SourceGrid, dst: DestGrid, acc: _Accumulator):
    n = src.n
    xe = src.geom["xe"]
    t_int = dst.lat_edges[1:-1]
    want = 1.0 if p == 5 else -1.0
    with np.errstate(divide="ignore"):
        r_edges = np.where((np.sign(t_int) == want) &
                           (np.abs(np.abs(t_int) - HALFPI) > 1e-14),
                           1.0 / np.tan(np.abs(t_int)), np.nan)

    for I in range(dst.nlon):
        lam = dst.lon_edges[I]
        sl, cl = np.sin(lam), np.cos(lam)
        if abs(sl) < 1e-14:
            continue                      # vertical ray: zero contribution
        if abs(cl) < 1e-14:
            continue                      # horizontal ray: y=0 walk covers it
        m = (-cl / sl) if p == 5 else (cl / sl)
        xsign = np.sign(sl)
        # ray: x from 0 to exit of the unit square (x = xsign at most,
        # or where |y| = |m*x| = 1)
        xend = xsign * min(1.0, 1.0 / abs(m)) if abs(m) > 1 else xsign * 1.0
        # crossings
        ycross = xe / m
        okc = (ycross * xsign > TOL) & (np.abs(ycross) < abs(xend) + TOL)
        ycross = ycross[okc]
        par_x = r_edges * sl
        par_x = par_x[np.isfinite(par_x)]
        lo, hi = (0.0, xend) if xsign > 0 else (xend, 0.0)
        b = _breakpoints(lo, hi, xe, ycross, par_x)
        if b.size < 2:
            continue
        xm = 0.5 * (b[:-1] + b[1:])
        ym = m * xm
        A = a_meridian_po(m, b)
        W = A[:, 1:] - A[:, :-1]
        icol = np.searchsorted(xe, xm)
        irow = np.searchsorted(xe, ym)
        latm = _lat(p, xm, ym)
        band = dst.lat_band(latm, side="below")   # never coincident: split at parallels
        # the walked ray IS destination lon edge I: columns by index
        col_e = I % dst.nlon + 1
        col_w = (I - 1) % dst.nlon + 1
        lonside = xsign if p == 5 else -xsign     # sign of dlon/dy at fixed x
        col_above = col_e if lonside > 0 else col_w
        col_below = col_w if lonside > 0 else col_e
        ca = np.full(xm.shape, col_above, dtype=np.int64)
        cb = np.full(xm.shape, col_below, dtype=np.int64)
        acc.add(W, p, icol, irow, dst.cell_id(ca, band), -1.0)
        acc.add(W, p, icol, irow, dst.cell_id(cb, band), +1.0)


def generate_weights(src: SourceGrid, dst: DestGrid, check=True):
    """Compute the exact overlap moments.  Returns dict with
    src (3, npair) int32 (p,i,j), dst (npair,) int32 1-based cell ids,
    S (6, npair) float64, and diagnostics."""
    acc = _Accumulator(src.n, dst)
    for p in range(1, 7):
        _walk_horizontal_lines(p, src, dst, acc)
        _walk_parallels(p, src, dst, acc)
        if p >= 5:
            _walk_meridians_po(p, src, dst, acc)
    psrc, pdst, S = acc.collect()

    diags = {}
    if check:
        area_b = dst.areas()
        got_b = np.bincount(pdst - 1, weights=S[0], minlength=dst.ncell)
        diags["dst_area_relerr"] = float(
            (np.abs(got_b - area_b) / area_b).max())
        diags["dst_area_abserr"] = float(np.abs(got_b - area_b).max())
        n = src.n
        g = src.geom
        lin = ((psrc[0].astype(np.int64) - 1) * n
               + (psrc[2].astype(np.int64) - 1)) * n + (psrc[1].astype(np.int64) - 1)
        amom = np.empty((6, n, n))
        amom[0] = g["area"]
        for kk in range(5):
            amom[kk + 1] = g["centroid"][kk] * g["area"]
        # exact per-cell moments, panel-replicated in (p,j,i) linear order
        # (amom[k] is indexed [i,j]; the linear key has i fastest)
        exact_panel = np.stack([amom[kk].reshape(-1, order="F")
                                for kk in range(6)])
        exact = np.concatenate([exact_panel] * 6, axis=1)
        scale = max(g["area"].max(), 1e-30)
        err = 0.0
        for kk in range(6):
            got = np.bincount(lin, weights=S[kk], minlength=6 * n * n)
            err = max(err, np.abs(got - exact[kk]).max() / scale)
        diags["src_moment_relerr"] = float(err)

    return dict(src=psrc, dst=pdst, S=S, diags=diags)
