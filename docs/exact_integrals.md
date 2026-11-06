# Exact line integrals for the CSLAM quadratic basis

All overlap moments are computed via Green's theorem with dx-potentials:

    w_pq = -∮_{∂Ω} F_pq(x, y) dx ,   ∂F_pq/∂y = x^p y^q (1+x²+y²)^(-3/2)

with (p,q) ∈ {00, 10, 01, 20, 02, 11} (order: area, x, y, x², y², xy) and the
CSLAM potentials (Lauritzen, Nair & Ullrich 2010 JCP, eqs. 23–28;
`fvm_analytic_mod.F90`), ρ = √(1+x²+y²), T = √(1+x²):

    F00 = y/(T²ρ)      F10 = xy/(T²ρ)      F01 = -1/ρ
    F20 = x²y/(T²ρ)    F02 = -y/ρ + asinh(y/T)      F11 = -x/ρ

Every boundary curve of an overlap region between a gnomonic cell
[x₁,x₂]×[y₁,y₂] and a lat-lon cell belongs to one of five families. In a
panel's gnomonic plane, with the segment parameterized by x, the antiderivative
A_pq(x) = ∫ F_pq(x, y(x)) dx is **elementary** for every family (verified by
`tools/derive_integrals.py`; run it after any change):

## (0) Vertical lines (x = const)

Source x-edges, equatorial-panel meridians, panel seams: dx = 0 ⇒ contribution
is exactly **zero**.

## (A) Horizontal lines (y = c)

Source y-edges (and destination curves that degenerate to y = const).
A_pq = I_pq(x, c), the dycore antiderivatives (`fvm_analytic_mod.F90`,
I_00 … I_11):

    I00 = atan(xy/ρ)                  I10 = -asinh(y/T)·¹
    I01 = -asinh(x/√(1+y²))           I20 = y·asinh(x/√(1+y²)) + acos(xy/(T√(1+y²)))
    I02 = x·asinh(y/T) + acos(xy/(T√(1+y²)))        I11 = -ρ

¹ written in the code as -asinh(y·cos(atan x)).

## (B) Parallel on an equatorial panel: y = t·T, t = tan(lat)

ρ factors: ρ = T·q, q = √(1+t²). With that:

    A00 = (t/q)·atan(x)              A10 = (t/(2q))·ln(1+x²)
    A01 = -asinh(x)/q                A20 = (t/q)·(x - atan x)
    A02 = (asinh(t) - t/q)·x         A11 = -T/q

(F02 is *constant* along the curve.)

## (C) Parallel on a polar panel: circle x² + y² = r², r = cot|lat|

Branch y = s·S, S = √(r²-x²), s = ±1 (arcs split at y = 0 so each segment is a
single-valued graph). ρ = P = √(1+r²) is constant on the circle;
θ = asin(x/r):

    A00 = s·( atan(Px/S) - θ/P )
    A10 = s·S/P - s·atanh(S/P)
    A01 = -x/P
    A20 = (s/P)·(xS + r²θ)/2 - s·( atan(Px/S) - θ/P )
    A02 = -(s/(2P))·(xS + r²θ) + x·asinh(sS/T) + sPθ - s·atan(Px/S)
    A11 = -x²/(2P)

Numerical notes: use atan2(P·x, S) so arc endpoints (S→0, x→±r) evaluate
cleanly to ±π/2; clamp x/r to [-1,1] in θ; S < P always, so atanh is safe.

## (D) Meridian on a polar panel: radial line y = m·x

(m = ±tan/cot of the longitude in panel coordinates; the m→∞ line is x = 0,
vertical ⇒ zero; m = 0 is horizontal ⇒ family (A).) With k² = 1+m²,
R = √(1+k²x²):

    A00 = atan(R/m)
    A10 = (m/k)·asinh(kx) - asinh(mx/T)
    A01 = -asinh(kx)/k
    A20 = (m/k²)·R - atan(R/m)
    A02 = -(m/k²)·R + x·asinh(mx/T) - atan(R/m)
    A11 = -R/k²

## Usage and sign convention

A segment from x_a → x_b (x_a < x_b) carries W_pq = A_pq(x_b) - A_pq(x_a).
With counterclockwise (positively oriented) region boundaries,
∬ = -∮ F dx means each walked segment contributes

    -W to (source-cell-above, dest-cell-above)
    +W to (source-cell-below, dest-cell-below)

(above/below = limits y±ε on the segment; see DESIGN.md §3.4). Antiderivatives
are defined up to per-curve constants, which cancel within each segment; branch
cuts are never crossed inside a segment because segments are split at every
crossing and at circle y=0 points.
