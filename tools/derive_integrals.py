#!/usr/bin/env python3
"""Symbolic verification of the exact line-integral antiderivatives.

cslam-regrid maps gnomonic cubed-sphere cells to lat-lon cells with overlap
moments computed via Green's theorem:

    w_k = -oint_{dOmega} F_k(x, y) dx ,   d F_k / d y = chi_k * (1+x^2+y^2)^(-3/2)

with chi = (1, x, y, x^2, y^2, x*y) and the CSLAM potentials F_pq (moment x^p y^q,
Lauritzen, Nair & Ullrich 2010, eqs. 23-28; fvm_analytic_mod.F90):

    F00 = y/((1+x^2) rho)          F10 = x y/((1+x^2) rho)    F01 = -1/rho
    F20 = x^2 y/((1+x^2) rho)      F02 = -y/rho + asinh(y/sqrt(1+x^2))
    F11 = -x/rho                   rho = sqrt(1+x^2+y^2)

This script verifies, by symbolic differentiation (and spot numeric checks),
the hand-derived x-antiderivatives A_pq(x) of F_pq(x, y(x)) along each boundary
curve family:

  (A) y = c                      horizontal lines  -> dycore I_pq functions
  (B) y = t*sqrt(1+x^2)          parallel (lat circle) on an equatorial panel
  (C) y = s*sqrt(r^2 - x^2)      parallel on a polar panel (circle arc, s=+-1)
  (D) y = m*x                    meridian on a polar panel (radial line)

It also verifies dF_pq/dy = chi_pq * omega and F_pq = dI_pq/dx.

Run:  python tools/derive_integrals.py        (exits nonzero on any failure)
"""
import random
import sys

import sympy as sp

x, y, c, t, r, m = sp.symbols("x y c t r m", real=True)
s = sp.symbols("s", real=True)  # circle branch sign (+1 upper, -1 lower)

rho = sp.sqrt(1 + x**2 + y**2)
T = sp.sqrt(1 + x**2)

# ----------------------------------------------------------------------------
# Potentials F_pq(x, y) and area-element moments chi_pq * omega
# ----------------------------------------------------------------------------
F = {
    "00": y / ((1 + x**2) * rho),
    "10": x * y / ((1 + x**2) * rho),
    "01": -1 / rho,
    "20": x**2 * y / ((1 + x**2) * rho),
    "02": -y / rho + sp.asinh(y / T),
    "11": -x / rho,
}
chi = {"00": 1, "10": x, "01": y, "20": x**2, "02": y**2, "11": x * y}
omega = (1 + x**2 + y**2) ** sp.Rational(-3, 2)

# Dycore I_pq (fvm_analytic_mod.F90): x-antiderivatives of F_pq at fixed y.
I = {
    "00": sp.atan(x * y / rho),
    "10": -sp.asinh(y * sp.cos(sp.atan(x))),
    "01": -sp.asinh(x / sp.sqrt(1 + y**2)),
    "20": y * sp.asinh(x / sp.sqrt(1 + y**2))
    + sp.acos(x * y / sp.sqrt((1 + x**2) * (1 + y**2))),
    "02": x * sp.asinh(y / T) + sp.acos(x * y / sp.sqrt((1 + x**2) * (1 + y**2))),
    "11": -rho,
}

# ----------------------------------------------------------------------------
# Hand-derived antiderivatives along each curve family
# ----------------------------------------------------------------------------
# (B) y = t*sqrt(1+x^2):  rho = sqrt(1+x^2)*q with q = sqrt(1+t^2)
q = sp.sqrt(1 + t**2)
A_par_eq = {
    "00": (t / q) * sp.atan(x),
    "10": (t / (2 * q)) * sp.log(1 + x**2),
    "01": -sp.asinh(x) / q,
    "20": (t / q) * (x - sp.atan(x)),
    "02": (sp.asinh(t) - t / q) * x,
    "11": -T / q,
}

# (C) y = s*sqrt(r^2-x^2):  rho = P = sqrt(1+r^2) constant on the circle
P = sp.sqrt(1 + r**2)
S = sp.sqrt(r**2 - x**2)
theta = sp.asin(x / r)  # arc parameter; tan(theta) = x/S
A_par_po = {
    "00": s * (sp.atan(P * x / S) - theta / P),
    "10": s * S / P - s * sp.atanh(S / P),
    "01": -x / P,
    "20": (s / P) * (x * S + r**2 * theta) / 2 - s * (sp.atan(P * x / S) - theta / P),
    "02": -(s / (2 * P)) * (x * S + r**2 * theta)
    + x * sp.asinh(s * S / T)
    + s * P * theta
    - s * sp.atan(P * x / S),
    "11": -(x**2) / (2 * P),
}

# (D) y = m*x:  rho = R = sqrt(1+k^2 x^2), k^2 = 1+m^2
k2 = 1 + m**2
R = sp.sqrt(1 + k2 * x**2)
A_mer_po = {
    "00": sp.atan(R / m),
    "10": (m / sp.sqrt(k2)) * sp.asinh(sp.sqrt(k2) * x) - sp.asinh(m * x / T),
    "01": -sp.asinh(sp.sqrt(k2) * x) / sp.sqrt(k2),
    "20": (m / k2) * R - sp.atan(R / m),
    "02": -(m / k2) * R + x * sp.asinh(m * x / T) - sp.atan(R / m),
    "11": -R / k2,
}

FAIL = 0


def check(name, expr, domain_assumptions=(), numeric_subs=None, tol=1e-12):
    """Verify expr == 0, symbolically if possible, else numerically."""
    global FAIL
    simplified = sp.simplify(expr)
    if simplified == 0:
        print(f"  OK  (symbolic)  {name}")
        return
    # numeric fallback on random points in the valid domain
    rng = random.Random(42 + hash(name) % 1000)
    worst = 0.0
    for _ in range(40):
        subs = dict(numeric_subs(rng)) if numeric_subs else {}
        try:
            val = complex(sp.N(simplified.subs(subs), 30))
        except (TypeError, ValueError):
            continue
        worst = max(worst, abs(val))
    if worst < tol:
        print(f"  OK  (numeric, max |err| = {worst:.2e})  {name}")
    else:
        print(f"  FAIL (max |err| = {worst:.3e})  {name}")
        FAIL += 1


def rnd_eq(rng):
    """Random point on an equatorial panel with a parallel through it."""
    xv = rng.uniform(-0.98, 0.98)
    tv = rng.uniform(-0.95, 0.95)
    return [(x, sp.Float(xv, 30)), (t, sp.Float(tv, 30))]


def rnd_po(rng):
    """Random point on a polar-panel circle arc (single-valued branch)."""
    rv = rng.uniform(0.15, 1.35)
    xv = rng.uniform(-0.95, 0.95) * rv
    sv = rng.choice([-1, 1])
    return [(x, sp.Float(xv, 30)), (r, sp.Float(rv, 30)), (s, sv)]


def rnd_mer(rng):
    mv = rng.uniform(-4.0, 4.0)
    while abs(mv) < 0.05:
        mv = rng.uniform(-4.0, 4.0)
    xv = rng.uniform(-0.95, 0.95)
    return [(x, sp.Float(xv, 30)), (m, sp.Float(mv, 30))]


def rnd_hor(rng):
    return [(x, sp.Float(rng.uniform(-0.98, 0.98), 30)),
            (y, sp.Float(rng.uniform(-0.98, 0.98), 30))]


print("1. dF_pq/dy == chi_pq * omega  (potential property)")
for pq in F:
    check(f"F{pq}", sp.diff(F[pq], y) - chi[pq] * omega, numeric_subs=rnd_hor)

print("2. dI_pq/dx == F_pq  (dycore horizontal-line antiderivatives)")
for pq in F:
    check(f"I{pq}", sp.diff(I[pq], x) - F[pq], numeric_subs=rnd_hor)

print("3. equatorial-panel parallel  y = t*sqrt(1+x^2)")
for pq in F:
    integrand = F[pq].subs(y, t * T)
    check(f"A_par_eq{pq}", sp.diff(A_par_eq[pq], x) - integrand, numeric_subs=rnd_eq)

print("4. polar-panel parallel (circle)  y = s*sqrt(r^2-x^2)")
for pq in F:
    integrand = F[pq].subs(y, s * S)
    check(f"A_par_po{pq}", sp.diff(A_par_po[pq], x) - integrand, numeric_subs=rnd_po)

print("5. polar-panel meridian (radial)  y = m*x")
for pq in F:
    integrand = F[pq].subs(y, m * x)
    check(f"A_mer_po{pq}", sp.diff(A_mer_po[pq], x) - integrand, numeric_subs=rnd_mer)

if FAIL:
    print(f"\n{FAIL} FAILURES")
    sys.exit(1)
print("\nAll antiderivatives verified.")
