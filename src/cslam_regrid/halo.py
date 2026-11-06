"""Panel-halo construction (one-time setup, stored in the weight file).

Two halo structures feed the reconstruction, mirroring CAM-SE:

* interpolated halo (``hi_*``): rings 1-2 of the extended panel array
  ``fext(-1:n+2, -1:n+2)``, filled by ns=3-point Lagrange interpolation of
  the neighbor panel's ring-h cell averages along the equiangular edge
  coordinate (the whole-panel equivalent of ``extend_panel_interpolate`` /
  ``create_interpolation_points``).  A halo cell center lies exactly on the
  neighbor's ring-h center line; only the along-edge position needs
  interpolation.  Stencils are clamped to the neighbor row ([1, n-2] base),
  which at cube corners reproduces CAM's one-sided extrapolation.  The
  north/south fills own the corner-diagonal blocks (along range -1..n+2);
  east/west fills cover along range 1..n.

* raw ring-1 halo (``li_*``): index-aligned neighbor cell averages in
  ``flim(0:n+1, 0:n+1)`` for the limiter's 3x3 patch bounds (CAM's limiter
  operates on the exchanged ``fcube``, not the interpolated extension),
  plus CAM's cube-corner diagonal fill rule (the net effect of the
  ``slope_limiter`` corner copies):
     NW: (0,n+1)   <- (1,n+1)      SW: (0,0)     <- (0,1)
     SE: (n+1,0)   <- (n+1,1)      NE: (n+1,n+1) <- (n+1,n)
"""
from __future__ import annotations

import numpy as np

from . import fkernels as fk

NS = 3     # Lagrange interpolation points (quadratic), CAM ns for pg3
NHR = 2    # halo rings

_EDGES = ("S", "N", "W", "E")


def _center_angle(i, d):
    """Center angle of (possibly extended) index i, 1-based interior."""
    return -np.pi / 4 + (i - 0.5) * d


def _home_halo_point(edge, i, h, d):
    """Gnomonic coords (home panel) of halo cell center (edge, along i, ring h)."""
    a = np.tan(_center_angle(i, d))
    b = np.tan(np.pi / 4 + (h - 0.5) * d)
    if edge == "N":
        return a, b
    if edge == "S":
        return a, -b
    if edge == "E":
        return b, a
    return -b, a  # W


def _edge_neighbor(p, edge, d):
    """Neighbor panel across an edge and that panel's shared-edge id."""
    x, y = _home_halo_point(edge, 1, 0.75, d)  # a point just outside midside-ish
    lon, lat = fk.lonlat_from_panel(np.array([x]), np.array([y]), p)
    xnb, ynb, pnb = fk.panel_from_lonlat(lon, lat)
    nb = int(pnb[0])
    if abs(xnb[0]) >= abs(ynb[0]):
        nb_edge = "E" if xnb[0] > 0 else "W"
    else:
        nb_edge = "N" if ynb[0] > 0 else "S"
    return nb, nb_edge


def _nb_ring_cell(nb_edge, h, m, n):
    """(i,j) of the neighbor's ring-h, along-index-m cell, from its edge."""
    if nb_edge == "S":
        return m, h
    if nb_edge == "N":
        return m, n + 1 - h
    if nb_edge == "W":
        return h, m
    return n + 1 - h, m  # E


def _dst_cell(edge, i, h, n):
    """(i,j) of the home halo slot (ring h beyond `edge`, along index i)."""
    if edge == "N":
        return i, n + h
    if edge == "S":
        return i, 1 - h
    if edge == "E":
        return n + h, i
    return 1 - h, i  # W


def build_halo(n):
    d = (np.pi / 2) / n
    hi_dst, hi_src, hi_w = [], [], []
    li_dst, li_src = [], []

    for p in range(1, 7):
        for edge in _EDGES:
            nb, nb_edge = _edge_neighbor(p, edge, d)

            # ---- interpolated halo ----
            along = range(-1, n + 3) if edge in ("N", "S") else range(1, n + 1)
            for h in (1, 2):
                for i in along:
                    x, y = _home_halo_point(edge, i, h, d)
                    lon, lat = fk.lonlat_from_panel(np.array([x]), np.array([y]), p)
                    xnb, ynb, valid = fk.panel_point_from_lonlat(lon, lat, nb)
                    assert valid[0], f"halo point behind horizon p{p}{edge} i={i} h={h}"
                    if nb_edge in ("E", "W"):
                        along_c, depth_c = ynb[0], xnb[0]
                    else:
                        along_c, depth_c = xnb[0], ynb[0]
                    # consistency: the point sits on the neighbor ring-h center line
                    expect = np.pi / 4 - (h - 0.5) * d
                    assert abs(abs(np.arctan(depth_c)) - expect) < 1e-10, \
                        f"ring mismatch p{p}{edge} i={i} h={h}"
                    theta = np.arctan(along_c)
                    k = (theta + np.pi / 4) / d + 0.5      # real-valued center index
                    ibase = int(np.clip(round(k) - 1, 1, n - NS + 1))
                    xi = (theta - _center_angle(ibase, d)) / d  # in units of d, node 0 at ibase
                    w = np.array([
                        (xi - 1) * (xi - 2) / 2.0,
                        -xi * (xi - 2),
                        xi * (xi - 1) / 2.0,
                    ])
                    src = [( nb, *_nb_ring_cell(nb_edge, h, ibase + m, n)) for m in range(NS)]
                    hi_dst.append((p, *_dst_cell(edge, i, h, n)))
                    hi_src.append(src)
                    hi_w.append(w)

            # ---- raw ring-1 halo (index aligned) ----
            for i in range(1, n + 1):
                x, y = _home_halo_point(edge, i, 0.5, d)  # quarter-cell beyond edge
                lon, lat = fk.lonlat_from_panel(np.array([x]), np.array([y]), p)
                xnb, ynb, valid = fk.panel_point_from_lonlat(lon, lat, nb)
                assert valid[0]
                along_c = ynb[0] if nb_edge in ("E", "W") else xnb[0]
                m = int(np.floor((np.arctan(along_c) + np.pi / 4) / d)) + 1
                m = min(max(m, 1), n)
                li_dst.append((p, *_dst_cell(edge, i, 1, n)))
                li_src.append((nb, *_nb_ring_cell(nb_edge, 1, m, n)))

        # ---- limiter cube-corner diagonal fill (after the edge slots) ----
        li_dst.append((p, 0, n + 1));     li_src.append((p, 1, n + 1))      # NW
        li_dst.append((p, 0, 0));         li_src.append((p, 0, 1))          # SW
        li_dst.append((p, n + 1, 0));     li_src.append((p, n + 1, 1))      # SE
        li_dst.append((p, n + 1, n + 1)); li_src.append((p, n + 1, n))      # NE

    return dict(
        hi_dst=np.array(hi_dst, dtype=np.int32).T,
        hi_src=np.array(hi_src, dtype=np.int32).transpose(2, 1, 0),
        hi_w=np.array(hi_w, dtype=np.float64).T,
        li_dst=np.array(li_dst, dtype=np.int32).T,
        li_src=np.array(li_src, dtype=np.int32).T,
    )
