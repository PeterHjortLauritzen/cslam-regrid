"""Source (ne*pg3) and destination (lat-lon) grid definitions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fkernels as fk

HALFPI = np.pi / 2
TWOPI = 2 * np.pi


# ---------------------------------------------------------------------------
# source grid
# ---------------------------------------------------------------------------

@dataclass
class SourceGrid:
    """Regular ne*pg physics grid, whole-panel layout (n = fv_nphys*ne)."""
    n: int
    geom: dict = field(repr=False)

    @classmethod
    def from_n(cls, n):
        return cls(n=n, geom=fk.grid_geometry(n))

    @classmethod
    def from_ne(cls, ne, fv_nphys=3):
        return cls.from_n(ne * fv_nphys)

    @property
    def ncol(self):
        return 6 * self.n * self.n

    def col2pij(self, lon_deg, lat_deg):
        """Map the file's ncol column centers to (panel,i,j); validates a
        bijection onto the 6*n*n cells."""
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        if lon.size != self.ncol:
            raise ValueError(
                f"file has ncol={lon.size}, grid expects {self.ncol} "
                f"(n={self.n})")
        x, y, p = fk.panel_from_lonlat(lon, lat)
        d = HALFPI / self.n
        i = np.floor((np.arctan(x) + np.pi / 4) / d).astype(np.int64) + 1
        j = np.floor((np.arctan(y) + np.pi / 4) / d).astype(np.int64) + 1
        i = np.clip(i, 1, self.n)
        j = np.clip(j, 1, self.n)
        lin = ((p.astype(np.int64) - 1) * self.n + (j - 1)) * self.n + (i - 1)
        counts = np.bincount(lin, minlength=self.ncol)
        if counts.max() != 1:
            bad = int((counts != 1).sum())
            raise ValueError(
                f"ncol -> cell mapping is not a bijection ({bad} cells off); "
                "is this file really on the expected pg grid?")
        out = np.empty((3, self.ncol), dtype=np.int32)
        out[0] = p
        out[1] = i
        out[2] = j
        return out

    def col2pij_partial(self, lon_deg, lat_deg, colok):
        """col2pij for files whose coordinates are only defined on a subset
        of columns (CLM writes _FillValue lat/lon over ocean).  Columns with
        colok False get the placeholder target (1,1,1); the masked apply
        kernel never scatters them.  The valid subset must be injective."""
        colok = np.asarray(colok, dtype=bool)
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64)[colok])
        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64)[colok])
        if colok.size != self.ncol:
            raise ValueError(
                f"file has ncol={colok.size}, grid expects {self.ncol} "
                f"(n={self.n})")
        x, y, p = fk.panel_from_lonlat(lon, lat)
        d = HALFPI / self.n
        i = np.floor((np.arctan(x) + np.pi / 4) / d).astype(np.int64) + 1
        j = np.floor((np.arctan(y) + np.pi / 4) / d).astype(np.int64) + 1
        i = np.clip(i, 1, self.n)
        j = np.clip(j, 1, self.n)
        lin = ((p.astype(np.int64) - 1) * self.n + (j - 1)) * self.n + (i - 1)
        counts = np.bincount(lin, minlength=self.ncol)
        if counts.max() > 1:
            bad = int((counts > 1).sum())
            raise ValueError(
                f"ncol -> cell mapping is not injective ({bad} cells hit "
                "more than once); is this file really on the expected "
                "pg grid?")
        out = np.ones((3, self.ncol), dtype=np.int32)
        out[0, colok] = p
        out[1, colok] = i
        out[2, colok] = j
        return out


# ---------------------------------------------------------------------------
# destination grids
# ---------------------------------------------------------------------------

@dataclass
class DestGrid:
    """Regular lat-lon destination grid.

    lat_edges: (nlat+1,) radians, strictly increasing, spanning [-pi/2, pi/2]
    lon_edges: (nlon+1,) radians, uniform spacing, lon_edges[0] may be
               negative (wrap); cell I spans [lon_edges[I-1], lon_edges[I]].
    Cell index (1-based, Fortran): id = (J-1)*nlon + I  (lon fastest).
    """
    kind: str
    nlat: int
    nlon: int
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    lat_centers: np.ndarray
    lon_centers: np.ndarray

    @classmethod
    def create(cls, kind, nlat, nlon):
        """kind: 'offset' (CMIP-style, centers offset from poles),
        'cap' (equally spaced with poles, CAM interpolated-history style),
        'gauss' (Gaussian latitudes)."""
        dlon = TWOPI / nlon
        if kind == "offset":
            lat_edges = np.linspace(-HALFPI, HALFPI, nlat + 1)
            lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
            lon_centers = (np.arange(nlon) + 0.5) * dlon
            lon_edges = np.arange(nlon + 1) * dlon
        elif kind == "cap":
            lat_centers = np.linspace(-HALFPI, HALFPI, nlat)
            lat_edges = np.empty(nlat + 1)
            lat_edges[0] = -HALFPI
            lat_edges[-1] = HALFPI
            lat_edges[1:-1] = 0.5 * (lat_centers[:-1] + lat_centers[1:])
            lon_centers = np.arange(nlon) * dlon
            lon_edges = (np.arange(nlon + 1) - 0.5) * dlon
        elif kind == "gauss":
            nodes, wts = np.polynomial.legendre.leggauss(nlat)
            lat_centers = np.arcsin(nodes)
            sin_edges = -1.0 + np.concatenate(([0.0], np.cumsum(wts)))
            sin_edges[0], sin_edges[-1] = -1.0, 1.0
            lat_edges = np.arcsin(np.clip(sin_edges, -1, 1))
            lon_centers = (np.arange(nlon) + 0.5) * dlon
            lon_edges = np.arange(nlon + 1) * dlon
        else:
            raise ValueError(f"unknown destination grid kind '{kind}'")
        return cls(kind, nlat, nlon, lat_edges, lon_edges,
                   lat_centers, lon_centers)

    @property
    def ncell(self):
        return self.nlat * self.nlon

    @property
    def dlon(self):
        return TWOPI / self.nlon

    def areas(self):
        """Exact spherical cell areas, (ncell,) in cell-id order."""
        band = np.sin(self.lat_edges[1:]) - np.sin(self.lat_edges[:-1])
        return np.repeat(band, self.nlon) * self.dlon

    def lat_weights(self):
        """CAM-convention latitude weights w(lat), sum = 2."""
        return np.sin(self.lat_edges[1:]) - np.sin(self.lat_edges[:-1])

    def lon_col(self, lon, side="left"):
        """1-based lon column of angle(s) `lon` (radians, any branch).
        side='right' assigns a point exactly on an edge to the column east
        of it, side='left' to the column west of it."""
        rel = (np.asarray(lon) - self.lon_edges[0]) / self.dlon
        if side == "right":
            col = np.floor(rel).astype(np.int64) + 1
        else:
            col = np.ceil(rel).astype(np.int64)
        return (col - 1) % self.nlon + 1

    def lat_band(self, lat, side="below"):
        """1-based lat band of angle(s) `lat`.  For a point exactly on an
        interior edge, side='above' returns the band above it, side='below'
        the band below."""
        s = "right" if side == "above" else "left"
        band = np.searchsorted(self.lat_edges, np.asarray(lat), side=s) - 1
        return np.clip(band, 0, self.nlat - 1) + 1

    def cell_id(self, col, band):
        return (np.asarray(band, dtype=np.int64) - 1) * self.nlon + col
