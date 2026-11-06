"""cslam-regrid command-line interface.

    cslam-regrid weights --ne 30 --dst 180x360 [--dst-type offset|cap|gauss]
                         [--out MAP.nc] [--force]
    cslam-regrid apply   --map MAP.nc IN.nc [IN2.nc ...]
                         [--out FILE.nc | --out-dir DIR] [--vars A,B,...]
                         [--limiter shape|posdef|none]
                         [--limiter-overrides VAR=MODE,VAR=MODE]
                         [--nthreads N] [--jobs J]
    cslam-regrid info    --map MAP.nc | FILE.nc
    cslam-regrid check   --map MAP.nc
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import sys
import time

import numpy as np

from . import __version__


def _parse_dst(spec):
    m = re.fullmatch(r"(\d+)x(\d+)", spec)
    if not m:
        raise argparse.ArgumentTypeError(
            f"destination grid '{spec}' not of the form NLATxNLON")
    return int(m.group(1)), int(m.group(2))


def cmd_weights(args):
    from .grids import DestGrid, SourceGrid
    from .halo import build_halo
    from .walker import generate_weights
    from .weights_io import write_weights

    nlat, nlon = args.dst
    out = args.out or (f"map_ne{args.ne}pg{args.pg}_to_"
                       f"{args.dst_type}_{nlat}x{nlon}_cslam.nc")
    if os.path.exists(out) and not args.force:
        sys.exit(f"{out} exists (use --force to overwrite)")

    t0 = time.time()
    src = SourceGrid.from_ne(args.ne, fv_nphys=args.pg)
    dst = DestGrid.create(args.dst_type, nlat, nlon)
    print(f"source: ne{args.ne}pg{args.pg} (n={src.n}, ncol={src.ncol})")
    print(f"destination: {args.dst_type} {nlat}x{nlon}")
    print("building halo ...")
    halo = build_halo(src.n)
    print("computing exact overlap moments ...")
    w = generate_weights(src, dst, check=True)
    d = w["diags"]
    print(f"  pairs: {w['dst'].size}")
    print(f"  invariants: dst area abs err {d['dst_area_abserr']:.2e}, "
          f"src moment rel err {d['src_moment_relerr']:.2e}")
    if d["dst_area_abserr"] > 1e-12 or d["src_moment_relerr"] > 1e-11:
        sys.exit("ERROR: exactness invariants violated; not writing map")
    write_weights(out, src, dst, halo,
                  dict(src=w["src"], dst=w["dst"], S=w["S"]), d, ne=args.ne,
                  history=" ".join(sys.argv))
    print(f"wrote {out} ({time.time()-t0:.1f}s)")


def _apply_one(weight_path, infile, outfile, variables, limiter, overrides,
               nthreads):
    from .apply import RemapEngine
    eng = RemapEngine(weight_path, nthreads=nthreads)
    t0 = time.time()
    print(f"{infile} -> {outfile}")
    eng.remap_file(infile, outfile, variables=variables,
                   default_limiter=limiter, limiter_overrides=overrides)
    print(f"  done ({time.time()-t0:.1f}s)")


def cmd_apply(args):
    overrides = {}
    if args.limiter_overrides:
        for item in args.limiter_overrides.split(","):
            name, mode = item.split("=")
            if mode not in ("shape", "posdef", "none"):
                sys.exit(f"unknown limiter mode '{mode}' for {name}")
            overrides[name.strip()] = mode.strip()
    variables = args.vars.split(",") if args.vars else None

    if args.out and len(args.infiles) > 1:
        sys.exit("--out only valid with a single input file; use --out-dir")
    jobs = []
    for infile in args.infiles:
        if args.out:
            outfile = args.out
        else:
            base = os.path.basename(infile)
            stem, ext = os.path.splitext(base)
            outfile = os.path.join(args.out_dir or ".",
                                   f"{stem}.cslam_latlon{ext}")
        if os.path.abspath(outfile) == os.path.abspath(infile):
            sys.exit(f"refusing to overwrite input {infile}")
        jobs.append((infile, outfile))

    if args.jobs > 1 and len(jobs) > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_apply_one, args.map, i, o, variables,
                              args.limiter, overrides, args.nthreads)
                    for i, o in jobs]
            for f in futs:
                f.result()
    else:
        for i, o in jobs:
            _apply_one(args.map, i, o, variables, args.limiter, overrides,
                       args.nthreads)


def cmd_info(args):
    import netCDF4
    if args.map:
        from .weights_io import read_weights
        w = read_weights(args.map)
        a = w["attrs"]
        print(f"weight file: {args.map}")
        for key in ("title", "method", "source_grid", "dest_grid",
                    "cslam_regrid_version", "created", "host",
                    "dst_area_abserr", "src_moment_relerr", "history"):
            if key in a:
                print(f"  {key}: {a[key]}")
        print(f"  pairs: {w['pairs']['dst'].size}")
    for f in args.files:
        ds = netCDF4.Dataset(f)
        ncol = len(ds.dimensions["ncol"]) if "ncol" in ds.dimensions else None
        nvar = sum(1 for v in ds.variables.values()
                   if "ncol" in v.dimensions and v.dimensions[-1] == "ncol")
        print(f"{f}: ncol={ncol}, remappable variables: {nvar}, "
              f"format={ds.data_model}")
        if ncol:
            n = int(round((ncol / 6) ** 0.5))
            if 6 * n * n == ncol and n % 3 == 0:
                print(f"  looks like ne{n//3}pg3 (n={n})")
        ds.close()


def cmd_check(args):
    from .weights_io import read_weights
    w = read_weights(args.map)
    n = w["n"]
    area_b = w["area_b"]
    got = np.bincount(w["pairs"]["dst"] - 1, weights=w["pairs"]["S"][0],
                      minlength=area_b.size)
    abserr = np.abs(got - area_b).max()
    amom = w["geom"]["area"]
    lin = ((w["pairs"]["src"][0].astype(np.int64) - 1) * n
           + (w["pairs"]["src"][2] - 1)) * n + (w["pairs"]["src"][1] - 1)
    exact = np.tile(amom.reshape(-1, order="F"), 6)
    gots = np.bincount(lin, weights=w["pairs"]["S"][0], minlength=6 * n * n)
    srcerr = np.abs(gots - exact).max() / amom.max()
    print(f"dst area abs err: {abserr:.3e}")
    print(f"src area rel err: {srcerr:.3e}")
    ok = abserr < 1e-12 and srcerr < 1e-11
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cslam-regrid",
        description="Conservative, shape-preserving, geometrically exact "
                    "remapping of CAM ne*pg3 physics-grid output to lat-lon "
                    "using CSLAM reconstruction technology.")
    ap.add_argument("--version", action="version",
                    version=f"cslam-regrid {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("weights", help="generate a weight file")
    pw.add_argument("--ne", type=int, required=True)
    pw.add_argument("--pg", type=int, default=3,
                    help="physics grid cells per element edge (default 3)")
    pw.add_argument("--dst", type=_parse_dst, required=True,
                    metavar="NLATxNLON")
    pw.add_argument("--dst-type", choices=("offset", "cap", "gauss"),
                    default="offset",
                    help="offset: CMIP-style cell-centered; cap: equally "
                         "spaced with poles (CAM style); gauss: Gaussian")
    pw.add_argument("--out")
    pw.add_argument("--force", action="store_true")
    pw.set_defaults(func=cmd_weights)

    pa = sub.add_parser("apply", help="remap native history files")
    pa.add_argument("--map", required=True)
    pa.add_argument("infiles", nargs="+")
    pa.add_argument("--out")
    pa.add_argument("--out-dir")
    pa.add_argument("--vars", help="comma-separated subset (default: all)")
    pa.add_argument("--limiter", choices=("shape", "posdef", "none"),
                    default="shape")
    pa.add_argument("--limiter-overrides", metavar="VAR=MODE,...")
    pa.add_argument("--nthreads", type=int, default=0,
                    help="OpenMP threads for the apply kernel")
    pa.add_argument("--jobs", type=int, default=1,
                    help="files remapped in parallel processes")
    pa.set_defaults(func=cmd_apply)

    pi = sub.add_parser("info", help="describe a weight file / history file")
    pi.add_argument("--map")
    pi.add_argument("files", nargs="*")
    pi.set_defaults(func=cmd_info)

    pc = sub.add_parser("check", help="re-verify weight file invariants")
    pc.add_argument("--map", required=True)
    pc.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
