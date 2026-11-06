import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def pytest_configure(config):
    """(Re)build the Fortran library if any source is newer."""
    lib = os.path.join(ROOT, "src", "cslam_regrid", "libcslamregrid.so")
    srcdir = os.path.join(ROOT, "src", "fortran")
    sources = [os.path.join(srcdir, f) for f in os.listdir(srcdir) if f.endswith(".F90")]
    sources.append(os.path.join(ROOT, "Makefile"))
    if not os.path.exists(lib) or any(os.path.getmtime(s) > os.path.getmtime(lib) for s in sources):
        subprocess.run(["make", "-C", ROOT], check=True)
