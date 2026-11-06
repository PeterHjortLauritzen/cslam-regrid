# Build libcslamregrid.so (Fortran kernels for the cslam-regrid Python package).
#
#   make            optimized build
#   make DEBUG=1    bounds checking + traps
#   make FC=ifx     Intel compiler

FC      ?= gfortran
DEBUG   ?=
BUILDDIR = build
LIB      = src/cslam_regrid/libcslamregrid.so

SRC = src/fortran/cr_kinds.F90 \
      src/fortran/cr_gnomonic.F90 \
      src/fortran/cr_moments.F90 \
      src/fortran/cr_grid.F90 \
      src/fortran/cr_recon.F90 \
      src/fortran/cr_apply.F90 \
      src/fortran/cr_capi.F90

ifeq ($(findstring ifx,$(FC)),ifx)
  FFLAGS_OPT   = -O2 -fp-model precise -qopenmp
  FFLAGS_DEBUG = -O0 -g -check bounds -fpe0 -traceback -qopenmp
  MODFLAG      = -module $(BUILDDIR)
else
  FFLAGS_OPT   = -O2 -fopenmp
  FFLAGS_DEBUG = -O0 -g -fcheck=bounds -ffpe-trap=invalid,zero,overflow -fbacktrace -fopenmp
  MODFLAG      = -J$(BUILDDIR)
endif

FFLAGS = $(if $(DEBUG),$(FFLAGS_DEBUG),$(FFLAGS_OPT)) -fPIC $(MODFLAG)

$(LIB): $(SRC) | $(BUILDDIR)
	$(FC) $(FFLAGS) -shared -o $@ $(SRC)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(LIB)

.PHONY: clean
