module cr_capi
  !-----------------------------------------------------------------------
  ! iso_c_binding entry points for the Python layer (ctypes).
  ! Every routine here is a thin wrapper; all logic lives in the cr_* modules.
  !-----------------------------------------------------------------------
  use iso_c_binding, only: c_double, c_int, c_int64_t
  use cr_kinds
  use cr_gnomonic
  use cr_moments
  use cr_grid,  only: panel_grid_t, grid_init
  use cr_apply, only: apply_slices, slice_recon, apply_slices_masked
  implicit none

contains

  subroutine c_set_num_threads(nt) bind(C, name="cr_set_num_threads")
    !$ use omp_lib, only: omp_set_num_threads
    integer(c_int), intent(in), value :: nt
    !$ if (nt > 0) call omp_set_num_threads(nt)
  end subroutine c_set_num_threads

  !----------------------------- transforms ------------------------------

  subroutine c_lonlat_from_panel(n, x, y, ip, lon, lat) bind(C, name="cr_lonlat_from_panel")
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in)  :: x(n), y(n)
    integer(c_int), intent(in)  :: ip(n)
    real(c_double), intent(out) :: lon(n), lat(n)
    integer :: i
    do i = 1, n
      call lonlat_from_panel(x(i), y(i), int(ip(i)), lon(i), lat(i))
    end do
  end subroutine c_lonlat_from_panel

  subroutine c_panel_from_lonlat(n, lon, lat, x, y, ip) bind(C, name="cr_panel_from_lonlat")
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in)  :: lon(n), lat(n)
    real(c_double), intent(out) :: x(n), y(n)
    integer(c_int), intent(out) :: ip(n)
    integer :: i, ipl
    do i = 1, n
      call panel_from_lonlat(lon(i), lat(i), x(i), y(i), ipl)
      ip(i) = ipl
    end do
  end subroutine c_panel_from_lonlat

  subroutine c_cart_from_panel(n, x, y, ip, cart) bind(C, name="cr_cart_from_panel")
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in)  :: x(n), y(n)
    integer(c_int), intent(in)  :: ip(n)
    real(c_double), intent(out) :: cart(3, n)
    integer :: i
    do i = 1, n
      call cart_from_panel(x(i), y(i), int(ip(i)), cart(:, i))
    end do
  end subroutine c_cart_from_panel

  subroutine c_panel_point_from_lonlat(n, lon, lat, ip, x, y, valid) &
       bind(C, name="cr_panel_point_from_lonlat")
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in)  :: lon(n), lat(n)
    integer(c_int), intent(in), value :: ip
    real(c_double), intent(out) :: x(n), y(n)
    integer(c_int), intent(out) :: valid(n)
    integer :: i
    logical :: v
    do i = 1, n
      call panel_point_from_lonlat(lon(i), lat(i), ip, x(i), y(i), v)
      valid(i) = merge(1, 0, v)
    end do
  end subroutine c_panel_point_from_lonlat

  !--------------------------- exact integrals ---------------------------

  subroutine c_seg_w6_horizontal(c, xa, xb, w) bind(C, name="cr_seg_w6_horizontal")
    real(c_double), intent(in), value :: c, xa, xb
    real(c_double), intent(out) :: w(6)
    call seg_w6_horizontal(c, xa, xb, w)
  end subroutine c_seg_w6_horizontal

  subroutine c_seg_w6_parallel_eq(t, xa, xb, w) bind(C, name="cr_seg_w6_parallel_eq")
    real(c_double), intent(in), value :: t, xa, xb
    real(c_double), intent(out) :: w(6)
    call seg_w6_parallel_eq(t, xa, xb, w)
  end subroutine c_seg_w6_parallel_eq

  subroutine c_seg_w6_parallel_po(r, sbr, xa, xb, w) bind(C, name="cr_seg_w6_parallel_po")
    real(c_double), intent(in), value :: r, sbr, xa, xb
    real(c_double), intent(out) :: w(6)
    call seg_w6_parallel_po(r, sbr, xa, xb, w)
  end subroutine c_seg_w6_parallel_po

  subroutine c_seg_w6_meridian_po(m, xa, xb, w) bind(C, name="cr_seg_w6_meridian_po")
    real(c_double), intent(in), value :: m, xa, xb
    real(c_double), intent(out) :: w(6)
    call seg_w6_meridian_po(m, xa, xb, w)
  end subroutine c_seg_w6_meridian_po

  subroutine c_cell_moments(x1, x2, y1, y2, mom) bind(C, name="cr_cell_moments")
    real(c_double), intent(in), value :: x1, x2, y1, y2
    real(c_double), intent(out) :: mom(6)
    call cell_moments(x1, x2, y1, y2, mom)
  end subroutine c_cell_moments

  !------------------------- grid geometry -------------------------------

  subroutine c_grid_geometry(n, xe, xc, area, centroid, metrics, &
                             metrics_int, stretch, vtxw) bind(C, name="cr_grid_geometry")
    integer(c_int), intent(in), value :: n
    real(c_double), intent(out) :: xe(0:n), xc(n), area(n, n)
    real(c_double), intent(out) :: centroid(5, n, n), metrics(3, n, n)
    real(c_double), intent(out) :: metrics_int(3, n, n), stretch(7, n, n)
    real(c_double), intent(out) :: vtxw(4, 5, n, n)
    type(panel_grid_t) :: g
    call grid_init(g, n)
    xe = g%xe; xc = g%xc; area = g%area
    centroid = g%centroid; metrics = g%metrics
    metrics_int = g%metrics_int; stretch = g%stretch; vtxw = g%vtxw
  end subroutine c_grid_geometry

  !---------------------- per-slice reconstruction ------------------------
  ! (test/diagnostic entry: returns the reconstruction coefficients and
  !  both halo arrays for one field slice)

  subroutine c_slice_recon(n, ncol, col2pij, nhi, hi_dst, hi_src, hi_w,  &
                           nli, li_dst, li_src, stretch, centroid,       &
                           metrics, metrics_int, vtxw, xe, limiter_mode, &
                           f_slice, fext, flim, rec) bind(C, name="cr_slice_recon")
    integer(c_int), intent(in), value :: n, ncol, nhi, nli, limiter_mode
    integer(c_int), intent(in)  :: col2pij(3, ncol)
    integer(c_int), intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(c_double), intent(in)  :: hi_w(3, nhi)
    integer(c_int), intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(c_double), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(c_double), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(c_double), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    real(c_double), intent(in)  :: f_slice(ncol)
    real(c_double), intent(out) :: fext(-1:n+2, -1:n+2, 6)
    real(c_double), intent(out) :: flim(0:n+1, 0:n+1, 6)
    real(c_double), intent(out) :: rec(6, n, n, 6)
    call slice_recon(n, ncol, col2pij, nhi, hi_dst, hi_src, hi_w, nli,   &
                     li_dst, li_src, stretch, centroid, metrics,         &
                     metrics_int, vtxw, xe, limiter_mode, f_slice,       &
                     fext, flim, rec)
  end subroutine c_slice_recon

  !---------------------------- apply ------------------------------------

  subroutine c_apply_slices(n, ncol, nslice, nb, col2pij,                &
                            nhi, hi_dst, hi_src, hi_w,                   &
                            nli, li_dst, li_src,                         &
                            stretch, centroid, metrics, metrics_int,     &
                            vtxw, xe, npair, pr_src, pr_dst, smat,       &
                            area_b, limiter_mode, f_in, f_out)           &
                            bind(C, name="cr_apply_slices")
    integer(c_int), intent(in), value :: n, ncol, nslice, nb, nhi, nli, npair
    integer(c_int), intent(in), value :: limiter_mode
    integer(c_int), intent(in)  :: col2pij(3, ncol)
    integer(c_int), intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(c_double), intent(in)  :: hi_w(3, nhi)
    integer(c_int), intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(c_double), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(c_double), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(c_double), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    integer(c_int), intent(in)  :: pr_src(3, npair), pr_dst(npair)
    real(c_double), intent(in)  :: smat(6, npair), area_b(nb)
    real(c_double), intent(in)  :: f_in(ncol, nslice)
    real(c_double), intent(out) :: f_out(nb, nslice)
    call apply_slices(n, ncol, nslice, nb, col2pij, nhi, hi_dst, hi_src, &
                      hi_w, nli, li_dst, li_src, stretch, centroid,      &
                      metrics, metrics_int, vtxw, xe, npair, pr_src,     &
                      pr_dst, smat, area_b, limiter_mode, f_in, f_out)
  end subroutine c_apply_slices

  subroutine c_apply_slices_masked(n, ncol, nslice, nb, col2pij,         &
                            nhi, hi_dst, hi_src, hi_w,                   &
                            nli, li_dst, li_src,                         &
                            stretch, centroid, metrics, metrics_int,     &
                            vtxw, xe, npair, pr_src, pr_dst, smat,       &
                            area_b, limiter_mode, frac_tol, fillval,     &
                            lf, mask, f_in, f_out, frac_out)             &
                            bind(C, name="cr_apply_slices_masked")
    integer(c_int), intent(in), value :: n, ncol, nslice, nb, nhi, nli, npair
    integer(c_int), intent(in), value :: limiter_mode
    real(c_double), intent(in), value :: frac_tol, fillval
    integer(c_int), intent(in)  :: col2pij(3, ncol)
    integer(c_int), intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(c_double), intent(in)  :: hi_w(3, nhi)
    integer(c_int), intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(c_double), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(c_double), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(c_double), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    integer(c_int), intent(in)  :: pr_src(3, npair), pr_dst(npair)
    real(c_double), intent(in)  :: smat(6, npair), area_b(nb)
    real(c_double), intent(in)  :: lf(ncol)
    integer(c_int), intent(in)  :: mask(ncol, nslice)
    real(c_double), intent(in)  :: f_in(ncol, nslice)
    real(c_double), intent(out) :: f_out(nb, nslice)
    real(c_double), intent(out) :: frac_out(nb, nslice)
    call apply_slices_masked(n, ncol, nslice, nb, col2pij, nhi, hi_dst,  &
                      hi_src, hi_w, nli, li_dst, li_src, stretch,        &
                      centroid, metrics, metrics_int, vtxw, xe, npair,   &
                      pr_src, pr_dst, smat, area_b, limiter_mode,        &
                      frac_tol, fillval, lf, mask, f_in, f_out, frac_out)
  end subroutine c_apply_slices_masked

end module cr_capi
