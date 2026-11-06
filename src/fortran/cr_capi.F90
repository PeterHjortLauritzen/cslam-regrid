module cr_capi
  !-----------------------------------------------------------------------
  ! iso_c_binding entry points for the Python layer (ctypes).
  ! Every routine here is a thin wrapper; all logic lives in the cr_* modules.
  !-----------------------------------------------------------------------
  use iso_c_binding, only: c_double, c_int, c_int64_t
  use cr_kinds
  use cr_gnomonic
  use cr_moments
  implicit none

contains

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

end module cr_capi
