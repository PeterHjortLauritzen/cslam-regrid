module cr_kinds
  !-----------------------------------------------------------------------
  ! Precision and mathematical constants for cslam-regrid.
  !-----------------------------------------------------------------------
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  public

  integer, parameter :: r8 = c_double
  integer, parameter :: i8 = c_int64_t

  real(r8), parameter :: pi      = 3.14159265358979323846264338327950288_r8
  real(r8), parameter :: halfpi  = 0.5_r8*pi
  real(r8), parameter :: twopi   = 2.0_r8*pi
  real(r8), parameter :: fourth_pi = 0.25_r8*pi
  real(r8), parameter :: deg2rad = pi/180.0_r8
  real(r8), parameter :: rad2deg = 180.0_r8/pi

end module cr_kinds
