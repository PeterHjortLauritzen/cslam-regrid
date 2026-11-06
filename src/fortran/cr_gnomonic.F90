module cr_gnomonic
  !-----------------------------------------------------------------------
  ! Cubed-sphere panel conventions and coordinate transforms.
  !
  ! Panels 1..4 are equatorial (centered at lon = 0, 90, 180, 270 deg),
  ! panel 5 is north, panel 6 is south (GECoRe/HOMME conventions, shifted
  ! to 1-based indexing for Fortran).  Gnomonic coordinates on each panel
  ! are x = tan(alpha), y = tan(beta) in [-1,1]; the spherical area element
  ! is (1+x^2+y^2)^(-3/2) dx dy.
  !
  ! Panel -> unit-sphere Cartesian (X,Y,Z), with s = 1/sqrt(1+x^2+y^2):
  !   1: ( s  ,  s*x,  s*y)    2: (-s*x,  s  ,  s*y)
  !   3: (-s  , -s*x,  s*y)    4: ( s*x, -s  ,  s*y)
  !   5: (-s*y,  s*x,  s  )    6: ( s*y,  s*x, -s  )
  !
  ! Inverses (from the table above):
  !   1: x =  Y/X,  y =  Z/X        2: x = -X/Y,  y =  Z/Y
  !   3: x =  Y/X,  y = -Z/X        4: x = -X/Y,  y = -Z/Y
  !   5: x =  Y/Z,  y = -X/Z        6: x = -Y/Z,  y = -X/Z
  !-----------------------------------------------------------------------
  use cr_kinds
  implicit none
  private

  public :: cart_from_panel, panel_from_cart, lonlat_from_panel, panel_from_lonlat
  public :: panel_point_from_lonlat

contains

  pure subroutine cart_from_panel(x, y, ip, cart)
    real(r8), intent(in)  :: x, y
    integer,  intent(in)  :: ip
    real(r8), intent(out) :: cart(3)
    real(r8) :: s
    s = 1.0_r8/sqrt(1.0_r8 + x*x + y*y)
    select case (ip)
    case (1); cart = [  s    ,  s*x,  s*y ]
    case (2); cart = [ -s*x  ,  s  ,  s*y ]
    case (3); cart = [ -s    , -s*x,  s*y ]
    case (4); cart = [  s*x  , -s  ,  s*y ]
    case (5); cart = [ -s*y  ,  s*x,  s   ]
    case default; cart = [  s*y,  s*x, -s  ]  ! panel 6
    end select
  end subroutine cart_from_panel

  pure subroutine panel_from_cart(cart, x, y, ip)
    !
    ! Panel = largest |Cartesian component|; ties resolve polar first,
    ! then the Y panels, then the X panels.
    !
    real(r8), intent(in)  :: cart(3)
    real(r8), intent(out) :: x, y
    integer,  intent(out) :: ip
    real(r8) :: ax, ay, az

    ax = abs(cart(1)); ay = abs(cart(2)); az = abs(cart(3))
    if (az >= ax .and. az >= ay) then
      if (cart(3) > 0.0_r8) then
        ip = 5; x =  cart(2)/cart(3); y = -cart(1)/cart(3)
      else
        ip = 6; x = -cart(2)/cart(3); y = -cart(1)/cart(3)
      end if
    else if (ay >= ax) then
      if (cart(2) > 0.0_r8) then
        ip = 2; x = -cart(1)/cart(2); y =  cart(3)/cart(2)
      else
        ip = 4; x = -cart(1)/cart(2); y = -cart(3)/cart(2)
      end if
    else
      if (cart(1) > 0.0_r8) then
        ip = 1; x =  cart(2)/cart(1); y =  cart(3)/cart(1)
      else
        ip = 3; x =  cart(2)/cart(1); y = -cart(3)/cart(1)
      end if
    end if
  end subroutine panel_from_cart

  pure subroutine lonlat_from_panel(x, y, ip, lon, lat)
    !
    ! lon in [0, 2*pi), lat in [-pi/2, pi/2].
    !
    real(r8), intent(in)  :: x, y
    integer,  intent(in)  :: ip
    real(r8), intent(out) :: lon, lat
    select case (ip)
    case (1:4)
      lon = atan(x) + real(ip-1, r8)*halfpi
      lat = atan(y/sqrt(1.0_r8 + x*x))
    case (5)
      lon = atan2(x, -y)
      lat = halfpi - atan(sqrt(x*x + y*y))
    case default  ! 6
      lon = atan2(x, y)
      lat = -halfpi + atan(sqrt(x*x + y*y))
    end select
    if (lon < 0.0_r8) lon = lon + twopi
    if (lon >= twopi) lon = lon - twopi
  end subroutine lonlat_from_panel

  pure subroutine panel_from_lonlat(lon, lat, x, y, ip)
    real(r8), intent(in)  :: lon, lat
    real(r8), intent(out) :: x, y
    integer,  intent(out) :: ip
    real(r8) :: cart(3)
    cart(1) = cos(lat)*cos(lon)
    cart(2) = cos(lat)*sin(lon)
    cart(3) = sin(lat)
    call panel_from_cart(cart, x, y, ip)
  end subroutine panel_from_lonlat

  pure subroutine panel_point_from_lonlat(lon, lat, ip, x, y, valid)
    !
    ! (x,y) of a lon/lat point expressed in a *given* panel's gnomonic
    ! plane (may lie outside [-1,1]; invalid on/behind the panel horizon).
    !
    real(r8), intent(in)  :: lon, lat
    integer,  intent(in)  :: ip
    real(r8), intent(out) :: x, y
    logical,  intent(out) :: valid
    real(r8) :: cart(3), d
    cart(1) = cos(lat)*cos(lon)
    cart(2) = cos(lat)*sin(lon)
    cart(3) = sin(lat)
    select case (ip)
    case (1); d =  cart(1); x =  cart(2)/d; y =  cart(3)/d
    case (2); d =  cart(2); x = -cart(1)/d; y =  cart(3)/d
    case (3); d = -cart(1); x = -cart(2)/d; y =  cart(3)/d
    case (4); d = -cart(2); x =  cart(1)/d; y =  cart(3)/d
    case (5); d =  cart(3); x =  cart(2)/d; y = -cart(1)/d
    case default; d = -cart(3); x = cart(2)/d; y =  cart(1)/d  ! panel 6
    end select
    valid = d > 1.0e-12_r8
    if (.not. valid) then
      x = huge(1.0_r8); y = huge(1.0_r8)
    end if
  end subroutine panel_point_from_lonlat

end module cr_gnomonic
