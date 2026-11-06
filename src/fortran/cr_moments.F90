module cr_moments
  !-----------------------------------------------------------------------
  ! Exact line integrals of the CSLAM moment potentials.
  !
  ! Overlap moments are computed via Green's theorem with dx-potentials,
  !     w_k = -oint F_k(x,y) dx,   dF_k/dy = chi_k * (1+x^2+y^2)^(-3/2),
  ! chi = (1, x, y, x^2, y^2, x*y)  [indices 1..6 <-> pq = 00,10,01,20,02,11].
  !
  ! Each seg_w6_* routine returns W(1:6) = integral_{xa}^{xb} F_k(x, y(x)) dx
  ! in closed form along one boundary-curve family (see docs/exact_integrals.md
  ! and tools/derive_integrals.py for derivations and symbolic verification):
  !
  !   seg_w6_horizontal   y = c                       (source y-edges)
  !   seg_w6_parallel_eq  y = t*sqrt(1+x^2)           (parallel, equatorial panel)
  !   seg_w6_parallel_po  y = s*sqrt(r^2-x^2)         (parallel, polar panel)
  !   seg_w6_meridian_po  y = m*x                     (meridian, polar panel)
  !
  ! Vertical lines carry no dx and contribute exactly zero.
  ! I_pq are the dycore antiderivatives (fvm_analytic_mod.F90, CAM-SE).
  !-----------------------------------------------------------------------
  use cr_kinds
  implicit none
  private

  public :: seg_w6_horizontal, seg_w6_parallel_eq, seg_w6_parallel_po
  public :: seg_w6_meridian_po, cell_moments

contains

  !--------------------------- horizontal: y = c -------------------------

  pure subroutine seg_w6_horizontal(c, xa, xb, w)
    real(r8), intent(in)  :: c, xa, xb
    real(r8), intent(out) :: w(6)
    w(1) = i_00(xb, c) - i_00(xa, c)
    w(2) = i_10(xb, c) - i_10(xa, c)
    w(3) = i_01(xb, c) - i_01(xa, c)
    w(4) = i_20(xb, c) - i_20(xa, c)
    w(5) = i_02(xb, c) - i_02(xa, c)
    w(6) = i_11(xb, c) - i_11(xa, c)
  end subroutine seg_w6_horizontal

  pure function i_00(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = atan(x*y/sqrt(1.0_r8 + x*x + y*y))
  end function i_00

  pure function i_10(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = -asinh(y/sqrt(1.0_r8 + x*x))
  end function i_10

  pure function i_01(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = -asinh(x/sqrt(1.0_r8 + y*y))
  end function i_01

  pure function i_20(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = y*asinh(x/sqrt(1.0_r8 + y*y)) + acos(acos_arg(x, y))
  end function i_20

  pure function i_02(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = x*asinh(y/sqrt(1.0_r8 + x*x)) + acos(acos_arg(x, y))
  end function i_02

  pure function i_11(x, y) result(v)
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = -sqrt(1.0_r8 + x*x + y*y)
  end function i_11

  pure function acos_arg(x, y) result(v)
    ! x*y/sqrt((1+x^2)(1+y^2)) is in (-1,1) exactly; clamp roundoff.
    real(r8), intent(in) :: x, y
    real(r8) :: v
    v = x*y/sqrt((1.0_r8 + x*x)*(1.0_r8 + y*y))
    v = min(1.0_r8, max(-1.0_r8, v))
  end function acos_arg

  !------------------ parallel on an equatorial panel ---------------------
  !  y(x) = t*sqrt(1+x^2),  t = tan(lat);  rho = sqrt(1+x^2)*sqrt(1+t^2)

  pure subroutine seg_w6_parallel_eq(t, xa, xb, w)
    real(r8), intent(in)  :: t, xa, xb
    real(r8), intent(out) :: w(6)
    real(r8) :: q, toq
    q   = sqrt(1.0_r8 + t*t)
    toq = t/q
    w(1) = toq*(atan(xb) - atan(xa))
    w(2) = 0.5_r8*toq*(log(1.0_r8 + xb*xb) - log(1.0_r8 + xa*xa))
    w(3) = -(asinh(xb) - asinh(xa))/q
    w(4) = toq*((xb - atan(xb)) - (xa - atan(xa)))
    w(5) = (asinh(t) - toq)*(xb - xa)
    w(6) = -(sqrt(1.0_r8 + xb*xb) - sqrt(1.0_r8 + xa*xa))/q
  end subroutine seg_w6_parallel_eq

  !-------------------- parallel on a polar panel -------------------------
  !  circle x^2 + y^2 = r^2, branch y = sbr*sqrt(r^2-x^2), sbr = +-1;
  !  rho = P = sqrt(1+r^2) constant on the circle.  Segments never span
  !  the branch points (xa, xb in [-r, r], single-valued branch).

  pure subroutine seg_w6_parallel_po(r, sbr, xa, xb, w)
    real(r8), intent(in)  :: r, sbr, xa, xb
    real(r8), intent(out) :: w(6)
    real(r8) :: p
    p = sqrt(1.0_r8 + r*r)
    w(1) = a_po00(xb) - a_po00(xa)
    w(2) = a_po10(xb) - a_po10(xa)
    w(3) = -(xb - xa)/p
    w(4) = a_po20(xb) - a_po20(xa)
    w(5) = a_po02(xb) - a_po02(xa)
    w(6) = -(xb*xb - xa*xa)/(2.0_r8*p)
  contains
    pure function arc_s(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v
      v = sqrt(max(0.0_r8, r*r - x*x))
    end function arc_s
    pure function arc_theta(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v
      v = asin(min(1.0_r8, max(-1.0_r8, x/r)))
    end function arc_theta
    pure function a_po00(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v
      v = sbr*(atan2(p*x, arc_s(x)) - arc_theta(x)/p)
    end function a_po00
    pure function a_po10(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v, s
      s = arc_s(x)
      v = sbr*(s/p - 0.5_r8*log((p + s)/(p - s)))
    end function a_po10
    pure function a_po20(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v
      v = 0.5_r8*sbr*(x*arc_s(x) + r*r*arc_theta(x))/p - a_po00(x)
    end function a_po20
    pure function a_po02(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v, s, th
      s  = arc_s(x)
      th = arc_theta(x)
      v = -0.5_r8*sbr*(x*s + r*r*th)/p                          &
          + x*asinh(sbr*s/sqrt(1.0_r8 + x*x))                   &
          + sbr*(p*th - atan2(p*x, s))
    end function a_po02
  end subroutine seg_w6_parallel_po

  !-------------------- meridian on a polar panel -------------------------
  !  radial line y = m*x (m /= 0 finite; m = 0 -> horizontal family,
  !  |m| = inf -> vertical, zero).  rho = R = sqrt(1+(1+m^2)*x^2).

  pure subroutine seg_w6_meridian_po(m, xa, xb, w)
    real(r8), intent(in)  :: m, xa, xb
    real(r8), intent(out) :: w(6)
    real(r8) :: k2, k
    k2 = 1.0_r8 + m*m
    k  = sqrt(k2)
    w(1) = atan(rr(xb)/m) - atan(rr(xa)/m)
    w(2) = (m/k)*(asinh(k*xb) - asinh(k*xa)) - (ymt(xb) - ymt(xa))
    w(3) = -(asinh(k*xb) - asinh(k*xa))/k
    w(4) = (m/k2)*(rr(xb) - rr(xa)) - w(1)
    w(5) = -(m/k2)*(rr(xb) - rr(xa)) + (xb*ymt(xb) - xa*ymt(xa)) - w(1)
    w(6) = -(rr(xb) - rr(xa))/k2
  contains
    pure function rr(x) result(v)
      real(r8), intent(in) :: x
      real(r8) :: v
      v = sqrt(1.0_r8 + k2*x*x)
    end function rr
    pure function ymt(x) result(v)
      ! asinh(m*x/sqrt(1+x^2))
      real(r8), intent(in) :: x
      real(r8) :: v
      v = asinh(m*x/sqrt(1.0_r8 + x*x))
    end function ymt
  end subroutine seg_w6_meridian_po

  !------------------------------------------------------------------------
  ! Exact moments of one gnomonic cell [x1,x2] x [y1,y2]:
  !   mom(1) = spherical area;  mom(2:6) = area averages <x>,<y>,<x2>,<y2>,<xy>
  ! (corner-evaluated antiderivatives, as moment_onsphere in CAM-SE).
  !------------------------------------------------------------------------

  pure subroutine cell_moments(x1, x2, y1, y2, mom)
    real(r8), intent(in)  :: x1, x2, y1, y2
    real(r8), intent(out) :: mom(6)
    mom(1) = corner4(i_00(x2,y2), i_00(x1,y2), i_00(x1,y1), i_00(x2,y1))
    mom(2) = corner4(i_10(x2,y2), i_10(x1,y2), i_10(x1,y1), i_10(x2,y1))/mom(1)
    mom(3) = corner4(i_01(x2,y2), i_01(x1,y2), i_01(x1,y1), i_01(x2,y1))/mom(1)
    mom(4) = corner4(i_20(x2,y2), i_20(x1,y2), i_20(x1,y1), i_20(x2,y1))/mom(1)
    mom(5) = corner4(i_02(x2,y2), i_02(x1,y2), i_02(x1,y1), i_02(x2,y1))/mom(1)
    mom(6) = corner4(i_11(x2,y2), i_11(x1,y2), i_11(x1,y1), i_11(x2,y1))/mom(1)
  end subroutine cell_moments

  pure function corner4(vpp, vmp, vmm, vpm) result(v)
    real(r8), intent(in) :: vpp, vmp, vmm, vpm
    real(r8) :: v
    v = vpp - vmp + vmm - vpm
  end function corner4

end module cr_moments
