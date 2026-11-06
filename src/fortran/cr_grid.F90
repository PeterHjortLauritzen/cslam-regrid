module cr_grid
  !-----------------------------------------------------------------------
  ! Geometry of the regular ne*pg physics grid (whole-panel layout) and
  ! the reconstruction precomputations, ported from CAM-SE
  ! (fvm_analytic_mod.F90: moment_onsphere, compute_reconstruct_matrix).
  !
  ! The grid is globally equiangular: n = fv_nphys*ne cells per panel edge,
  ! cell (i,j) = [xe(i-1), xe(i)] x [xe(j-1), xe(j)] in gnomonic coords,
  ! xe(i) = tan(-pi/4 + i*dalpha), dalpha = (pi/2)/n.  All six panels are
  ! congruent, so per-cell geometry is computed once for one panel.
  !
  ! Per cell (interior only; the tool reconstructs no halo cells):
  !   area              spherical area
  !   centroid(5)       <x>,<y>,<x^2>,<y^2>,<xy>  (spherical area averages)
  !   metrics(3)        cx^2-<x^2>, cy^2-<y^2>, cx*cy-<xy>
  !   metrics_int(3)    2cx^2-<x^2>, 2cy^2-<y^2>, 2cx*cy-<xy>
  !   stretch(7)        finite-difference * chain-rule factors (dycore
  !                     centroid_stretch, incl. the 1/(12 dalpha) etc.)
  !   vtxw(4,5)         limiter vertex evaluation weights
  !-----------------------------------------------------------------------
  use cr_kinds
  use cr_moments, only: cell_moments
  implicit none
  private

  public :: panel_grid_t, grid_init

  type panel_grid_t
    integer :: n = 0
    real(r8) :: dalpha = 0.0_r8
    real(r8), allocatable :: alpha_e(:)       ! (0:n) edge angles
    real(r8), allocatable :: xe(:)            ! (0:n) edge gnomonic coords
    real(r8), allocatable :: xc(:)            ! (1:n) center gnomonic coords (tan of center angle)
    real(r8), allocatable :: area(:,:)        ! (n,n)
    real(r8), allocatable :: centroid(:,:,:)  ! (5,n,n)
    real(r8), allocatable :: metrics(:,:,:)   ! (3,n,n)
    real(r8), allocatable :: metrics_int(:,:,:) ! (3,n,n)
    real(r8), allocatable :: stretch(:,:,:)   ! (7,n,n)
    real(r8), allocatable :: vtxw(:,:,:,:)    ! (4,5,n,n)
  end type panel_grid_t

contains

  subroutine grid_init(g, n)
    type(panel_grid_t), intent(out) :: g
    integer, intent(in) :: n
    integer :: i, j, v
    real(r8) :: mom(6), cx, cy, coef
    real(r8) :: vx(4), vy(4)

    g%n = n
    g%dalpha = halfpi/real(n, r8)
    allocate(g%alpha_e(0:n), g%xe(0:n), g%xc(1:n))
    allocate(g%area(n,n), g%centroid(5,n,n), g%metrics(3,n,n), &
             g%metrics_int(3,n,n), g%stretch(7,n,n), g%vtxw(4,5,n,n))

    do i = 0, n
      g%alpha_e(i) = -fourth_pi + real(i, r8)*g%dalpha
      g%xe(i) = tan(g%alpha_e(i))
    end do
    ! enforce exact symmetry of the grid about the panel center
    do i = 0, n/2
      g%xe(n-i) = -g%xe(i)
    end do
    if (mod(n,2) == 0) g%xe(n/2) = 0.0_r8
    do i = 1, n
      g%xc(i) = tan(-fourth_pi + (real(i, r8) - 0.5_r8)*g%dalpha)
    end do

    do j = 1, n
      do i = 1, n
        call cell_moments(g%xe(i-1), g%xe(i), g%xe(j-1), g%xe(j), mom)
        g%area(i,j)       = mom(1)
        g%centroid(:,i,j) = mom(2:6)
        cx = mom(2); cy = mom(3)

        ! recons_metrics / recons_metrics_integral (compute_reconstruct_matrix)
        g%metrics(1,i,j) = cx*cx - mom(4)
        g%metrics(2,i,j) = cy*cy - mom(5)
        g%metrics(3,i,j) = cx*cy - mom(6)
        g%metrics_int(1,i,j) = 2.0_r8*cx*cx - mom(4)
        g%metrics_int(2,i,j) = 2.0_r8*cy*cy - mom(5)
        g%metrics_int(3,i,j) = 2.0_r8*cx*cy - mom(6)

        ! centroid_stretch: finite-difference coefficient * chain rule at
        ! the centroid (x = tan(alpha) => d/dalpha = (1+x^2) d/dx)
        coef = 1.0_r8/(12.0_r8*g%dalpha)
        g%stretch(1,i,j) = coef/(1.0_r8 + cx*cx)
        g%stretch(2,i,j) = coef/(1.0_r8 + cy*cy)
        coef = 1.0_r8/(12.0_r8*g%dalpha**2)
        g%stretch(3,i,j) = coef*0.5_r8/((1.0_r8 + cx*cx)**2)
        g%stretch(4,i,j) = coef*0.5_r8/((1.0_r8 + cy*cy)**2)
        coef = 1.0_r8/(4.0_r8*g%dalpha**2)
        g%stretch(5,i,j) = coef/((1.0_r8 + cx*cx)*(1.0_r8 + cy*cy))
        g%stretch(6,i,j) = -cx/(1.0_r8 + cx*cx)
        g%stretch(7,i,j) = -cy/(1.0_r8 + cy*cy)

        ! limiter vertex weights (vertex order: (i-1,j-1),(i,j-1),(i,j),(i-1,j))
        vx = [g%xe(i-1), g%xe(i), g%xe(i), g%xe(i-1)]
        vy = [g%xe(j-1), g%xe(j-1), g%xe(j), g%xe(j)]
        do v = 1, 4
          g%vtxw(v,1,i,j) = vx(v) - cx
          g%vtxw(v,2,i,j) = vy(v) - cy
          g%vtxw(v,3,i,j) = g%metrics(1,i,j) + (vx(v) - cx)**2
          g%vtxw(v,4,i,j) = g%metrics(2,i,j) + (vy(v) - cy)**2
          g%vtxw(v,5,i,j) = (vx(v) - cx)*(vy(v) - cy) + g%metrics(3,i,j)
        end do
      end do
    end do
  end subroutine grid_init

end module cr_grid
