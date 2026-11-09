module cr_recon
  !-----------------------------------------------------------------------
  ! CSLAM quadratic reconstruction (irecons=6) and shape-preserving
  ! limiter, ported from CAM-SE fvm_reconstruction_mod.F90
  ! (get_gradients, slope_limiter, and the centroid->absolute-monomial
  ! conversion in `reconstruction`), restructured for whole-panel arrays.
  !
  ! Interior cells (1:n,1:n) only are reconstructed; halo cells feed the
  ! stencils:
  !   fext  (-1:n+2, -1:n+2)  interpolated extension of the panel
  !                           (extend_panel_interpolate equivalent) - used
  !                           by the finite-difference derivatives,
  !   flim  ( 0:n+1,  0:n+1)  raw index-aligned neighbor cell averages -
  !                           used by the limiter 3x3 patch bounds
  !                           (CAM's limiter operates on exchanged fcube).
  !
  ! Limiter modes: 0 = none, 1 = shape-preserving (CAM), 2 = positive-
  ! definite (same extrema machinery, bounds [0, +inf); phi clamped to
  ! [0,1], and phi = 0 if the cell mean itself is negative).
  !
  ! Masked variant (recon_panel_masked): validity masks for both halo
  ! arrays.  A cell whose finite-difference stencil touches any invalid
  ! cell falls back to first order (the cell mean); limiter patch bounds
  ! use only valid neighbors; invalid cells get zero coefficients.
  !-----------------------------------------------------------------------
  use cr_kinds
  implicit none
  private

  public :: recon_panel, recon_panel_masked

  real(r8), parameter :: threshold = 1.0e-40_r8

contains

  subroutine recon_panel(n, fext, flim, stretch, centroid, metrics, &
                         metrics_int, vtxw, xe, limiter_mode, recons)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: fext(-1:n+2, -1:n+2)
    real(r8), intent(in)  :: flim(0:n+1, 0:n+1)
    real(r8), intent(in)  :: stretch(7, n, n)
    real(r8), intent(in)  :: centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n)
    real(r8), intent(in)  :: metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n)
    real(r8), intent(in)  :: xe(0:n)
    integer,  intent(in)  :: limiter_mode
    real(r8), intent(out) :: recons(6, n, n)

    integer  :: i, j
    real(r8) :: g2, g3, g4, g5, g6, f0

    ! 4th-order centered differences * chain-rule stretching
    ! (get_gradients, irecons=6 path)
    do j = 1, n
      do i = 1, n
        g2 = -fext(i+2,j) + 8.0_r8*fext(i+1,j) - 8.0_r8*fext(i-1,j) + fext(i-2,j)
        g3 = -fext(i,j+2) + 8.0_r8*fext(i,j+1) - 8.0_r8*fext(i,j-1) + fext(i,j-2)
        g4 = -fext(i+2,j) + 16.0_r8*fext(i+1,j) - 30.0_r8*fext(i,j) &
             + 16.0_r8*fext(i-1,j) - fext(i-2,j)
        g5 = -fext(i,j+2) + 16.0_r8*fext(i,j+1) - 30.0_r8*fext(i,j) &
             + 16.0_r8*fext(i,j-1) - fext(i,j-2)
        g6 =  fext(i+1,j+1) - fext(i+1,j-1) - fext(i-1,j+1) + fext(i-1,j-1)

        g2 = stretch(1,i,j)*g2
        g3 = stretch(2,i,j)*g3
        recons(1,i,j) = fext(i,j)
        recons(2,i,j) = g2
        recons(3,i,j) = g3
        recons(4,i,j) = stretch(3,i,j)*g4 + stretch(6,i,j)*g2
        recons(5,i,j) = stretch(4,i,j)*g5 + stretch(7,i,j)*g3
        recons(6,i,j) = stretch(5,i,j)*g6
      end do
    end do

    if (limiter_mode /= 0) then
      call limit_panel(n, flim, centroid, metrics, vtxw, xe, limiter_mode, recons)
    end if

    ! centroid form -> absolute monomials c1..c6 (reconstruction, case(6))
    do j = 1, n
      do i = 1, n
        f0 = recons(1,i,j)
        recons(1,i,j) = f0 &
             - recons(2,i,j)*centroid(1,i,j) &
             - recons(3,i,j)*centroid(2,i,j) &
             + recons(4,i,j)*metrics_int(1,i,j) &
             + recons(5,i,j)*metrics_int(2,i,j) &
             + recons(6,i,j)*metrics_int(3,i,j)
        recons(2,i,j) = recons(2,i,j) &
             - recons(4,i,j)*2.0_r8*centroid(1,i,j) &
             - recons(6,i,j)*centroid(2,i,j)
        recons(3,i,j) = recons(3,i,j) &
             - recons(5,i,j)*2.0_r8*centroid(2,i,j) &
             - recons(6,i,j)*centroid(1,i,j)
      end do
    end do
  end subroutine recon_panel

  subroutine limit_panel(n, flim, centroid, metrics, vtxw, xe, limiter_mode, recons)
    ! slope_limiter, irecons=6 path.  recons holds (f0, r2..r6) in
    ! centroid form; r2..r6 are scaled in place.
    integer,  intent(in)    :: n
    real(r8), intent(in)    :: flim(0:n+1, 0:n+1)
    real(r8), intent(in)    :: centroid(5, n, n)
    real(r8), intent(in)    :: metrics(3, n, n)
    real(r8), intent(in)    :: vtxw(4, 5, n, n)
    real(r8), intent(in)    :: xe(0:n)
    integer,  intent(in)    :: limiter_mode
    real(r8), intent(inout) :: recons(6, n, n)

    integer  :: i, j
    real(r8) :: minval_patch, maxval_patch, f0

    do j = 1, n
      do i = 1, n
        f0 = recons(1,i,j)

        if (limiter_mode == 1) then
          minval_patch = min(min(flim(i-1,j-1), flim(i,j-1), flim(i+1,j-1)),  &
                             min(flim(i-1,j  ), flim(i,j  ), flim(i+1,j  )),  &
                             min(flim(i-1,j+1), flim(i,j+1), flim(i+1,j+1)))
          maxval_patch = max(max(flim(i-1,j-1), flim(i,j-1), flim(i+1,j-1)),  &
                             max(flim(i-1,j  ), flim(i,j  ), flim(i+1,j  )),  &
                             max(flim(i-1,j+1), flim(i,j+1), flim(i+1,j+1)))
        else
          ! positive definite: lower bound 0, no upper bound
          if (f0 < 0.0_r8) then
            recons(2:6,i,j) = 0.0_r8
            cycle
          end if
          minval_patch = 0.0_r8
          maxval_patch = huge(1.0_r8)
        end if

        recons(2:6,i,j) = recons(2:6,i,j) * cell_phi(                       &
             f0, recons(2,i,j), recons(3,i,j), recons(4,i,j),               &
             recons(5,i,j), recons(6,i,j),                                  &
             centroid(1,i,j), centroid(2,i,j),                              &
             metrics(1,i,j), metrics(2,i,j), metrics(3,i,j),                &
             xe(i-1), xe(i), xe(j-1), xe(j), vtxw(:,:,i,j),                 &
             minval_patch, maxval_patch)
      end do
    end do
  end subroutine limit_panel

  pure function cell_phi(f0, r2, r3, r4, r5, r6, scx, scy, m1, m2, m3, &
                         xlo, xhi, ylo, yhi, vtw, minval_patch,        &
                         maxval_patch) result(min_phi)
    ! The slope_limiter extrema machinery for one cell: candidate extrema
    ! of the centroid-form quadratic over the cell (4 vertices, interior
    ! stationary point, edge extrema), then the scalar phi that pulls the
    ! deviation from the cell mean inside [minval_patch, maxval_patch].
    real(r8), intent(in) :: f0, r2, r3, r4, r5, r6, scx, scy, m1, m2, m3
    real(r8), intent(in) :: xlo, xhi, ylo, yhi, vtw(4, 5)
    real(r8), intent(in) :: minval_patch, maxval_patch
    real(r8) :: min_phi

    integer  :: vertex, nn
    real(r8) :: min_val, max_val, phi
    real(r8) :: disc, ex1, ex2, dx, dy, v1, val, invtmp

    min_val = f0; max_val = f0
    do vertex = 1, 4
      val = f0 + r2*vtw(vertex,1) + r3*vtw(vertex,2) &
               + r4*vtw(vertex,3) + r5*vtw(vertex,4) &
               + r6*vtw(vertex,5)
      min_val = min(min_val, val)
      max_val = max(max_val, val)
    end do

    ! interior stationary point of the quadratic
    disc = 4.0_r8*r4*r5 - r6*r6
    if (abs(disc) > threshold) then
      ex1 = (r6*r3 - 2.0_r8*r5*r2)
      ex2 = (r6*r2 - 2.0_r8*r4*r3)
      disc = 1.0_r8/disc
      ex1 = ex1*disc + scx
      ex2 = ex2*disc + scy
      if (ex1 > xlo - threshold .and. ex1 < xhi + threshold .and. &
          ex2 > ylo - threshold .and. ex2 < yhi + threshold) then
        dx = ex1 - scx; dy = ex2 - scy
        v1 = f0 + r2*dx + r3*dy + r4*(m1 + dx*dx) + r5*(m2 + dy*dy) &
             + r6*(m3 + dx*dy)
        max_val = max(max_val, v1)
        min_val = min(min_val, v1)
      end if
    end if

    ! extrema on y = const edges (du/dx = 0)
    if (abs(r4) > threshold) then
      invtmp = 1.0_r8/(2.0_r8*r4)
      do nn = 1, 2
        dy = merge(ylo, yhi, nn == 1) - scy
        ex1 = scx + invtmp*(-r2 - r6*dy)
        if (ex1 > xlo - threshold .and. ex1 < xhi + threshold) then
          dx = ex1 - scx
          v1 = f0 + r2*dx + r3*dy + r4*(m1 + dx*dx) + r5*(m2 + dy*dy) &
               + r6*(m3 + dx*dy)
          max_val = max(max_val, v1)
          min_val = min(min_val, v1)
        end if
      end do
    end if

    ! extrema on x = const edges (du/dy = 0)
    if (abs(r5) > threshold) then
      invtmp = 1.0_r8/(2.0_r8*r5)
      do nn = 1, 2
        dx = merge(xlo, xhi, nn == 1) - scx
        ex1 = scy + invtmp*(-r3 - r6*dx)
        if (ex1 > ylo - threshold .and. ex1 < yhi + threshold) then
          dy = ex1 - scy
          v1 = f0 + r2*dx + r3*dy + r4*(m1 + dx*dx) + r5*(m2 + dy*dy) &
               + r6*(m3 + dx*dy)
          max_val = max(max_val, v1)
          min_val = min(min_val, v1)
        end if
      end do
    end if

    min_phi = 1.0_r8
    if (max_val > maxval_patch .and. abs(max_val - f0) > threshold) then
      phi = (maxval_patch - f0)/(max_val - f0)
      if (phi < min_phi) min_phi = phi
    end if
    if (min_val < minval_patch .and. abs(min_val - f0) > threshold) then
      phi = (minval_patch - f0)/(min_val - f0)
      if (phi < min_phi) min_phi = phi
    end if
    min_phi = max(0.0_r8, min_phi)
  end function cell_phi

  !------------------------------ masked path ------------------------------

  subroutine recon_panel_masked(n, fext, flim, vext, vlim, stretch,       &
                                centroid, metrics, metrics_int, vtxw, xe, &
                                limiter_mode, recons)
    ! recon_panel with validity masks.  vext/vlim flag valid entries of
    ! fext/flim (1 valid, 0 invalid).  Invalid cells get zero coefficients;
    ! a valid cell whose FD stencil (the 12 fext neighbors used by g2..g6)
    ! touches an invalid entry keeps only its mean (first order).
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: fext(-1:n+2, -1:n+2)
    real(r8), intent(in)  :: flim(0:n+1, 0:n+1)
    integer,  intent(in)  :: vext(-1:n+2, -1:n+2)
    integer,  intent(in)  :: vlim(0:n+1, 0:n+1)
    real(r8), intent(in)  :: stretch(7, n, n)
    real(r8), intent(in)  :: centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n)
    real(r8), intent(in)  :: metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n)
    real(r8), intent(in)  :: xe(0:n)
    integer,  intent(in)  :: limiter_mode
    real(r8), intent(out) :: recons(6, n, n)

    integer  :: i, j
    real(r8) :: g2, g3, g4, g5, g6, f0

    do j = 1, n
      do i = 1, n
        if (vext(i,j) == 0) then
          recons(:,i,j) = 0.0_r8
          cycle
        end if
        if (min(vext(i-2,j), vext(i-1,j), vext(i+1,j), vext(i+2,j),       &
                vext(i,j-2), vext(i,j-1), vext(i,j+1), vext(i,j+2),       &
                vext(i-1,j-1), vext(i+1,j-1), vext(i-1,j+1),              &
                vext(i+1,j+1)) == 0) then
          ! stencil touches an invalid cell: first order (cell mean)
          recons(1,i,j) = fext(i,j)
          recons(2:6,i,j) = 0.0_r8
          cycle
        end if
        g2 = -fext(i+2,j) + 8.0_r8*fext(i+1,j) - 8.0_r8*fext(i-1,j) + fext(i-2,j)
        g3 = -fext(i,j+2) + 8.0_r8*fext(i,j+1) - 8.0_r8*fext(i,j-1) + fext(i,j-2)
        g4 = -fext(i+2,j) + 16.0_r8*fext(i+1,j) - 30.0_r8*fext(i,j) &
             + 16.0_r8*fext(i-1,j) - fext(i-2,j)
        g5 = -fext(i,j+2) + 16.0_r8*fext(i,j+1) - 30.0_r8*fext(i,j) &
             + 16.0_r8*fext(i,j-1) - fext(i,j-2)
        g6 =  fext(i+1,j+1) - fext(i+1,j-1) - fext(i-1,j+1) + fext(i-1,j-1)

        g2 = stretch(1,i,j)*g2
        g3 = stretch(2,i,j)*g3
        recons(1,i,j) = fext(i,j)
        recons(2,i,j) = g2
        recons(3,i,j) = g3
        recons(4,i,j) = stretch(3,i,j)*g4 + stretch(6,i,j)*g2
        recons(5,i,j) = stretch(4,i,j)*g5 + stretch(7,i,j)*g3
        recons(6,i,j) = stretch(5,i,j)*g6
      end do
    end do

    if (limiter_mode /= 0) then
      call limit_panel_masked(n, flim, vext, vlim, centroid, metrics, &
                              vtxw, xe, limiter_mode, recons)
    end if

    do j = 1, n
      do i = 1, n
        if (vext(i,j) == 0) cycle
        f0 = recons(1,i,j)
        recons(1,i,j) = f0 &
             - recons(2,i,j)*centroid(1,i,j) &
             - recons(3,i,j)*centroid(2,i,j) &
             + recons(4,i,j)*metrics_int(1,i,j) &
             + recons(5,i,j)*metrics_int(2,i,j) &
             + recons(6,i,j)*metrics_int(3,i,j)
        recons(2,i,j) = recons(2,i,j) &
             - recons(4,i,j)*2.0_r8*centroid(1,i,j) &
             - recons(6,i,j)*centroid(2,i,j)
        recons(3,i,j) = recons(3,i,j) &
             - recons(5,i,j)*2.0_r8*centroid(2,i,j) &
             - recons(6,i,j)*centroid(1,i,j)
      end do
    end do
  end subroutine recon_panel_masked

  subroutine limit_panel_masked(n, flim, vext, vlim, centroid, metrics, &
                                vtxw, xe, limiter_mode, recons)
    ! limit_panel over valid cells only; shape-preserving patch bounds use
    ! only the valid flim neighbors (the cell itself is always valid here).
    integer,  intent(in)    :: n
    real(r8), intent(in)    :: flim(0:n+1, 0:n+1)
    integer,  intent(in)    :: vext(-1:n+2, -1:n+2)
    integer,  intent(in)    :: vlim(0:n+1, 0:n+1)
    real(r8), intent(in)    :: centroid(5, n, n)
    real(r8), intent(in)    :: metrics(3, n, n)
    real(r8), intent(in)    :: vtxw(4, 5, n, n)
    real(r8), intent(in)    :: xe(0:n)
    integer,  intent(in)    :: limiter_mode
    real(r8), intent(inout) :: recons(6, n, n)

    integer  :: i, j, ii, jj
    real(r8) :: minval_patch, maxval_patch, f0

    do j = 1, n
      do i = 1, n
        if (vext(i,j) == 0) cycle
        f0 = recons(1,i,j)

        if (limiter_mode == 1) then
          minval_patch = flim(i,j)
          maxval_patch = flim(i,j)
          do jj = j-1, j+1
            do ii = i-1, i+1
              if (vlim(ii,jj) == 1) then
                minval_patch = min(minval_patch, flim(ii,jj))
                maxval_patch = max(maxval_patch, flim(ii,jj))
              end if
            end do
          end do
        else
          if (f0 < 0.0_r8) then
            recons(2:6,i,j) = 0.0_r8
            cycle
          end if
          minval_patch = 0.0_r8
          maxval_patch = huge(1.0_r8)
        end if

        recons(2:6,i,j) = recons(2:6,i,j) * cell_phi(                       &
             f0, recons(2,i,j), recons(3,i,j), recons(4,i,j),               &
             recons(5,i,j), recons(6,i,j),                                  &
             centroid(1,i,j), centroid(2,i,j),                              &
             metrics(1,i,j), metrics(2,i,j), metrics(3,i,j),                &
             xe(i-1), xe(i), xe(j-1), xe(j), vtxw(:,:,i,j),                 &
             minval_patch, maxval_patch)
      end do
    end do
  end subroutine limit_panel_masked

end module cr_recon
