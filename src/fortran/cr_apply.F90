module cr_apply
  !-----------------------------------------------------------------------
  ! Apply stage: ncol field slices -> destination lat-lon cell averages.
  !
  ! Per slice:
  !   1. scatter ncol -> panel arrays (col2pij map from the weight file)
  !   2. build fext  (interpolated halo; gather slots: 3-point Lagrange)
  !   3. build flim  (raw ring-1 halo; ordered single-source copies so the
  !      cube-corner fill rule can reference already-filled halo cells)
  !   4. reconstruct + limit (cr_recon)
  !   5. accumulate destination integrals: sum_k S(k,pair)*c_k(src cell),
  !      divide by destination areas.
  !
  ! All index arrays are 1-based Fortran conventions, prepared by the
  ! Python layer and stored in the weight file.  OpenMP over slices.
  !-----------------------------------------------------------------------
  ! Masked path (apply_slices_masked): per-slice validity masks (dynamic:
  ! CLM fields differ in their valid cell sets, even in time) and a
  ! per-cell area fraction lf (landfrac).  Source moments are weighted by
  ! lf (land assumed uniformly distributed within a cell); destination
  ! rows are normalized by the covered (fracarea) overlap, so output values
  ! are means over the represented portion of each destination cell, and
  ! the lf-weighted (land) integral is conserved exactly.  Cells with
  ! coverage <= frac_tol*area are set to fillval.
  !-----------------------------------------------------------------------
  use cr_kinds
  use cr_recon, only: recon_panel, recon_panel_masked
  implicit none
  private

  public :: apply_slices, slice_recon, apply_slices_masked

contains

  subroutine slice_recon(n, ncol, col2pij, nhi, hi_dst, hi_src, hi_w,    &
                         nli, li_dst, li_src, stretch, centroid,         &
                         metrics, metrics_int, vtxw, xe, limiter_mode,   &
                         f_slice, fext, flim, rec)
    ! Steps 1-4 for one slice: scatter, halos, reconstruction.
    integer,  intent(in)  :: n, ncol, nhi, nli, limiter_mode
    integer,  intent(in)  :: col2pij(3, ncol)
    integer,  intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(r8), intent(in)  :: hi_w(3, nhi)
    integer,  intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(r8), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    real(r8), intent(in)  :: f_slice(ncol)
    real(r8), intent(out) :: fext(-1:n+2, -1:n+2, 6)
    real(r8), intent(out) :: flim(0:n+1, 0:n+1, 6)
    real(r8), intent(out) :: rec(6, n, n, 6)

    integer  :: ic, s, p
    real(r8) :: acc

    do ic = 1, ncol
      fext(col2pij(2,ic), col2pij(3,ic), col2pij(1,ic)) = f_slice(ic)
    end do
    do p = 1, 6
      flim(1:n, 1:n, p) = fext(1:n, 1:n, p)
    end do

    do s = 1, nhi
      acc = hi_w(1,s)*fext(hi_src(2,1,s), hi_src(3,1,s), hi_src(1,1,s)) &
          + hi_w(2,s)*fext(hi_src(2,2,s), hi_src(3,2,s), hi_src(1,2,s)) &
          + hi_w(3,s)*fext(hi_src(2,3,s), hi_src(3,3,s), hi_src(1,3,s))
        fext(hi_dst(2,s), hi_dst(3,s), hi_dst(1,s)) = acc
    end do

    do s = 1, nli
      flim(li_dst(2,s), li_dst(3,s), li_dst(1,s)) = &
           flim(li_src(2,s), li_src(3,s), li_src(1,s))
    end do

    do p = 1, 6
      call recon_panel(n, fext(:,:,p), flim(:,:,p), stretch, centroid, &
                       metrics, metrics_int, vtxw, xe, limiter_mode,   &
                       rec(:,:,:,p))
    end do
  end subroutine slice_recon

  subroutine apply_slices(n, ncol, nslice, nb, col2pij,                  &
                          nhi, hi_dst, hi_src, hi_w,                     &
                          nli, li_dst, li_src,                           &
                          stretch, centroid, metrics, metrics_int, vtxw, &
                          xe, npair, pr_src, pr_dst, smat, area_b,       &
                          limiter_mode, f_in, f_out)
    integer,  intent(in)  :: n, ncol, nslice, nb, nhi, nli, npair
    integer,  intent(in)  :: col2pij(3, ncol)     ! (p,i,j) per column
    integer,  intent(in)  :: hi_dst(3, nhi)       ! (p,i,j) halo slot in fext
    integer,  intent(in)  :: hi_src(3, 3, nhi)    ! 3 x (p,i,j) interior sources
    real(r8), intent(in)  :: hi_w(3, nhi)
    integer,  intent(in)  :: li_dst(3, nli)       ! (p,i,j) halo slot in flim
    integer,  intent(in)  :: li_src(3, nli)       ! (p,i,j) source in flim
    real(r8), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    integer,  intent(in)  :: pr_src(3, npair)     ! (p,i,j) source cell
    integer,  intent(in)  :: pr_dst(npair)        ! destination cell (1-based)
    real(r8), intent(in)  :: smat(6, npair)       ! overlap moments
    real(r8), intent(in)  :: area_b(nb)
    integer,  intent(in)  :: limiter_mode         ! 0 none, 1 shape, 2 posdef
    real(r8), intent(in)  :: f_in(ncol, nslice)
    real(r8), intent(out) :: f_out(nb, nslice)

    real(r8), allocatable :: fext(:,:,:), flim(:,:,:), rec(:,:,:,:)
    integer  :: is, p, i, j, ip

    !$omp parallel default(shared) private(fext, flim, rec, is, p, i, j, ip)
    allocate(fext(-1:n+2, -1:n+2, 6), flim(0:n+1, 0:n+1, 6), rec(6, n, n, 6))
    !$omp do schedule(dynamic)
    do is = 1, nslice
      call slice_recon(n, ncol, col2pij, nhi, hi_dst, hi_src, hi_w,      &
                       nli, li_dst, li_src, stretch, centroid, metrics,  &
                       metrics_int, vtxw, xe, limiter_mode,              &
                       f_in(:, is), fext, flim, rec)

      ! 5. overlap accumulation
      f_out(:, is) = 0.0_r8
      do ip = 1, npair
        p = pr_src(1,ip); i = pr_src(2,ip); j = pr_src(3,ip)
        f_out(pr_dst(ip), is) = f_out(pr_dst(ip), is)      &
             + smat(1,ip)*rec(1,i,j,p) + smat(2,ip)*rec(2,i,j,p) &
             + smat(3,ip)*rec(3,i,j,p) + smat(4,ip)*rec(4,i,j,p) &
             + smat(5,ip)*rec(5,i,j,p) + smat(6,ip)*rec(6,i,j,p)
      end do
      f_out(:, is) = f_out(:, is)/area_b(:)
    end do
    !$omp end do
    deallocate(fext, flim, rec)
    !$omp end parallel
  end subroutine apply_slices

  !------------------------------ masked path ------------------------------

  subroutine slice_recon_masked(n, ncol, col2pij, nhi, hi_dst, hi_src,    &
                                hi_w, nli, li_dst, li_src, stretch,       &
                                centroid, metrics, metrics_int, vtxw, xe, &
                                limiter_mode, f_slice, mask, lf,          &
                                fext, flim, vext, vlim, lfp, rec)
    ! Masked steps 1-4: only columns with mask==1 are scattered (columns
    ! without valid coordinates carry a placeholder col2pij target, which
    ! is therefore never written); halo validity propagates (interpolated
    ! slot valid iff all three sources are, raw slot copies validity).
    integer,  intent(in)  :: n, ncol, nhi, nli, limiter_mode
    integer,  intent(in)  :: col2pij(3, ncol)
    integer,  intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(r8), intent(in)  :: hi_w(3, nhi)
    integer,  intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(r8), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    real(r8), intent(in)  :: f_slice(ncol), lf(ncol)
    integer,  intent(in)  :: mask(ncol)
    real(r8), intent(out) :: fext(-1:n+2, -1:n+2, 6)
    real(r8), intent(out) :: flim(0:n+1, 0:n+1, 6)
    integer,  intent(out) :: vext(-1:n+2, -1:n+2, 6)
    integer,  intent(out) :: vlim(0:n+1, 0:n+1, 6)
    real(r8), intent(out) :: lfp(n, n, 6)
    real(r8), intent(out) :: rec(6, n, n, 6)

    integer  :: ic, s, p
    real(r8) :: acc

    fext = 0.0_r8
    vext = 0
    lfp  = 0.0_r8
    do ic = 1, ncol
      if (mask(ic) == 1) then
        fext(col2pij(2,ic), col2pij(3,ic), col2pij(1,ic)) = f_slice(ic)
        vext(col2pij(2,ic), col2pij(3,ic), col2pij(1,ic)) = 1
        lfp (col2pij(2,ic), col2pij(3,ic), col2pij(1,ic)) = lf(ic)
      end if
    end do
    flim = 0.0_r8
    vlim = 0
    do p = 1, 6
      flim(1:n, 1:n, p) = fext(1:n, 1:n, p)
      vlim(1:n, 1:n, p) = vext(1:n, 1:n, p)
    end do

    do s = 1, nhi
      if (min(vext(hi_src(2,1,s), hi_src(3,1,s), hi_src(1,1,s)),  &
              vext(hi_src(2,2,s), hi_src(3,2,s), hi_src(1,2,s)),  &
              vext(hi_src(2,3,s), hi_src(3,3,s), hi_src(1,3,s))) == 1) then
        acc = hi_w(1,s)*fext(hi_src(2,1,s), hi_src(3,1,s), hi_src(1,1,s)) &
            + hi_w(2,s)*fext(hi_src(2,2,s), hi_src(3,2,s), hi_src(1,2,s)) &
            + hi_w(3,s)*fext(hi_src(2,3,s), hi_src(3,3,s), hi_src(1,3,s))
        fext(hi_dst(2,s), hi_dst(3,s), hi_dst(1,s)) = acc
        vext(hi_dst(2,s), hi_dst(3,s), hi_dst(1,s)) = 1
      end if
    end do

    do s = 1, nli
      flim(li_dst(2,s), li_dst(3,s), li_dst(1,s)) = &
           flim(li_src(2,s), li_src(3,s), li_src(1,s))
      vlim(li_dst(2,s), li_dst(3,s), li_dst(1,s)) = &
           vlim(li_src(2,s), li_src(3,s), li_src(1,s))
    end do

    do p = 1, 6
      call recon_panel_masked(n, fext(:,:,p), flim(:,:,p), vext(:,:,p),  &
                              vlim(:,:,p), stretch, centroid, metrics,   &
                              metrics_int, vtxw, xe, limiter_mode,       &
                              rec(:,:,:,p))
    end do
  end subroutine slice_recon_masked

  subroutine apply_slices_masked(n, ncol, nslice, nb, col2pij,            &
                                 nhi, hi_dst, hi_src, hi_w,               &
                                 nli, li_dst, li_src,                     &
                                 stretch, centroid, metrics, metrics_int, &
                                 vtxw, xe, npair, pr_src, pr_dst, smat,   &
                                 area_b, limiter_mode, frac_tol, fillval, &
                                 lf, mask, f_in, f_out, frac_out)
    integer,  intent(in)  :: n, ncol, nslice, nb, nhi, nli, npair
    integer,  intent(in)  :: col2pij(3, ncol)
    integer,  intent(in)  :: hi_dst(3, nhi), hi_src(3, 3, nhi)
    real(r8), intent(in)  :: hi_w(3, nhi)
    integer,  intent(in)  :: li_dst(3, nli), li_src(3, nli)
    real(r8), intent(in)  :: stretch(7, n, n), centroid(5, n, n)
    real(r8), intent(in)  :: metrics(3, n, n), metrics_int(3, n, n)
    real(r8), intent(in)  :: vtxw(4, 5, n, n), xe(0:n)
    integer,  intent(in)  :: pr_src(3, npair), pr_dst(npair)
    real(r8), intent(in)  :: smat(6, npair), area_b(nb)
    integer,  intent(in)  :: limiter_mode
    real(r8), intent(in)  :: frac_tol        ! report cells with coverage > frac_tol*area
    real(r8), intent(in)  :: fillval         ! value for uncovered cells
    real(r8), intent(in)  :: lf(ncol)        ! per-cell area fraction (landfrac)
    integer,  intent(in)  :: mask(ncol, nslice)  ! 1 = valid, per slice
    real(r8), intent(in)  :: f_in(ncol, nslice)
    real(r8), intent(out) :: f_out(nb, nslice)
    real(r8), intent(out) :: frac_out(nb, nslice)  ! covered fraction per dst cell

    real(r8), allocatable :: fext(:,:,:), flim(:,:,:), rec(:,:,:,:)
    real(r8), allocatable :: lfp(:,:,:), num(:), den(:)
    integer,  allocatable :: vext(:,:,:), vlim(:,:,:)
    integer  :: is, p, i, j, ip, ib
    real(r8) :: wa

    !$omp parallel default(shared) private(fext, flim, vext, vlim, lfp,  &
    !$omp                                  rec, num, den, is, p, i, j,   &
    !$omp                                  ip, ib, wa)
    allocate(fext(-1:n+2, -1:n+2, 6), flim(0:n+1, 0:n+1, 6),             &
             vext(-1:n+2, -1:n+2, 6), vlim(0:n+1, 0:n+1, 6),             &
             lfp(n, n, 6), rec(6, n, n, 6), num(nb), den(nb))
    !$omp do schedule(dynamic)
    do is = 1, nslice
      call slice_recon_masked(n, ncol, col2pij, nhi, hi_dst, hi_src,     &
                              hi_w, nli, li_dst, li_src, stretch,        &
                              centroid, metrics, metrics_int, vtxw, xe,  &
                              limiter_mode, f_in(:, is), mask(:, is),    &
                              lf, fext, flim, vext, vlim, lfp, rec)

      num = 0.0_r8
      den = 0.0_r8
      do ip = 1, npair
        p = pr_src(1,ip); i = pr_src(2,ip); j = pr_src(3,ip)
        if (vext(i,j,p) == 1) then
          wa = lfp(i,j,p)
          num(pr_dst(ip)) = num(pr_dst(ip)) + wa*(                   &
                 smat(1,ip)*rec(1,i,j,p) + smat(2,ip)*rec(2,i,j,p)   &
               + smat(3,ip)*rec(3,i,j,p) + smat(4,ip)*rec(4,i,j,p)   &
               + smat(5,ip)*rec(5,i,j,p) + smat(6,ip)*rec(6,i,j,p))
          den(pr_dst(ip)) = den(pr_dst(ip)) + wa*smat(1,ip)
        end if
      end do
      do ib = 1, nb
        if (den(ib) > frac_tol*area_b(ib)) then
          f_out(ib, is) = num(ib)/den(ib)
        else
          f_out(ib, is) = fillval
        end if
        frac_out(ib, is) = den(ib)/area_b(ib)
      end do
    end do
    !$omp end do
    deallocate(fext, flim, vext, vlim, lfp, rec, num, den)
    !$omp end parallel
  end subroutine apply_slices_masked

end module cr_apply
