#include <AMReX_EBMultiFabUtilInv.H>
#include <AMReX_EBMultiFabUtilInv_C.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFabUtil_C.H>
#include <AMReX_EBMultiFabUtil_C.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_MultiCutFab.H>

#include <AMReX_VisMF.H>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace amrex
{
  void
  EB_set_covered (MultiFab& mf, Real val, bool invert_eb)
  {
    EB_set_covered(mf, 0, mf.nComp(), 0, val, invert_eb);
  }

  void
  EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow, Real val, bool invert_eb)
  {
    const auto factory = dynamic_cast<EBFArrayBoxFactory const*>(&(mf.Factory()));
    if (factory == nullptr) return;
    const auto& flags = factory->getMultiEBCellFlagFab();

    AMREX_ALWAYS_ASSERT(mf.ixType().cellCentered() || mf.ixType().nodeCentered());
    bool is_cell_centered = mf.ixType().cellCentered();
    int ng = std::min(mf.nGrow(),ngrow);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.growntilebox(ng);
        const auto& flagarr = flags.const_array(mfi);
        Array4<Real> const& arr = mf.array(mfi);

        if (is_cell_centered) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if ((flagarr(i,j,k).isCovered() && !invert_eb) || (flagarr(i,j,k).isRegular() && invert_eb)) {
						  arr(i,j,k,n+icomp) = val;
						}
					      });
        } else {
	  AMREX_HOST_DEVICE_FOR_4D (bx, ncomp, i, j, k, n,
				    {
				      eb_set_covered_nodes(i,j,k,n,icomp,arr,flagarr,val, invert_eb);
				    });
        }
      }
  }

  void
  EB_set_covered (MultiFab& mf, int icomp, int ncomp, const Vector<Real>& vals, bool invert_eb)
  {
    EB_set_covered(mf, icomp, ncomp, 0, vals, invert_eb);
  }

  void
  EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow, const Vector<Real>& a_vals, bool invert_eb)
  {
    const auto factory = dynamic_cast<EBFArrayBoxFactory const*>(&(mf.Factory()));
    if (factory == nullptr) return;
    const auto& flags = factory->getMultiEBCellFlagFab();

    AMREX_ALWAYS_ASSERT(mf.ixType().cellCentered() || mf.ixType().nodeCentered());
    bool is_cell_centered = mf.ixType().cellCentered();
    int ng = std::min(mf.nGrow(),ngrow);

    Gpu::AsyncArray<Real> vals_aa(a_vals.data(), ncomp);
    Real const* AMREX_RESTRICT vals = vals_aa.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.growntilebox(ng);
        const auto& flagarr = flags.const_array(mfi);
        Array4<Real> const& arr = mf.array(mfi);

        if (is_cell_centered) {
	  AMREX_HOST_DEVICE_FOR_4D ( bx, ncomp, i, j, k, n,
				     {
				       if ((flagarr(i,j,k).isCovered() && !invert_eb) || (flagarr(i,j,k).isRegular() && invert_eb)) {
					 arr(i,j,k,n+icomp) = vals[n];
				       }
				     });
        } else {
	  AMREX_HOST_DEVICE_FOR_4D (bx, ncomp, i, j, k, n,
				    {
				      eb_set_covered_nodes(i,j,k,n,icomp,arr,flagarr,vals,invert_eb);
				    });
        }
      }
  }

  
  void
  EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, const MultiFab& vol_fine,
		   const MultiFab& vfrac_fine, int scomp, int ncomp, const IntVect& ratio, bool invert_eb)
  {
    BL_PROFILE("EB_average_down");

    AMREX_ASSERT(S_fine.ixType().cellCentered());
    AMREX_ASSERT(S_crse.ixType().cellCentered());

    const DistributionMapping& fine_dm = S_fine.DistributionMap();
    BoxArray crse_S_fine_BA = S_fine.boxArray();
    crse_S_fine_BA.coarsen(ratio);

    MultiFab crse_S_fine(crse_S_fine_BA,fine_dm,ncomp,0,MFInfo(),FArrayBoxFactory());

    Dim3 dratio = ratio.dim3();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& tbx = mfi.tilebox();
        auto& crse_fab = crse_S_fine[mfi];
        const auto& fine_fab = S_fine[mfi];

        const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
        FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

        Array4<Real> const& crse_arr = crse_fab.array();
        Array4<Real const> const& fine_arr = fine_fab.const_array();
        Array4<Real const> const& vol = vol_fine.const_array(mfi);

        if (typ == FabType::regular || typ == FabType::covered)
	  {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
					    {
					      amrex_avgdown_with_vol(b, crse_arr, fine_arr, vol, 0, scomp, ncomp, ratio);
					    });
	  }
        else if (typ == FabType::singlevalued)
	  {
            Array4<Real const> const& vfrac = vfrac_fine.const_array(mfi);
            AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
				     {
				       eb_avgdown_with_vol(i,j,k,fine_arr,scomp,crse_arr,0,vol,vfrac,dratio,ncomp,invert_eb);
				     });
	  }
        else
	  {
            amrex::Abort("multi-valued avgdown to be implemented");
	  }
      }

    S_crse.copy(crse_S_fine,0,scomp,ncomp);
  }


  void
  EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp, int ratio, bool invert_eb)
  {
    EB_average_down(S_fine, S_crse, scomp, ncomp, IntVect(ratio),invert_eb);
  }

  void
  EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp, const IntVect& ratio, bool invert_eb)
  {
    if (!S_fine.hasEBFabFactory())
      {
        amrex::average_down(S_fine, S_crse, scomp, ncomp, ratio);
      }
    else
      {
        Dim3 dratio = ratio.dim3();

        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(S_fine.Factory());
        const auto& vfrac_fine = factory.getVolFrac();

        BL_ASSERT(S_crse.nComp() == S_fine.nComp());
        BL_ASSERT(S_crse.is_cell_centered() && S_fine.is_cell_centered());

        BoxArray crse_S_fine_BA = S_fine.boxArray(); crse_S_fine_BA.coarsen(ratio);

        if (crse_S_fine_BA == S_crse.boxArray()
            and S_fine.DistributionMap() == S_crse.DistributionMap())
	  {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(S_crse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	      {
                const Box& tbx = mfi.tilebox();
                auto& crse_fab = S_crse[mfi];
                const auto& fine_fab = S_fine[mfi];

                const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
                FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

                Array4<Real> const& crse = crse_fab.array();
                Array4<Real const> const& fine = fine_fab.const_array();

                if (typ == FabType::regular || typ == FabType::covered)
		  {
                    AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
						    {
						      amrex_avgdown(b,crse,fine,scomp,scomp,ncomp,ratio);
						    });
		  }
                else
		  {
                    Array4<Real const> const& vfrc = vfrac_fine.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
					     {
					       eb_avgdown(i,j,k,fine,scomp,crse,scomp,vfrc,dratio,ncomp,invert_eb);
					     });
		  }
	      }
	  }
        else
	  {
            MultiFab crse_S_fine(crse_S_fine_BA, S_fine.DistributionMap(),
                                 ncomp, 0, MFInfo(),FArrayBoxFactory());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	      {
                const Box& tbx = mfi.tilebox();
                auto& crse_fab = crse_S_fine[mfi];
                const auto& fine_fab = S_fine[mfi];

                const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
                FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

                Array4<Real> const& crse_arr = crse_fab.array();
                Array4<Real const> const& fine_arr = fine_fab.const_array();

                if (typ == FabType::regular || typ == FabType::covered)
		  {
                    AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
						    {
						      amrex_avgdown(b,crse_arr,fine_arr,0,scomp,ncomp,ratio);
						    });
		  }
                else if (typ == FabType::singlevalued)
		  {

                    Array4<Real const> const& vfrc = vfrac_fine.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
					     {
					       eb_avgdown(i,j,k,fine_arr,scomp,crse_arr,scomp,vfrc,dratio,ncomp,invert_eb);
					     });
		  }
                else
		  {
                    amrex::Abort("multi-valued avgdown to be implemented");
		  }
	      }

            S_crse.copy(crse_S_fine,0,scomp,ncomp);
	  }
      }
  }


  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
			      const Array<MultiFab*,AMREX_SPACEDIM>& crse,
			      int ratio, int ngcrse, bool invert_eb)
  {
    EB_average_down_faces(fine, crse, IntVect{ratio}, ngcrse, invert_eb);
  }

  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
			      const Array<MultiFab*,AMREX_SPACEDIM>& crse,
			      const IntVect& ratio, int ngcrse, bool invert_eb)
  {
    AMREX_ASSERT(crse[0]->nComp() == fine[0]->nComp());

    int ncomp = crse[0]->nComp();
    if (!(*fine[0]).hasEBFabFactory())
      {
        amrex::average_down_faces(fine, crse, ratio, ngcrse);
      }
    else
      {
        Dim3 dratio = ratio.dim3();

        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>((*fine[0]).Factory());
        const auto&  aspect = factory.getAreaFrac();

        if (isMFIterSafe(*fine[0], *crse[0]))
	  {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (int n=0; n<AMREX_SPACEDIM; ++n) {
	      for (MFIter mfi(*crse[n],TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
		  const auto& flag_fab = amrex::getEBCellFlagFab((*fine[n])[mfi]);
		  const Box& tbx = mfi.growntilebox(ngcrse);
		  FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

		  Array4<Real> const& ca = crse[n]->array(mfi);
		  Array4<Real const> const& fa = fine[n]->const_array(mfi);

		  if(typ == FabType::regular || typ == FabType::covered)
                    {
		      AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
						      {
							amrex_avgdown_faces(b, ca, fa, 0, 0, ncomp, ratio, n);
						      });
                    }
		  else
                    {
		      Array4<Real const> const& ap = aspect[n]->const_array(mfi);
		      if (n == 0) {
			AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
						 {
						   eb_avgdown_face_x(i,j,k,fa,0,ca,0,ap,dratio,ncomp,invert_eb);
						 });
		      } else if (n == 1) {
			AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
						 {
						   eb_avgdown_face_y(i,j,k,fa,0,ca,0,ap,dratio,ncomp,invert_eb);
						 });
		      } else {
#if (AMREX_SPACEDIM == 3)
			AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
						 {
						   eb_avgdown_face_z(i,j,k,fa,0,ca,0,ap,dratio,ncomp,invert_eb);
						 });
#endif
		      }
                    }
                }
            }
	  }
        else
	  {
            Array<MultiFab,AMREX_SPACEDIM> ctmp;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
	      {
                BoxArray cba = fine[idim]->boxArray();
                cba.coarsen(ratio);
                ctmp[idim].define(cba, fine[idim]->DistributionMap(), ncomp, ngcrse, MFInfo(), FArrayBoxFactory());
	      }
            EB_average_down_faces(fine, amrex::GetArrOfPtrs(ctmp), ratio, ngcrse, invert_eb);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
	      {
                crse[idim]->ParallelCopy(ctmp[idim],0,0,ncomp,ngcrse,ngcrse);
	      }
	  }
      }
  }

  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
                            const Array<MultiFab*,AMREX_SPACEDIM>& crse,
			      const IntVect& ratio, const Geometry& crse_geom, bool invert_eb)
{
    AMREX_ASSERT(crse[0]->nComp() == fine[0]->nComp());

    if (!(*fine[0]).hasEBFabFactory())
    {
        amrex::average_down_faces(fine, crse, ratio, crse_geom);
    }
    else
    {
        int ngcrse = 0;
        int ncomp = crse[0]->nComp();
        Array<MultiFab,AMREX_SPACEDIM> ctmp;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray cba = fine[idim]->boxArray();
            cba.coarsen(ratio);
            ctmp[idim].define(cba, fine[idim]->DistributionMap(), ncomp, ngcrse, MFInfo(), FArrayBoxFactory());
        }
        EB_average_down_faces(fine, amrex::GetArrOfPtrs(ctmp), ratio, ngcrse, invert_eb);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            crse[idim]->ParallelCopy(ctmp[idim],0,0,ncomp,crse_geom.periodicity());
        }
    }
}


  void
  EB_average_face_to_cellcenter (MultiFab& ccmf, int dcomp,
				 const Array<MultiFab const*,AMREX_SPACEDIM>& fmf, bool invert_eb)
  {
    AMREX_ASSERT(ccmf.nComp() >= dcomp + AMREX_SPACEDIM);

    if (!fmf[0]->hasEBFabFactory())
      {
        average_face_to_cellcenter(ccmf, dcomp, fmf);
      }
    else
      {
        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(fmf[0]->Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& area = factory.getAreaFrac();

        MFItInfo info;
        if (Gpu::notInLaunchRegion()) info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ccmf,info); mfi.isValid(); ++mfi)
	  {
            const Box& bx = mfi.tilebox();
            const auto& flagfab = flags[mfi];
            Array4<Real> const& ccfab = ccmf.array(mfi);
            AMREX_D_TERM(Array4<Real const> const& xfab = fmf[0]->const_array(mfi);,
                         Array4<Real const> const& yfab = fmf[1]->const_array(mfi);,
                         Array4<Real const> const& zfab = fmf[2]->const_array(mfi););
            const auto fabtyp = flagfab.getType(bx);
            if ( (fabtyp == FabType::covered && !invert_eb) || (fabtyp == FabType::regular && invert_eb) ) {
	      AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
				       {
					 ccfab(i,j,k,dcomp) = 0.0;
				       });
            } else if ((fabtyp == FabType::regular && !invert_eb) || (fabtyp == FabType::covered && invert_eb) ) {
	      AMREX_LAUNCH_HOST_DEVICE_LAMBDA(bx, b,
					      {
						amrex_avg_fc_to_cc(b,ccfab,AMREX_D_DECL(xfab,yfab,zfab),dcomp);
					      });
            } else {
	      AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
			   Array4<Real const> const& apy = area[1]->const_array(mfi);,
			   Array4<Real const> const& apz = area[2]->const_array(mfi););
	      Array4<EBCellFlag const> const& flagarr = flagfab.const_array();
	      AMREX_HOST_DEVICE_FOR_3D(bx,i,j,k,
				       {
					 eb_avg_fc_to_cc(i,j,k,dcomp,ccfab,AMREX_D_DECL(xfab,yfab,zfab),
							 AMREX_D_DECL(apx,apy,apz),flagarr,invert_eb);
				       });
            }
	  }
      }
  }

}
