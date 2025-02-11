#include <AMReX_MLEBABecLapInv.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBMultiFabUtilInv.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MLABecLap_K.H>
#include <AMReX_MLEBABecLap_K.H>
#include <AMReX_MLEBABecLapInv_K.H>
#include <AMReX_MLLinOp_K.H>

#ifdef AMREX_USE_HYPRE
#include <AMReX_HypreABecLap3.H>
#include <AMReX_HypreABecLap3Inv.H>
#endif

#ifdef AMREX_USE_PETSC
#include <petscksp.h>
#include <AMReX_PETSc.H>
#endif

namespace amrex {

  MLEBABecLapInv::MLEBABecLapInv (const Vector<Geometry>& a_geom,
				  const Vector<BoxArray>& a_grids,
				  const Vector<DistributionMapping>& a_dmap,
				  const LPInfo& a_info,
				  const Vector<EBFArrayBoxFactory const*>& a_factory,
				  const int a_ncomp,
				  const bool& invert_eb)
    : m_ncomp(a_ncomp)
  {
    define(a_geom, a_grids, a_dmap, a_info, a_factory, invert_eb);
  }

  std::unique_ptr<FabFactory<FArrayBox> >
  MLEBABecLapInv::makeFactory (int amrlev, int mglev) const
  {
    return makeEBFabFactory(m_geom[amrlev][mglev],
                            m_grids[amrlev][mglev],
                            m_dmap[amrlev][mglev],
                            {1,1,1}, EBSupport::full);
  }

  void
  MLEBABecLapInv::define (const Vector<Geometry>& a_geom,
			  const Vector<BoxArray>& a_grids,
			  const Vector<DistributionMapping>& a_dmap,
			  const LPInfo& a_info,
			  const Vector<EBFArrayBoxFactory const*>& a_factory,
			  const bool invert_eb)
  {
    BL_PROFILE("MLEBABecLapInv::define()");

    Vector<FabFactory<FArrayBox> const*> _factory;
    for (auto x : a_factory) {
      _factory.push_back(static_cast<FabFactory<FArrayBox> const*>(x));
    }

    MLCellABecLap::define(a_geom, a_grids, a_dmap, a_info, _factory);

    const int ncomp = getNComp();

    m_invert_eb=invert_eb;
    
    m_a_coeffs.resize(m_num_amr_levels);
    m_b_coeffs.resize(m_num_amr_levels);
    m_cc_mask.resize(m_num_amr_levels);
    m_eb_phi.resize(m_num_amr_levels);
    m_eb_b_coeffs.resize(m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
      {
        m_a_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        m_b_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        m_cc_mask[amrlev].resize(m_num_mg_levels[amrlev]);
        m_eb_b_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev)
	  {
            m_a_coeffs[amrlev][mglev].define(m_grids[amrlev][mglev],
                                             m_dmap[amrlev][mglev],
                                             1, 0, MFInfo(), *m_factory[amrlev][mglev]);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
	      {
                const BoxArray& ba = amrex::convert(m_grids[amrlev][mglev],
                                                    IntVect::TheDimensionVector(idim));
                const int ng = 1;
                m_b_coeffs[amrlev][mglev][idim].define(ba,
                                                       m_dmap[amrlev][mglev],
                                                       ncomp, ng, MFInfo(), *m_factory[amrlev][mglev]);
                m_b_coeffs[amrlev][mglev][idim].setVal(0.0);
	      }

            m_cc_mask[amrlev][mglev].define(m_grids[amrlev][mglev], m_dmap[amrlev][mglev], 1, 1);

	    m_cc_mask[amrlev][mglev].BuildMask(m_geom[amrlev][mglev].Domain(),
					       m_geom[amrlev][mglev].periodicity(),
					       1, 0, 0, 1); 
	    
	  }
      }
    // Default to cell center; can be re-set to cell centroid via setPhiOnCentroid call
    m_phi_loc = Location::CellCenter;
  }

  MLEBABecLapInv::~MLEBABecLapInv ()
  {}

  void
  MLEBABecLapInv::setPhiOnCentroid ()
  {
    m_phi_loc = Location::CellCentroid;
  }


  void
  MLEBABecLapInv::setScalars (Real a, Real b)
  {
    m_a_scalar = a;
    m_b_scalar = b;
    if (a == 0.0)
      {
        for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
	  {
            m_a_coeffs[amrlev][0].setVal(0.0);
	  }
      }
  }

  void
  MLEBABecLapInv::setACoeffs (int amrlev, const MultiFab& alpha)
  {
    MultiFab::Copy(m_a_coeffs[amrlev][0], alpha, 0, 0, 1, 0);
    m_needs_update = true;
  }

  void
  MLEBABecLapInv::setACoeffs (int amrlev, Real alpha)
  {
    m_a_coeffs[amrlev][0].setVal(alpha);
    m_needs_update = true;
  }

  void
  MLEBABecLapInv::setBCoeffs (int amrlev, const Array<MultiFab const*,AMREX_SPACEDIM>& beta,
			   Location a_beta_loc)
  {
    const int ncomp = getNComp();
    const int beta_ncomp = beta[0]->nComp();

    m_beta_loc     = a_beta_loc;

    AMREX_ALWAYS_ASSERT(beta_ncomp == 1 or beta_ncomp == ncomp);
    if (beta[0]->nComp() == ncomp) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
	for (int icomp = 0; icomp < ncomp; ++icomp) {
	  MultiFab::Copy(m_b_coeffs[amrlev][0][idim], *beta[idim], icomp, icomp, 1, 0);
	}
      }
    } else {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
	for (int icomp = 0; icomp < ncomp; ++icomp) {
	  MultiFab::Copy(m_b_coeffs[amrlev][0][idim], *beta[idim], 0, icomp, 1, 0);
	}
      }
    }
    m_needs_update = true;
  }

  void
  MLEBABecLapInv::setBCoeffs (int amrlev, Real beta)
  {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_b_coeffs[amrlev][0][idim].setVal(beta);
    }
    m_needs_update = true;
    m_beta_loc     = Location::FaceCenter;
  }

  void
  MLEBABecLapInv::setBCoeffs (int amrlev, Vector<Real> const& beta)
  {
    const int ncomp = getNComp();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
	m_b_coeffs[amrlev][0][idim].setVal(beta[icomp]);
      }
    }
    m_needs_update = true;
    m_beta_loc     = Location::FaceCenter;
  }

  void
  MLEBABecLapInv::setEBDirichlet (int amrlev, const MultiFab& phi, const MultiFab& beta)
  {
    const int ncomp = getNComp();
    const int beta_ncomp = beta.nComp();
    AMREX_ALWAYS_ASSERT(beta_ncomp == 1 or beta_ncomp == ncomp);

    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phiout = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						phiout(i,j,k,n) = 0.0;
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  Array4<Real const> const& phiin = phi.const_array(mfi);
	  Array4<Real const> const& betain = beta.const_array(mfi);
	  const auto& flag = flags->const_array(mfi);
	  if (beta_ncomp == ncomp) {
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
						{
						  if (flag(i,j,k).isSingleValued()) {
						    phiout(i,j,k,n) = phiin(i,j,k,n);
						    betaout(i,j,k,n) = betain(i,j,k,n);
						  } else {
						    phiout(i,j,k,n) = 0.0;
						    betaout(i,j,k,n) = 0.0;
						  }
						});
	  } else {
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
						{
						  if (flag(i,j,k).isSingleValued()) {
						    phiout(i,j,k,n) = phiin(i,j,k,n);
						    betaout(i,j,k,n) = betain(i,j,k,0);
						  } else {
						    phiout(i,j,k,n) = 0.0;
						    betaout(i,j,k,n) = 0.0;
						  }
						});
	  }
        }
      }
  }

  void
  MLEBABecLapInv::setEBDirichlet (int amrlev, const MultiFab& phi, Real beta)
  {
    const int ncomp = getNComp();
    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phiout = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						phiout(i,j,k,n) = 0.0;
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  Array4<Real const> const& phiin = phi.const_array(mfi);
	  const auto& flag = flags->const_array(mfi);
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if (flag(i,j,k).isSingleValued()) {
						  phiout(i,j,k,n) = phiin(i,j,k,n);
						  betaout(i,j,k,n) = beta;
						} else {
						  phiout(i,j,k,n) = 0.0;
						  betaout(i,j,k,n) = 0.0;
						}
					      });
        }
      }
  }

  void
  MLEBABecLapInv::setEBDirichlet (int amrlev, const MultiFab& phi, Vector<Real> const& hv_beta)
  {
    const int ncomp = getNComp();
    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    Gpu::DeviceVector<Real> dv_beta(hv_beta.size());
    Gpu::copy(Gpu::hostToDevice, hv_beta.begin(), hv_beta.end(), dv_beta.begin());
    Real const* beta = dv_beta.data();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phiout = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						phiout(i,j,k,n) = 0.0;
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  Array4<Real const> const& phiin = phi.const_array(mfi);
	  const auto& flag = flags->const_array(mfi);
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if (flag(i,j,k).isSingleValued()) {
						  phiout(i,j,k,n) = phiin(i,j,k,n);
						  betaout(i,j,k,n) = beta[n];
						} else {
						  phiout(i,j,k,n) = 0.0;
						  betaout(i,j,k,n) = 0.0;
						}
					      });
        }
      }
  }


  void
  MLEBABecLapInv::setEBHomogDirichlet (int amrlev, const MultiFab& beta)
  {
    const int ncomp = getNComp();
    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_eb_phi[amrlev], mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phifab = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					    {
					      phifab(i,j,k,n) = 0.0;
					    });
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  Array4<Real const> const& betain = beta.const_array(mfi);
	  const auto& flag = flags->const_array(mfi);
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if (flag(i,j,k).isSingleValued()) {
						  betaout(i,j,k,n) = betain(i,j,k,0);
						} else {
						  betaout(i,j,k,n) = 0.0;
						}
					      });
        }
      }
  }

  void
  MLEBABecLapInv::setEBHomogDirichlet (int amrlev, Real beta)
  {
    const int ncomp = getNComp();
    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_eb_phi[amrlev], mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phifab = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					    {
					      phifab(i,j,k,n) = 0.0;
					    });
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  const auto& flag = flags->const_array(mfi);
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if (flag(i,j,k).isSingleValued()) {
						  betaout(i,j,k,n) = beta;
						} else {
						  betaout(i,j,k,n) = 0.0;
						}
					      });
        }
      }
  }

  void
  MLEBABecLapInv::setEBHomogDirichlet (int amrlev, Vector<Real> const& hv_beta)
  {
    const int ncomp = getNComp();
    if (m_eb_phi[amrlev] == nullptr) {
      const int mglev = 0;
      m_eb_phi[amrlev].reset(new MultiFab(m_grids[amrlev][mglev], m_dmap[amrlev][mglev],
					  ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]));
    }
    if (m_eb_b_coeffs[amrlev][0] == nullptr) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	m_eb_b_coeffs[amrlev][mglev].reset(new MultiFab(m_grids[amrlev][mglev],
							m_dmap[amrlev][mglev],
							ncomp, 0, MFInfo(),
							*m_factory[amrlev][mglev]));
      }
    }

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    Gpu::DeviceVector<Real> dv_beta(hv_beta.size());
    Gpu::copy(Gpu::hostToDevice, hv_beta.begin(), hv_beta.end(), dv_beta.begin());
    Real const* beta = dv_beta.data();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_eb_phi[amrlev], mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& phifab = m_eb_phi[amrlev]->array(mfi);
        Array4<Real> const& betaout = m_eb_b_coeffs[amrlev][0]->array(mfi);
        FabType t = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					    {
					      phifab(i,j,k,n) = 0.0;
					    });
        if (FabType::regular == t or FabType::covered == t) {
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						betaout(i,j,k,n) = 0.0;
					      });
        } else {
	  const auto& flag = flags->const_array(mfi);
	  AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
					      {
						if (flag(i,j,k).isSingleValued()) {
						  betaout(i,j,k,n) = beta[n];
						} else {
						  betaout(i,j,k,n) = 0.0;
						}
					      });
        }
      }
  }
  
  void
  MLEBABecLapInv::averageDownCoeffs ()
  {
    for (int amrlev = m_num_amr_levels-1; amrlev > 0; --amrlev)
      {
        auto& fine_a_coeffs = m_a_coeffs[amrlev];
        auto& fine_b_coeffs = m_b_coeffs[amrlev];
        
	averageDownCoeffsSameAmrLevel(amrlev, fine_a_coeffs, fine_b_coeffs,
                                      amrex::GetVecOfPtrs(m_eb_b_coeffs[0]));
        averageDownCoeffsToCoarseAmrLevel(amrlev);
      }

    averageDownCoeffsSameAmrLevel(0, m_a_coeffs[0], m_b_coeffs[0],
                                  amrex::GetVecOfPtrs(m_eb_b_coeffs[0]));

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
      for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
	  m_b_coeffs[amrlev][mglev][idim].FillBoundary(m_geom[amrlev][mglev].periodicity());
	}
      }
    }
  }

  void
  MLEBABecLapInv::averageDownCoeffsSameAmrLevel (int amrlev, Vector<MultiFab>& a,
						 Vector<Array<MultiFab,AMREX_SPACEDIM> >& b,
						 const Vector<MultiFab*>& b_eb)
  {
    int nmglevs = a.size();
    for (int mglev = 1; mglev < nmglevs; ++mglev)
      {
        IntVect ratio = (amrlev > 0) ? IntVect(mg_coarsen_ratio) : mg_coarsen_ratio_vec[mglev-1];
	
        if (m_a_scalar == 0.0)
	  {
            a[mglev].setVal(0.0);
	  }
        else
	  {
            amrex::EB_average_down(a[mglev-1], a[mglev], 0, 1, ratio, m_invert_eb);
	  }

        amrex::EB_average_down_faces(amrex::GetArrOfConstPtrs(b[mglev-1]),
                                     amrex::GetArrOfPtrs(b[mglev]),
                                     ratio, 0, m_invert_eb);

        if (b_eb[mglev])
	  {
	    amrex::EB_average_down_boundaries(*b_eb[mglev-1], *b_eb[mglev],
                                              ratio, 0);
	  }
      }
  }

  void
  MLEBABecLapInv::averageDownCoeffsToCoarseAmrLevel (int flev)
  {
    auto const& fine_a_coeffs = m_a_coeffs[flev  ].back();
    auto const& fine_b_coeffs = m_b_coeffs[flev  ].back();
    auto      & crse_a_coeffs = m_a_coeffs[flev-1].front();
    auto      & crse_b_coeffs = m_b_coeffs[flev-1].front();
    auto const& fine_eb_b_coeffs = m_eb_b_coeffs[flev  ].back();
    auto      & crse_eb_b_coeffs = m_eb_b_coeffs[flev-1].front();

    if (m_a_scalar != 0.0) {
      amrex::EB_average_down(fine_a_coeffs, crse_a_coeffs, 0, 1, mg_coarsen_ratio, m_invert_eb);
    }

    amrex::EB_average_down_faces(amrex::GetArrOfConstPtrs(fine_b_coeffs),
                                 amrex::GetArrOfPtrs(crse_b_coeffs),
                                 IntVect(mg_coarsen_ratio), m_geom[flev-1][0], m_invert_eb);

    if (fine_eb_b_coeffs) {
      amrex::EB_average_down_boundaries(*fine_eb_b_coeffs, *crse_eb_b_coeffs, mg_coarsen_ratio, 0);
    }
  }

  void
  MLEBABecLapInv::prepareForSolve ()
  {
    BL_PROFILE("MLEBABecLapInv::prepareForSolve()");
    MLCellABecLap::prepareForSolve();
    
    averageDownCoeffs();

    if (m_eb_phi[0]) {
      for (int amrlev = m_num_amr_levels-1; amrlev > 0; --amrlev) {
	amrex::EB_average_down_boundaries(*m_eb_phi[amrlev], *m_eb_phi[amrlev-1],
					  mg_coarsen_ratio, 0);
      }
    }

    m_is_singular.clear();
    m_is_singular.resize(m_num_amr_levels, false);
    auto itlo = std::find(m_lobc[0].begin(), m_lobc[0].end(), BCType::Dirichlet);
    auto ithi = std::find(m_hibc[0].begin(), m_hibc[0].end(), BCType::Dirichlet);
    if (itlo == m_lobc[0].end() && ithi == m_hibc[0].end() && !isEBDirichlet())
      {  // No Dirichlet
        for (int alev = 0; alev < m_num_amr_levels; ++alev)
	  {
	    if (m_domain_covered[alev])
	      {
                if (m_a_scalar == 0.0)
		  {
                    m_is_singular[alev] = true;
		  }
                else
		  {
                    Real asum = m_a_coeffs[alev].back().sum();
                    Real amax = m_a_coeffs[alev].back().norm0();
                    m_is_singular[alev] = (asum <= amax * 1.e-12);
		  }
	      }
	  }
      }

    m_needs_update = false;
  }

  void
  MLEBABecLapInv::Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const
  {
    BL_PROFILE("MLEBABecLapInv::Fapply()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
		 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
		 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];
    
    const auto dxinvarr = m_geom[amrlev][mglev].InvCellSizeArray();

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    auto fcent = (factory) ? ( (m_invert_eb) ? factory->getInvFaceCent() : factory->getFaceCent() )
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    
    const MultiCutFab* barea = (factory) ? &(factory->getBndryArea()) : nullptr;
    const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;

    const bool is_eb_dirichlet =  isEBDirichlet();
    const bool is_eb_inhomog = m_is_eb_inhomog;

    const int ncomp = getNComp();

    Array4<Real const> foo;

    const Real ascalar = m_a_scalar;
    const Real bscalar = m_b_scalar;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, mfi_info); mfi.isValid(); ++mfi)
      {
	const Box& bx = mfi.tilebox();
	Array4<Real const> const& xfab = in.const_array(mfi);
	Array4<Real> const& yfab = out.array(mfi);
	Array4<Real const> const& afab = acoef.const_array(mfi);
	AMREX_D_TERM(Array4<Real const> const& bxfab = bxcoef.const_array(mfi);,
		     Array4<Real const> const& byfab = bycoef.const_array(mfi);,
		     Array4<Real const> const& bzfab = bzcoef.const_array(mfi););

	auto fabtyp = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;

	if (fabtyp == FabType::covered) {
	  if(!m_invert_eb) {
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D( bx, ncomp, i, j, k, n,
					       {
						 yfab(i,j,k,n) = 0.0;
					       });
	  }
	  else { //invert_eb
	    AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
					      {
						mlabeclap_adotx(tbx, yfab, xfab, afab,
								AMREX_D_DECL(bxfab,byfab,bzfab),
								dxinvarr, ascalar, bscalar, ncomp);
					      });
	  }
	}
      
	else if (fabtyp == FabType::regular) {
	  if(!m_invert_eb) {
	    AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
					      {
						mlabeclap_adotx(tbx, yfab, xfab, afab,
								AMREX_D_DECL(bxfab,byfab,bzfab),
								dxinvarr, ascalar, bscalar, ncomp);
					      });
	  }
	  else { //invert_eb -> zero outside EB
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D( bx, ncomp, i, j, k, n,
					       {
						 yfab(i,j,k,n) = 0.0;
					       });
	  }	
	}
      
	else { //cut-cell
	  Array4<int const> const& ccmfab = ccmask.const_array(mfi);
	  Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
	  Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
	  AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
		       Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
		       Array4<Real const> const& apzfab = area[2]->const_array(mfi););
	  AMREX_D_TERM(Array4<Real const> const& fcxfab = fcent[0]->const_array(mfi);,
		       Array4<Real const> const& fcyfab = fcent[1]->const_array(mfi);,
		       Array4<Real const> const& fczfab = fcent[2]->const_array(mfi););
	  Array4<Real const> const& bafab = barea->const_array(mfi);
	  Array4<Real const> const& bcfab = bcent->const_array(mfi);
	  Array4<Real const> const& bebfab = (is_eb_dirichlet)
	    ? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;
	  Array4<Real const> const& phiebfab = (is_eb_dirichlet && m_is_eb_inhomog)
	    ? m_eb_phi[amrlev]->const_array(mfi) : foo;

	  bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
	  bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);
	  
	  if (phi_on_centroid) amrex::Abort("phi_on_centroid is still a WIP");

	  AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
					    {
					      mlebabeclapinv_adotx(tbx, yfab, xfab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
								   ccmfab, flagfab, vfracfab,
								   AMREX_D_DECL(apxfab,apyfab,apzfab),
								   AMREX_D_DECL(fcxfab,fcyfab,fczfab),
								   bafab, bcfab, bebfab,
								   is_eb_dirichlet, phiebfab, is_eb_inhomog, dxinvarr,
								   ascalar, bscalar, ncomp, beta_on_centroid, phi_on_centroid,
								   m_invert_eb);
					    });
	}
      }
  }

  void
  MLEBABecLapInv::Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const
  {
    BL_PROFILE("MLEBABecLapInv::Fsmooth()");
    
    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];
    const auto& undrrelxr = m_undrrelxr[amrlev][mglev];
    const auto& maskvals  = m_maskvals [amrlev][mglev];

    OrientationIter oitr;

    const FabSet& f0 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f1 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 1)
    const FabSet& f2 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f3 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 2)
    const FabSet& f4 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f5 = undrrelxr[oitr()]; ++oitr;
#endif
#endif

    const MultiMask& mm0 = maskvals[0];
    const MultiMask& mm1 = maskvals[1];
#if (AMREX_SPACEDIM > 1)
    const MultiMask& mm2 = maskvals[2];
    const MultiMask& mm3 = maskvals[3];
#if (AMREX_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[4];
    const MultiMask& mm5 = maskvals[5];
#endif
#endif

    const int nc = getNComp();
    const Real* h = m_geom[amrlev][mglev].CellSize();
    AMREX_D_TERM(const Real dhx = m_b_scalar/(h[0]*h[0]);,
                 const Real dhy = m_b_scalar/(h[1]*h[1]);,
                 const Real dhz = m_b_scalar/(h[2]*h[2]));
    const Real alpha = m_a_scalar;

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    auto fcent = (factory) ? ( (m_invert_eb) ? factory->getInvFaceCent() : factory->getFaceCent() )
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    const MultiCutFab* barea = (factory) ? &(factory->getBndryArea()) : nullptr;
    const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;

    bool is_eb_dirichlet =  isEBDirichlet();

    Array4<Real const> foo;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol, mfi_info);  mfi.isValid(); ++mfi)
      {
	const auto& m0 = mm0.array(mfi);
        const auto& m1 = mm1.array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& m2 = mm2.array(mfi);
        const auto& m3 = mm3.array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& m4 = mm4.array(mfi);
        const auto& m5 = mm5.array(mfi);
#endif
#endif

        const Box& vbx = mfi.validbox();
        const auto& solnfab = sol.array(mfi);
        const auto& rhsfab  = rhs.const_array(mfi);
        const auto& afab    = acoef.const_array(mfi);

        AMREX_D_TERM(const auto& bxfab = bxcoef.const_array(mfi);,
                     const auto& byfab = bycoef.const_array(mfi);,
                     const auto& bzfab = bzcoef.const_array(mfi););

        const auto& f0fab = f0.const_array(mfi);
        const auto& f1fab = f1.const_array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& f2fab = f2.const_array(mfi);
        const auto& f3fab = f3.const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& f4fab = f4.const_array(mfi);
        const auto& f5fab = f5.const_array(mfi);
#endif
#endif

        auto fabtyp = (flags) ? (*flags)[mfi].getType(vbx) : FabType::regular;

        if (fabtyp == FabType::covered)
	  {
	    if(!m_invert_eb) {
	      AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( vbx, nc, i, j, k, n,
						  {
						    solnfab(i,j,k,n) = 0.0;
						  });
	    }
	    else { //invert_eb
	      AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
						{
						  abec_gsrb(thread_box, solnfab, rhsfab, alpha, afab,
							    AMREX_D_DECL(dhx, dhy, dhz),
							    AMREX_D_DECL(bxfab, byfab, bzfab),
							    AMREX_D_DECL(m0,m2,m4),
							    AMREX_D_DECL(m1,m3,m5),
							    AMREX_D_DECL(f0fab,f2fab,f4fab),
							    AMREX_D_DECL(f1fab,f3fab,f5fab),
							    vbx, redblack, nc);
						});
	    }
	  }
	
        else if (fabtyp == FabType::regular)
	  {
	    if(!m_invert_eb) {
	      AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
						{
						  abec_gsrb(thread_box, solnfab, rhsfab, alpha, afab,
							    AMREX_D_DECL(dhx, dhy, dhz),
							    AMREX_D_DECL(bxfab, byfab, bzfab),
							    AMREX_D_DECL(m0,m2,m4),
							    AMREX_D_DECL(m1,m3,m5),
							    AMREX_D_DECL(f0fab,f2fab,f4fab),
							    AMREX_D_DECL(f1fab,f3fab,f5fab),
							    vbx, redblack, nc);
						});
	    }
	    else { //invert_eb
	      AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( vbx, nc, i, j, k, n,
						  {
						    solnfab(i,j,k,n) = 0.0;
						  });
	    }
	  }
	
        else //cut-cell
	  {
            Array4<int const> const& ccmfab = ccmask.const_array(mfi);
            Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
            Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
            AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
                         Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
                         Array4<Real const> const& apzfab = area[2]->const_array(mfi););
            AMREX_D_TERM(Array4<Real const> const& fcxfab = fcent[0]->const_array(mfi);,
                         Array4<Real const> const& fcyfab = fcent[1]->const_array(mfi);,
                         Array4<Real const> const& fczfab = fcent[2]->const_array(mfi););
            Array4<Real const> const& bafab = barea->const_array(mfi);
            Array4<Real const> const& bcfab = bcent->const_array(mfi);
            Array4<Real const> const& bebfab = (is_eb_dirichlet)
	      ? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;

            bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
            bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);

            if (phi_on_centroid) amrex::Abort("phi_on_centroid is still a WIP");
	    
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
					      {
						mlebabeclapinv_gsrb(thread_box, solnfab, rhsfab, alpha, afab,
								    AMREX_D_DECL(dhx, dhy, dhz),
								    AMREX_D_DECL(bxfab,byfab,bzfab),
								    AMREX_D_DECL(m0,m2,m4),
								    AMREX_D_DECL(m1,m3,m5),
								    AMREX_D_DECL(f0fab,f2fab,f4fab),
								    AMREX_D_DECL(f1fab,f3fab,f5fab),
								    ccmfab, flagfab, vfracfab,
								    AMREX_D_DECL(apxfab,apyfab,apzfab),
								    AMREX_D_DECL(fcxfab,fcyfab,fczfab),
								    bafab, bcfab, bebfab,
								    is_eb_dirichlet, beta_on_centroid, phi_on_centroid,
								    vbx, redblack, nc, m_invert_eb);
					      });
	  }
      }

  }

  void
  MLEBABecLapInv::FFlux (int amrlev, const MFIter& mfi, const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
			 const FArrayBox& sol, Location flux_loc, const int face_only) const
  {
    BL_PROFILE("MLEBABecLapInv::FFlux()");
    const int compute_flux_at_centroid = (Location::FaceCentroid == flux_loc) ? 1 : 0;
    const int mglev = 0; 
    const Box& box = mfi.tilebox();
    const int ncomp = getNComp();

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get()); 
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr; 

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    AMREX_D_TERM(const auto& bx = m_b_coeffs[amrlev][mglev][0][mfi];,
                 const auto& by = m_b_coeffs[amrlev][mglev][1][mfi];,
                 const auto& bz = m_b_coeffs[amrlev][mglev][2][mfi];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];

    AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(box,0);,
                 Box const& ybx = amrex::surroundingNodes(box,1);,
                 Box const& zbx = amrex::surroundingNodes(box,2););
    AMREX_D_TERM(Array4<Real> const& fx = flux[0]->array();,
                 Array4<Real> const& fy = flux[1]->array();,
                 Array4<Real> const& fz = flux[2]->array(););

    const auto fabtyp = (flags) ? (*flags)[mfi].getType(box) : FabType::regular;
    
    if (fabtyp == FabType::covered) {

      if(!m_invert_eb) {
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(xbx, ncomp, i, j, k, n,
					  {
					    fx(i,j,k,n) = 0.0;
					  });
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(ybx, ncomp, i, j, k, n,
					  {
					    fy(i,j,k,n) = 0.0;
					  });
#if (AMREX_SPACEDIM == 3)
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(zbx, ncomp, i, j, k, n,
					  {
					    fz(i,j,k,n) = 0.0;
					  });
#endif
      }
      else { //invert_eb
	MLABecLaplacian::FFlux(box, dxinv, m_b_scalar,
			       Array<FArrayBox const*,AMREX_SPACEDIM>{AMREX_D_DECL(&bx,&by,&bz)},
			       flux, sol, face_only, ncomp);
      }
    }

    else if (fabtyp == FabType::regular) {
      if(!m_invert_eb)	{
	MLABecLaplacian::FFlux(box, dxinv, m_b_scalar,
			       Array<FArrayBox const*,AMREX_SPACEDIM>{AMREX_D_DECL(&bx,&by,&bz)},
			       flux, sol, face_only, ncomp);
      }
      else { //invert_eb
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(xbx, ncomp, i, j, k, n,
					  {
					    fx(i,j,k,n) = 0.0;
					  });
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(ybx, ncomp, i, j, k, n,
					  {
					    fy(i,j,k,n) = 0.0;
					  });
#if (AMREX_SPACEDIM == 3)
	AMREX_HOST_DEVICE_PARALLEL_FOR_4D(zbx, ncomp, i, j, k, n,
					  {
					    fz(i,j,k,n) = 0.0;
					  });
#endif
      }
      
    }

    else if (compute_flux_at_centroid) { //also cut-cell but at centroid...
      const auto& area = factory->getAreaFrac();
      const auto& fcent = (m_invert_eb) ? factory->getInvFaceCent() : factory->getFaceCent();
      AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
		   Array4<Real const> const& apy = area[1]->const_array(mfi);,
		   Array4<Real const> const& apz = area[2]->const_array(mfi););
      AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
		   Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
		   Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
      Array4<Real const> const& phi = sol.const_array();
      AMREX_D_TERM(Array4<Real const> const& bxcoef = bx.const_array();,
		   Array4<Real const> const& bycoef = by.const_array();,
		   Array4<Real const> const& bzcoef = bz.const_array(););
      Array4<int const> const& msk = ccmask.const_array(mfi);
      AMREX_D_TERM(Real dhx = m_b_scalar*dxinv[0];,
		   Real dhy = m_b_scalar*dxinv[1];,
		   Real dhz = m_b_scalar*dxinv[2];);

      bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
      bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);

      if (phi_on_centroid) amrex::Abort("phi_on_centroid is still a WIP");
      
      AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
				       xbx, txbx,
				       {
					 mlebabeclapinv_flux_x(txbx, fx, apx, fcx, phi, bxcoef, msk, dhx, face_only, ncomp, xbx,
							       beta_on_centroid, phi_on_centroid, m_invert_eb);
				       }
				       , ybx, tybx,
				       {
					 mlebabeclapinv_flux_y(tybx, fy, apy, fcy, phi, bycoef, msk, dhy, face_only, ncomp, ybx,
							       beta_on_centroid, phi_on_centroid, m_invert_eb);
				       }
#if (AMREX_SPACEDIM == 3)
				       , zbx, tzbx,
				       {
					 mlebabeclapinv_flux_z(tzbx, fz, apz, fcz, phi, bzcoef, msk, dhz, face_only, ncomp, zbx,
							       beta_on_centroid, phi_on_centroid, m_invert_eb);
				       }
#endif
				       );
    }

    else { //cut-cell
      const auto& area = factory->getAreaFrac();
      AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
		   Array4<Real const> const& apy = area[1]->const_array(mfi);,
		   Array4<Real const> const& apz = area[2]->const_array(mfi););
      Array4<Real const> const& phi = sol.const_array();
      AMREX_D_TERM(Array4<Real const> const& bxcoef = bx.const_array();,
		   Array4<Real const> const& bycoef = by.const_array();,
		   Array4<Real const> const& bzcoef = bz.const_array(););
      AMREX_D_TERM(Real dhx = m_b_scalar*dxinv[0];,
		   Real dhy = m_b_scalar*dxinv[1];,
		   Real dhz = m_b_scalar*dxinv[2];);

      AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
				       xbx, txbx,
				       {
					 mlebabeclapinv_flux_x_0(txbx, fx, apx, phi, bxcoef, dhx, face_only, ncomp, xbx, m_invert_eb);
				       }
				       , ybx, tybx,
				       {
					 mlebabeclapinv_flux_y_0(tybx, fy, apy, phi, bycoef, dhy, face_only, ncomp, ybx, m_invert_eb);
				       }
#if (AMREX_SPACEDIM == 3)
				       , zbx, tzbx,
				       {
					 mlebabeclapinv_flux_z_0(tzbx, fz, apz, phi, bzcoef, dhz, face_only, ncomp, zbx, m_invert_eb);
				       }
#endif
				       );
    }

  }


  //$$ compGrad is to update/fix...
  void
  MLEBABecLapInv::compGrad (int amrlev, const Array<MultiFab*,AMREX_SPACEDIM>& grad,
			    MultiFab& sol, Location grad_loc) const
  {
    BL_PROFILE("MLEBABecLapInv::compGrad()");
    
    const int ncomp = getNComp();
    const int compute_grad_at_centroid = (Location::FaceCentroid == grad_loc) ? 1 : 0;
    const int mglev = 0;
    applyBC(amrlev, mglev, sol, BCMode::Inhomogeneous, StateMode::Solution,
            m_bndry_sol[amrlev].get());

    AMREX_D_TERM(const Real dxi = m_geom[amrlev][mglev].InvCellSize(0);,
                 const Real dyi = m_geom[amrlev][mglev].InvCellSize(1);,
                 const Real dzi = m_geom[amrlev][mglev].InvCellSize(2););
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get()); 
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr; 
    auto area = (factory) ? factory->getAreaFrac() : 
      Array<const MultiCutFab*, AMREX_SPACEDIM>{AMREX_D_DECL(nullptr, nullptr, nullptr)}; 
    auto fcent = (factory) ? ( (m_invert_eb) ? factory->getInvFaceCent() : factory->getFaceCent() )
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& box = mfi.tilebox(); 
        auto fabtyp = (flags) ? (*flags)[mfi].getType(box) :FabType::regular;
        AMREX_D_TERM(Box const& fbx = mfi.nodaltilebox(0);,
                     Box const& fby = mfi.nodaltilebox(1);,
                     Box const& fbz = mfi.nodaltilebox(2););
        AMREX_D_TERM(const auto& gx = grad[0]->array(mfi);,
                     const auto& gy = grad[1]->array(mfi);,
                     const auto& gz = grad[2]->array(mfi););
        const auto& s = sol.const_array(mfi);
        if (fabtyp == FabType::covered) {
	  if(!m_invert_eb)	    {
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fbx, ncomp, i, j, k, n,
					      {
						gx(i,j,k,n) = 0.0;
					      });
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fby, ncomp, i, j, k, n,
					      {
						gy(i,j,k,n) = 0.0;
					      });
#if (AMREX_SPACEDIM == 3)
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fbz, ncomp, i, j, k, n,
					      {
						gz(i,j,k,n) = 0.0;
					      });
#endif
	  }
	  else { //invert_eb
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fbx, ncomp, i, j, k, n,
						{
						  gx(i,j,k,n) = dxi*(s(i,j,k,n) - s(i-1,j,k,n));
						});
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fby, ncomp, i, j, k, n,
						{
						  gy(i,j,k,n) = dyi*(s(i,j,k,n) - s(i,j-1,k,n));
						});
#if (AMREX_SPACEDIM == 3)
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fbz, ncomp, i, j, k, n,
						{
						  gz(i,j,k,n) = dzi*(s(i,j,k,n) - s(i,j,k-1,n));
						});
#endif    
	  }
        }
	
	else if(fabtyp == FabType::regular) {
	  if(!m_invert_eb) {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fbx, ncomp, i, j, k, n,
						{
						  gx(i,j,k,n) = dxi*(s(i,j,k,n) - s(i-1,j,k,n));
						});
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fby, ncomp, i, j, k, n,
						{
						  gy(i,j,k,n) = dyi*(s(i,j,k,n) - s(i,j-1,k,n));
						});
#if (AMREX_SPACEDIM == 3)
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( fbz, ncomp, i, j, k, n,
						{
						  gz(i,j,k,n) = dzi*(s(i,j,k,n) - s(i,j,k-1,n));
						});
#endif
	  }
	  else { //invert_eb
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fbx, ncomp, i, j, k, n,
					      {
						gx(i,j,k,n) = 0.0;
					      });
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fby, ncomp, i, j, k, n,
					      {
						gy(i,j,k,n) = 0.0;
					      });
#if (AMREX_SPACEDIM == 3)
	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D(fbz, ncomp, i, j, k, n,
					      {
						gz(i,j,k,n) = 0.0;
					      });
#endif
	  }
	  
        }

	else if (compute_grad_at_centroid) {
	  //TO FIX LATER... but not necessary $$
	  amrex::Abort("at_centroid is still a WIP");
	  
	  AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
		       Array4<Real const> const& apy = area[1]->const_array(mfi);,
		       Array4<Real const> const& apz = area[2]->const_array(mfi););
	  AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
		       Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
		       Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
	  Array4<int const> const& msk = ccmask.const_array(mfi);

	  
	  AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
					   fbx, txbx,
					   {
					     mlebabeclap_grad_x(txbx, gx, s, apx, fcx, msk, dxi, ncomp, true);
					   }
					   , fby, tybx,
					   {
					     mlebabeclap_grad_y(tybx, gy, s, apy, fcy, msk, dyi, ncomp, true); 
					   }
#if (AMREX_SPACEDIM == 3)
					   , fbz, tzbx,
					   {
					     mlebabeclap_grad_z(tzbx, gz, s, apz, fcz, msk, dzi, ncomp, true); 
					   }
#endif
					   );
        }
	
	else { //cut-cell
	  AMREX_D_TERM(Array4<Real const> const& ax = area[0]->const_array(mfi);,
		       Array4<Real const> const& ay = area[1]->const_array(mfi);,
		       Array4<Real const> const& az = area[2]->const_array(mfi););
	  AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
					   fbx, txbx,
					   {
					     mlebabeclapinv_grad_x_0(txbx, gx, s, ax, dxi, ncomp, m_invert_eb);
					   }
					   , fby, tybx,
					   {
					     mlebabeclapinv_grad_y_0(tybx, gy, s, ay, dyi, ncomp, m_invert_eb);
					   }
#if (AMREX_SPACEDIM == 3)
					   , fbz, tzbx,
					   {
					     mlebabeclapinv_grad_z_0(tzbx, gz, s, az, dzi, ncomp, m_invert_eb);
					   }
#endif
					   );
        }
      }
  }
  
  void
  MLEBABecLapInv::normalize (int amrlev, int mglev, MultiFab& mf) const
  {
    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];

    const auto dxinvarray = m_geom[amrlev][mglev].InvCellSizeArray();
    AMREX_D_TERM(Real dhx = m_b_scalar*dxinvarray[0]*dxinvarray[0];,
                 Real dhy = m_b_scalar*dxinvarray[1]*dxinvarray[1];,
                 Real dhz = m_b_scalar*dxinvarray[2]*dxinvarray[2];);

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    auto fcent = (factory) ? ( (m_invert_eb) ? factory->getInvFaceCent() : factory->getFaceCent() )
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    const MultiCutFab* barea = (factory) ? &(factory->getBndryArea()) : nullptr;
    const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;

    bool is_eb_dirichlet =  isEBDirichlet();

    Array4<Real const> foo;

    const Real ascalar = m_a_scalar;
    const Real bscalar = m_b_scalar;
    const int ncomp = getNComp();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& fab = mf.array(mfi);
        Array4<Real const> const& afab = acoef.const_array(mfi);
        AMREX_D_TERM(Array4<Real const> const& bxfab = bxcoef.const_array(mfi);,
                     Array4<Real const> const& byfab = bycoef.const_array(mfi);,
                     Array4<Real const> const& bzfab = bzcoef.const_array(mfi););

        auto fabtyp = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;
 
        if (fabtyp == FabType::regular && !m_invert_eb)
	  {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA (bx, tbx,
					     {
					       mlabeclap_normalize(tbx, fab, afab, AMREX_D_DECL(bxfab, byfab, bzfab),
								   dxinvarray, ascalar, bscalar, ncomp);
					     });
	  }
	else if (fabtyp == FabType::covered && m_invert_eb)
	  {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA (bx, tbx,
					     {
					       mlabeclap_normalize(tbx, fab, afab, AMREX_D_DECL(bxfab, byfab, bzfab),
								   dxinvarray, ascalar, bscalar, ncomp);
					     });
	  }
        else if (fabtyp == FabType::singlevalued)
	  {
            Array4<Real const> const& bebfab
	      = (is_eb_dirichlet) ? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;
            Array4<int const> const& ccmfab = ccmask.const_array(mfi);
            Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
            Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
            AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
                         Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
                         Array4<Real const> const& apzfab = area[2]->const_array(mfi););
            AMREX_D_TERM(Array4<Real const> const& fcxfab = fcent[0]->const_array(mfi);,
                         Array4<Real const> const& fcyfab = fcent[1]->const_array(mfi);,
                         Array4<Real const> const& fczfab = fcent[2]->const_array(mfi););
            Array4<Real const> const& bafab = barea->const_array(mfi);
            Array4<Real const> const& bcfab = bcent->const_array(mfi);

            bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
	    
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
					      {
						mlebabeclapinv_normalize(tbx, fab, ascalar, afab,
									 AMREX_D_DECL(dhx, dhy, dhz),
									 AMREX_D_DECL(bxfab, byfab, bzfab),
									 ccmfab, flagfab, vfracfab,
									 AMREX_D_DECL(apxfab,apyfab,apzfab),
									 AMREX_D_DECL(fcxfab,fcyfab,fczfab),
									 bafab, bcfab, bebfab, is_eb_dirichlet,
									 beta_on_centroid, ncomp, m_invert_eb);
					      });
	  }
      }
  }

  void
  MLEBABecLapInv::restriction (int amrlev, int cmglev, MultiFab& crse, MultiFab& fine) const
  {
    IntVect ratio = (amrlev > 0) ? IntVect(mg_coarsen_ratio) : mg_coarsen_ratio_vec[cmglev-1];
    const int ncomp = getNComp();
    amrex::EB_average_down(fine, crse, 0, ncomp, ratio, m_invert_eb);
  }

  void
  MLEBABecLapInv::interpolation (int amrlev, int fmglev, MultiFab& fine, const MultiFab& crse) const
  {

    BL_PROFILE("MLEBABecLapInv::interpolation()");
    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][fmglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    const int ncomp = getNComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        auto fabtyp = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;

        Array4<Real const> const& cfab = crse.const_array(mfi);
        Array4<Real> const& ffab = fine.array(mfi);

        if (fabtyp == FabType::regular && !m_invert_eb)
	  {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
						{
						  int ic = amrex::coarsen(i,2);
						  int jc = amrex::coarsen(j,2);
						  int kc = amrex::coarsen(k,2);
						  ffab(i,j,k,n) += cfab(ic,jc,kc,n);
						});
	  }
	else if (fabtyp == FabType::covered && m_invert_eb)
	  {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
						{
						  int ic = amrex::coarsen(i,2);
						  int jc = amrex::coarsen(j,2);
						  int kc = amrex::coarsen(k,2);
						  ffab(i,j,k,n) += cfab(ic,jc,kc,n);
						});
	  }	
        else if (fabtyp == FabType::singlevalued)
	  {
            Array4<EBCellFlag const> const& flg = flags->const_array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
						{
						  if ( (!flg(i,j,k).isCovered() && !m_invert_eb) || (!flg(i,j,k).isRegular() && m_invert_eb) ) {
						    int ic = amrex::coarsen(i,2);
						    int jc = amrex::coarsen(j,2);
						    int kc = amrex::coarsen(k,2);
						    ffab(i,j,k,n) += cfab(ic,jc,kc,n);
						  }
						});
	  }
      }
  }

  void
  MLEBABecLapInv::averageDownSolutionRHS (int camrlev, MultiFab& crse_sol, MultiFab& crse_rhs,
					  const MultiFab& fine_sol, const MultiFab& fine_rhs)
  {
    const auto amrrr = AMRRefRatio(camrlev);
    const int ncomp = getNComp();
    amrex::EB_average_down(fine_sol, crse_sol, 0, ncomp, amrrr, m_invert_eb);
    amrex::EB_average_down(fine_rhs, crse_rhs, 0, ncomp, amrrr, m_invert_eb);
  }


  void
  MLEBABecLapInv::applyBC (int amrlev, int mglev, MultiFab& in, BCMode bc_mode, StateMode s_mode,
			   const MLMGBndry* bndry, bool skip_fillboundary) const
  {
    BL_PROFILE("MLEBABecLapInv::applyBC()");

    // No coarsened boundary values, cannot apply inhomog at mglev>0.
    BL_ASSERT(mglev == 0 || bc_mode == BCMode::Homogeneous);
    BL_ASSERT(bndry != nullptr || bc_mode == BCMode::Homogeneous);
    const int ncomp = getNComp();
    if (!skip_fillboundary) {
      const int cross = false;
      in.FillBoundary(0, ncomp, m_geom[amrlev][mglev].periodicity(),cross);
    }

    int m_is_inhomog = bc_mode == BCMode::Inhomogeneous;
    int flagbc = m_is_inhomog;
    m_is_eb_inhomog = s_mode == StateMode::Solution;
    const int imaxorder = maxorder;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(imaxorder <= 4, "MLEBABecLapInv::applyBC: maxorder too high");

    const Real dxi = m_geom[amrlev][mglev].InvCellSize(0);
    const Real dyi = (AMREX_SPACEDIM >= 2) ? m_geom[amrlev][mglev].InvCellSize(1) : 1.0;
    const Real dzi = (AMREX_SPACEDIM == 3) ? m_geom[amrlev][mglev].InvCellSize(2) : 1.0;

    const auto& maskvals = m_maskvals[amrlev][mglev];
    const auto& bcondloc = *m_bcondloc[amrlev][mglev];

    const auto& ccmask = m_cc_mask[amrlev][mglev];
    
    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
      : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    
    FArrayBox foofab(Box::TheUnitBox(),ncomp);
    const auto& foo = foofab.array();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(in, mfi_info); mfi.isValid(); ++mfi)
      {
        const Box& vbx   = mfi.validbox();
        const auto& iofab = in.array(mfi);

        auto fabtyp = (flags) ? (*flags)[mfi].getType(vbx) : FabType::regular;
        if ( (fabtyp != FabType::covered && !m_invert_eb) || (fabtyp != FabType::regular && m_invert_eb) )
	  {
            const auto & bdlv = bcondloc.bndryLocs(mfi);
            const auto & bdcv = bcondloc.bndryConds(mfi);

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
	      {
                const Orientation olo(idim,Orientation::low);
                const Orientation ohi(idim,Orientation::high);
                Box blo = amrex::adjCellLo(vbx, idim);
                Box bhi = amrex::adjCellHi(vbx, idim);
		if ( (fabtyp != FabType::regular && !m_invert_eb) || (fabtyp != FabType::covered && m_invert_eb) ) {
		  blo.grow(IntVect(1)-IntVect::TheDimensionVector(idim));
		  bhi.grow(IntVect(1)-IntVect::TheDimensionVector(idim));
		}
                const int blen = vbx.length(idim);
                const auto& mlo = maskvals[olo].array(mfi);
                const auto& mhi = maskvals[ohi].array(mfi);
                const auto& bvlo = (bndry != nullptr) ? bndry->bndryValues(olo).array(mfi) : foo;
                const auto& bvhi = (bndry != nullptr) ? bndry->bndryValues(ohi).array(mfi) : foo;
                for (int icomp = 0; icomp < ncomp; ++icomp) {
		  const BoundCond bctlo = bdcv[icomp][olo];
		  const BoundCond bcthi = bdcv[icomp][ohi];
		  const Real bcllo = bdlv[icomp][olo];
		  const Real bclhi = bdlv[icomp][ohi];
		  if ( (fabtyp == FabType::regular && !m_invert_eb) || (fabtyp == FabType::covered && m_invert_eb) )
                    {
		      if (idim == 0) {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mllinop_apply_bc_x(0, tboxlo, blen, iofab, mlo,
									      bctlo, bcllo, bvlo,
									      imaxorder, dxi, flagbc, icomp);
							 },
							 bhi, tboxhi, {
							   mllinop_apply_bc_x(1, tboxhi, blen, iofab, mhi,
									      bcthi, bclhi, bvhi,
									      imaxorder, dxi, flagbc, icomp);
							 });
		      } else if (idim == 1) {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mllinop_apply_bc_y(0, tboxlo, blen, iofab, mlo,
									      bctlo, bcllo, bvlo,
									      imaxorder, dyi, flagbc, icomp);
							 },
							 bhi, tboxhi, {
							   mllinop_apply_bc_y(1, tboxhi, blen, iofab, mhi,
									      bcthi, bclhi, bvhi,
									      imaxorder, dyi, flagbc, icomp);
							 });
		      } else {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mllinop_apply_bc_z(0, tboxlo, blen, iofab, mlo,
									      bctlo, bcllo, bvlo,
									      imaxorder, dzi, flagbc, icomp);
							 },
							 bhi, tboxhi, {
							   mllinop_apply_bc_z(1, tboxhi, blen, iofab, mhi,
									      bcthi, bclhi, bvhi,
									      imaxorder, dzi, flagbc, icomp);
							 });
		      }
                    }
		  else // irregular (cut)
                    {
		      const auto& ap = area[idim]->const_array(mfi);
		      const auto& mask = ccmask.const_array(mfi);
		      if (idim == 0) {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mlebabeclapinv_apply_bc_x(0, tboxlo, blen, iofab, mask, ap,
										     bctlo, bcllo, bvlo,
										     imaxorder, dxi, flagbc, icomp, m_invert_eb);
							 },
							 bhi, tboxhi, {
							   mlebabeclapinv_apply_bc_x(1, tboxhi, blen, iofab, mask, ap,
										     bcthi, bclhi, bvhi,
										     imaxorder, dxi, flagbc, icomp, m_invert_eb);
							 });
		      } else if (idim == 1) {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mlebabeclapinv_apply_bc_y(0, tboxlo, blen, iofab, mask, ap,
										     bctlo, bcllo, bvlo,
										     imaxorder, dyi, flagbc, icomp, m_invert_eb);
							 },
							 bhi, tboxhi, {
							   mlebabeclapinv_apply_bc_y(1, tboxhi, blen, iofab, mask, ap,
										     bcthi, bclhi, bvhi,
										     imaxorder, dyi, flagbc, icomp, m_invert_eb);
							 });
		      } else {
			AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
							 blo, tboxlo, {
							   mlebabeclapinv_apply_bc_z(0, tboxlo, blen, iofab, mask, ap,
										     bctlo, bcllo, bvlo,
										     imaxorder, dzi, flagbc, icomp, m_invert_eb);
							 },
							 bhi, tboxhi, {
							   mlebabeclapinv_apply_bc_z(1, tboxhi, blen, iofab, mask, ap,
										     bcthi, bclhi, bvhi,
										     imaxorder, dzi, flagbc, icomp, m_invert_eb);
							 });
		      }
                    }
                }
	      }
	  }
      }
  }

  void
  MLEBABecLapInv::apply (int amrlev, int mglev, MultiFab& out, MultiFab& in, BCMode bc_mode,
			 StateMode s_mode, const MLMGBndry* bndry) const
  {
    BL_PROFILE("MLEBABecLapInv::apply()");
    applyBC(amrlev, mglev, in, bc_mode, s_mode, bndry);
    Fapply(amrlev, mglev, out, in);
  }

  void
  MLEBABecLapInv::update ()
  {
    if (MLCellABecLap::needsUpdate()) MLCellABecLap::update();

    averageDownCoeffs();

    m_is_singular.clear();
    m_is_singular.resize(m_num_amr_levels, false);
    auto itlo = std::find(m_lobc[0].begin(), m_lobc[0].end(), BCType::Dirichlet);
    auto ithi = std::find(m_hibc[0].begin(), m_hibc[0].end(), BCType::Dirichlet);
    if (itlo == m_lobc[0].end() && ithi == m_hibc[0].end() && !isEBDirichlet())
      {  // No Dirichlet
        for (int alev = 0; alev < m_num_amr_levels; ++alev)
	  {
	    if (m_domain_covered[alev] )
	      {
                if (m_a_scalar == 0.0)
		  {
                    m_is_singular[alev] = true;
		  }
                else
		  {
                    Real asum = m_a_coeffs[alev].back().sum();
                    Real amax = m_a_coeffs[alev].back().norm0();
                    m_is_singular[alev] = (asum <= amax * 1.e-12);
		  }
	      }
	  }
      }

    m_needs_update = false;
  }

  void
  MLEBABecLapInv::getEBFluxes (const Vector<MultiFab*>& a_flux, const Vector<MultiFab*>& a_sol) const
  {

    BL_PROFILE("MLEBABecLapInv::getEBFluxes()");
    const int ncomp = getNComp();
    const int mglev = 0;
    const int namrlevs = NAMRLevels();
    const bool is_eb_dirichlet =  isEBDirichlet();
    for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
      if (!is_eb_dirichlet) {
	a_flux[amrlev]->setVal(0.0); // Homogeneous Neumann
      } else {
	applyBC(amrlev, mglev, *a_sol[amrlev], BCMode::Inhomogeneous,
		StateMode::Solution, m_bndry_sol[amrlev].get());

	const auto dxinvarr = m_geom[amrlev][mglev].InvCellSizeArray();

	auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
	const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
	const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
	auto area = (factory) ? factory->getAreaFrac()
	  : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};

	const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;

	const bool is_eb_inhomog = m_is_eb_inhomog;

	Array4<Real const> foo;

	MFItInfo mfi_info;
	if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(*a_flux[amrlev], mfi_info); mfi.isValid(); ++mfi)
	  {
	    const Box& bx = mfi.tilebox();
	    Array4<Real const> const& xfab = a_sol[amrlev]->const_array(mfi);
	    Array4<Real> const& febfab = a_flux[amrlev]->array(mfi);

	    auto fabtyp = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;

	    if (fabtyp == FabType::covered or fabtyp == FabType::regular) {
	      AMREX_HOST_DEVICE_PARALLEL_FOR_4D( bx, ncomp, i, j, k, n,
						 {
						   febfab(i,j,k,n) = 0.0;
						 });
	    } else {
	      Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
	      Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
	      AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
			   Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
			   Array4<Real const> const& apzfab = area[2]->const_array(mfi););
	      Array4<Real const> const& bcfab = bcent->const_array(mfi);
	      Array4<Real const> const& bebfab = (is_eb_dirichlet)
		? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;
	      Array4<Real const> const& phiebfab = (is_eb_dirichlet && m_is_eb_inhomog)
		? m_eb_phi[amrlev]->const_array(mfi) : foo;

	      AMREX_HOST_DEVICE_FOR_4D ( bx, ncomp, i, j, k, n,
					 {
					   mlebabeclapinv_ebflux(i,j,k,n,febfab, xfab, flagfab, vfracfab,
								 AMREX_D_DECL(apxfab,apyfab,apzfab),
								 bcfab, bebfab, phiebfab,
								 is_eb_inhomog, dxinvarr, m_invert_eb);
					 });
	    }
	  }
      }
    }
  }

#ifdef AMREX_USE_HYPRE
  std::unique_ptr<Hypre>
  MLEBABecLapInv::makeHypre (Hypre::Interface hypre_interface) const
  {
    auto hypre_solver = MLCellABecLap::makeHypre(hypre_interface, m_invert_eb);
    if (m_invert_eb == true) {
      auto ijmatrix_solver = dynamic_cast<HypreABecLap3Inv*>(hypre_solver.get());
      ijmatrix_solver->setEBDirichlet(m_eb_b_coeffs[0].back().get());
    }
    else {
      auto ijmatrix_solver = dynamic_cast<HypreABecLap3*>(hypre_solver.get());
      ijmatrix_solver->setEBDirichlet(m_eb_b_coeffs[0].back().get());
    }
    return hypre_solver;
  }
#endif

#ifdef AMREX_USE_PETSC
  std::unique_ptr<PETScABecLap>
  MLEBABecLapInv::makePETSc () const
  {
    auto petsc_solver = MLCellABecLap::makePETSc();
    petsc_solver->setEBDirichlet(m_eb_b_coeffs[0].back().get());
    return petsc_solver;
  }
#endif


}
