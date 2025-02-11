#include <rincflo.H>

// using namespace amrex;

DiffusionTensorOp::DiffusionTensorOp (Rincflo* a_rincflo)
    : m_rincflo(a_rincflo)
{
    readParameters();

    int finest_level = m_rincflo->finestLevel();

    LPInfo info_solve; 
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);
    #if (AMREX_USE_EB)
    //if (!m_rincflo->EBFactory(0).isAllRegular())
    if (m_rincflo->has_eb())
        {
        Vector<EBFArrayBoxFactory const*> ebfact;
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            ebfact.push_back(&(m_rincflo->EBFactory(lev)));
        }
	
	    if (m_rincflo->useTensorSolve())
	    {
	        m_eb_solve_op.reset(
                new MLEBTensorOp(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_solve, ebfact));
            m_eb_solve_op->setMaxOrder(m_mg_maxorder);
            m_eb_solve_op->setDomainBC(
                m_rincflo->get_diffuse_tensor_bc(Orientation::low),
                m_rincflo->get_diffuse_tensor_bc(Orientation::high));
	  }
	
        if (m_rincflo->need_DivTau() || m_rincflo->useTensorCorrection())
	  {
            m_eb_apply_op.reset(
                new MLEBTensorOp(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_apply, ebfact));
            m_eb_apply_op->setMaxOrder(m_mg_maxorder);
            m_eb_apply_op->setDomainBC(
                m_rincflo->get_diffuse_tensor_bc(Orientation::low),
                m_rincflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
    else
    #endif
    {
        if (m_rincflo->useTensorSolve()) 
        {
	        m_reg_solve_op.reset(
                new MLTensorOp(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_solve));
	  m_reg_solve_op->setMaxOrder(m_mg_maxorder);
	  m_reg_solve_op->setDomainBC(
        m_rincflo->get_diffuse_tensor_bc(Orientation::low),
		m_rincflo->get_diffuse_tensor_bc(Orientation::high));
	}
      
      if (m_rincflo->need_DivTau() || m_rincflo->useTensorCorrection()) 
        {
	  m_reg_apply_op.reset(
        new MLTensorOp(
            m_rincflo->Geom(0,finest_level),
            m_rincflo->boxArray(0,finest_level),
            m_rincflo->DistributionMap(0,finest_level),
            info_apply));
	  m_reg_apply_op->setMaxOrder(m_mg_maxorder);
	  m_reg_apply_op->setDomainBC(
        m_rincflo->get_diffuse_tensor_bc(Orientation::low),
        m_rincflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
}

void
DiffusionTensorOp::readParameters ()
{
    ParmParse pp("diffusion_tensor");
    m_verbose = m_rincflo->m_verbose;
    pp.query("verbose", m_verbose);
    pp.query("mg_verbose", m_mg_verbose);
    pp.query("mg_bottom_verbose", m_mg_bottom_verbose);
    pp.query("mg_max_iter", m_mg_max_iter);
    pp.query("mg_bottom_maxiter", m_mg_bottom_maxiter);
    pp.query("mg_max_fmg_iter", m_mg_max_fmg_iter);
    pp.query("mg_max_coarsening_level", m_mg_max_coarsening_level);
    pp.query("mg_maxorder", m_mg_maxorder);
    pp.query("mg_rtol", m_mg_rtol);
    pp.query("mg_atol", m_mg_atol);
    pp.query("bottom_solver_type", m_bottom_solver_type);
    pp.query("num_pre_smooth", m_num_pre_smooth);
    pp.query("num_post_smooth", m_num_post_smooth);
    pp.query("num_final_smooth", m_num_final_smooth);
    pp.query("num_bottom_smooth", m_num_bottom_smooth);
}

void
DiffusionTensorOp::DiffuseVelocity (
    Vector<MultiFab*> const& velocity,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt)
{
    //
    //      alpha a - beta div ( b grad )   <--->   rho - dt div ( mu grad )
    //
    // So the constants and variable coefficients are:
    //
    //      alpha: 1
    //      beta: dt
    //      a: rho
    //      b: mu

    if (m_verbose > 0) 
    {
        Print() << "Diffusing velocity components all together..." << std::endl;
    }

    const int finest_level = m_rincflo->finestLevel();

    #if (AMREX_USE_EB)
    if (m_eb_solve_op)
    {
        m_eb_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_eb_solve_op->setACoeffs(lev, *density[lev]);
            Array<MultiFab,SpaceDim> b = m_rincflo->average_velocity_eta_to_faces(lev, *eta[lev]);
	        m_eb_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(b),MLMG::Location::FaceCentroid);
	        m_eb_solve_op->setEBShearViscosity(lev, *eta[lev]);	    
        }
    }
    else
    #endif
    {
        m_reg_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_reg_solve_op->setACoeffs(lev, *density[lev]);
            Array<MultiFab,SpaceDim> b = m_rincflo->average_velocity_eta_to_faces(lev, *eta[lev]);
            m_reg_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
        }
    }

    Vector<MultiFab> rhs(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        rhs[lev].define(velocity[lev]->boxArray(), velocity[lev]->DistributionMap(), SpaceDim, 0);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(rhs[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& rhs_a = rhs[lev].array(mfi);
            Array4<Real const> const& vel_a = velocity[lev]->const_array(mfi);
            Array4<Real const> const& rho_a = density[lev]->const_array(mfi);
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                rhs_a(i,j,k,n) = rho_a(i,j,k) * vel_a(i,j,k,n);
            };
            ParallelFor(bx, SpaceDim, func);
        }

        #if (AMREX_USE_EB)
        if (m_eb_solve_op) 
        {
            m_eb_solve_op->setLevelBC(lev, velocity[lev]);
        } 
        else
        #endif
        {
            m_reg_solve_op->setLevelBC(lev, velocity[lev]);
        }
    }

    #if (AMREX_USE_EB)
    MLMG mlmg(m_eb_solve_op ? static_cast<MLLinOp&>(*m_eb_solve_op)
              :               static_cast<MLLinOp&>(*m_reg_solve_op));
    #else
    MLMG mlmg(*m_reg_solve_op);
    #endif

    // The default bottom solver is BiCG
    if (m_bottom_solver_type == "smoother")
    {
        mlmg.setBottomSolver(MLMG::BottomSolver::smoother);
    }

    // Maximum iterations for MultiGrid / ConjugateGradients
    mlmg.setMaxIter(m_mg_max_iter);
    mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
    mlmg.setBottomMaxIter(m_mg_bottom_maxiter);

    // Verbosity for MultiGrid / ConjugateGradients
    mlmg.setVerbose(m_mg_verbose);
    mlmg.setBottomVerbose(m_mg_bottom_verbose);

    mlmg.setPreSmooth(m_num_pre_smooth);
    mlmg.setPostSmooth(m_num_post_smooth);
    mlmg.setFinalSmooth(m_num_final_smooth);
    mlmg.setBottomSmooth(m_num_bottom_smooth);
    
    mlmg.solve(velocity, GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
}

void DiffusionTensorOp::ComputeDivTau (
    Vector<MultiFab*> const& a_divtau,
    Vector<MultiFab const*> const& a_velocity,
    Vector<MultiFab const*> const& a_density,
    Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("DiffusionTensorOp::ComputeDivTau");

    int finest_level = m_rincflo->finestLevel();

    Vector<MultiFab> velocity(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        velocity[lev].define(
            a_velocity[lev]->boxArray(),
            a_velocity[lev]->DistributionMap(),
            SpaceDim, 1, MFInfo(),
            a_velocity[lev]->Factory());
        MultiFab::Copy(velocity[lev], *a_velocity[lev], 0, 0, SpaceDim, 1);
    }

    #if (AMREX_USE_EB)
    if (m_eb_apply_op)
    {
        Vector<MultiFab> divtau_tmp(finest_level+1);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            divtau_tmp[lev].define(
                a_divtau[lev]->boxArray(),
                a_divtau[lev]->DistributionMap(),
                SpaceDim, 2, MFInfo(),
                a_divtau[lev]->Factory());
            divtau_tmp[lev].setVal(0.0);
        }

        // We want to return div (mu grad)) phi
        m_eb_apply_op->setScalars(0.0, -1.0);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_eb_apply_op->setACoeffs(lev, *a_density[lev]);
            Array<MultiFab,SpaceDim> b = m_rincflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);
	        m_eb_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b),MLMG::Location::FaceCentroid);
            m_eb_apply_op->setEBShearViscosity(lev, *a_eta[lev]);
            m_eb_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_eb_apply_op);
        mlmg.apply(GetVecOfPtrs(divtau_tmp), GetVecOfPtrs(velocity));

        for(int lev = 0; lev <= finest_level; lev++)
        {
	        amrex::single_level_redistribute(divtau_tmp[lev], *a_divtau[lev], 0, SpaceDim, m_rincflo->Geom(lev));
        }
    }
    else
    #endif
    {
        // We want to return div (mu grad)) phi
        m_reg_apply_op->setScalars(0.0, -1.0);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_reg_apply_op->setACoeffs(lev, *a_density[lev]);
            Array<MultiFab,SpaceDim> b = m_rincflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);
            m_reg_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
            m_reg_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_reg_apply_op);
        mlmg.apply(a_divtau, GetVecOfPtrs(velocity));
    }

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        for (MFIter mfi(*a_divtau[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& divtau_arr = a_divtau[lev]->array(mfi);
            Array4<Real const> const& rho_arr = a_density[lev]->const_array(mfi);
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real rhoinv = 1.0/rho_arr(i,j,k);
                AMREX_D_TERM(
                divtau_arr(i,j,k,0) *= rhoinv;,
                divtau_arr(i,j,k,1) *= rhoinv;,
                divtau_arr(i,j,k,2) *= rhoinv;)
            };
            ParallelFor(bx, func);
        }
    }
}
