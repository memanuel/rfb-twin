#include <rincflo.H>

DiffusionScalarOp::DiffusionScalarOp (Rincflo* a_rincflo)
    : m_rincflo(a_rincflo)
{
    readParameters();

    LPInfo info_solve;
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    if(m_bottom_solver_type == "hypre") 
    {
      info_solve.setMaxCoarseningLevel(0);
    }
    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);
    #if (AMREX_USE_EB)
    int finest_level = m_rincflo->finestLevel();
    // if (!m_rincflo->EBFactory(0).isAllRegular())
    if (m_rincflo->has_eb())
    {
        Vector<EBFArrayBoxFactory const*> ebfact;
        for (int lev = 0; lev <= finest_level; ++lev) {
            ebfact.push_back(&(m_rincflo->EBFactory(lev)));
        }

	    if(m_rincflo->do_AdvectConc())
	    {
	        m_eb_scal_solve_op.reset(
                new MLEBABecLap(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_solve, ebfact));
            m_eb_scal_solve_op->setMaxOrder(m_mg_maxorder);
            m_eb_scal_solve_op->setDomainBC(
                m_rincflo->get_DiffuseScalar_bc(Orientation::low ),
                m_rincflo->get_DiffuseScalar_bc(Orientation::high));
	    }
	
        if (!m_rincflo->useTensorSolve())
	    {
            m_eb_vel_solve_op.reset(
                new MLEBABecLap(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_solve, ebfact));
            m_eb_vel_solve_op->setMaxOrder(m_mg_maxorder);
            //            m_eb_vel_solve_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,0),
            //                                           m_rincflo->get_DiffuseVelocity_bc(Orientation::high,0));
            // We don't call setDomainBC here because we will need to call it separately for each component
        }

        if (m_rincflo->need_DivTau()) 
        {
            m_eb_scal_apply_op.reset(
                new MLEBABecLap(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_apply, ebfact));
            m_eb_scal_apply_op->setMaxOrder(m_mg_maxorder);
            m_eb_scal_apply_op->setDomainBC(
                m_rincflo->get_DiffuseScalar_bc(Orientation::low),
                m_rincflo->get_DiffuseScalar_bc(Orientation::high));
        }
	
        if ( (m_rincflo->need_DivTau() && !m_rincflo->useTensorSolve()) || m_rincflo->useTensorCorrection() )
	    {
            m_eb_vel_apply_op.reset(
                new MLEBABecLap(
                    m_rincflo->Geom(0,finest_level),
                    m_rincflo->boxArray(0,finest_level),
                    m_rincflo->DistributionMap(0,finest_level),
                    info_apply, ebfact));
            m_eb_vel_apply_op->setMaxOrder(m_mg_maxorder);
            //            m_eb_vel_apply_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,0),
            //                                           m_rincflo->get_DiffuseVelocity_bc(Orientation::high,0));
            // We don't call setDomainBC here because we will need to call it separately for each component
	    }
    }
    else
    #endif
    {
      if(m_rincflo->do_AdvectConc())
	{
	    m_reg_scal_solve_op.reset(
            new MLABecLaplacian(
                m_rincflo->Geom(0,m_rincflo->finestLevel()),
                m_rincflo->boxArray(0,m_rincflo->finestLevel()),
                m_rincflo->DistributionMap(0,m_rincflo->finestLevel()),
                info_solve));
        m_reg_scal_solve_op->setMaxOrder(m_mg_maxorder);
        m_reg_scal_solve_op->setDomainBC(
            m_rincflo->get_DiffuseScalar_bc(Orientation::low),
                        m_rincflo->get_DiffuseScalar_bc(Orientation::high));
	}
      
    if (!m_rincflo->useTensorSolve())
    {
        m_reg_vel_solve_op.reset(
            new MLABecLaplacian(
                m_rincflo->Geom(0,m_rincflo->finestLevel()),
                m_rincflo->boxArray(0,m_rincflo->finestLevel()),
                m_rincflo->DistributionMap(0,m_rincflo->finestLevel()),
                info_solve));
        m_reg_vel_solve_op->setMaxOrder(m_mg_maxorder);
        //	  m_reg_vel_solve_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,0),
        //				  m_rincflo->get_DiffuseVelocity_bc(Orientation::high,0));
        // We don't call setDomainBC here because we will need to call it separately for each component
    }
      
    if (m_rincflo->need_DivTau()) 
    {
        m_reg_scal_apply_op.reset(
            new MLABecLaplacian(
                m_rincflo->Geom(0,m_rincflo->finestLevel()),
                m_rincflo->boxArray(0,m_rincflo->finestLevel()),
                m_rincflo->DistributionMap(0,m_rincflo->finestLevel()),
                info_apply));
        m_reg_scal_apply_op->setMaxOrder(m_mg_maxorder);
        m_reg_scal_apply_op->setDomainBC(m_rincflo->get_DiffuseScalar_bc(Orientation::low),
                        m_rincflo->get_DiffuseScalar_bc(Orientation::high));
    }

    if ( (m_rincflo->need_DivTau() && !m_rincflo->useTensorSolve()) || m_rincflo->useTensorCorrection() )
    {
        m_reg_vel_apply_op.reset(
            new MLABecLaplacian(
                m_rincflo->Geom(0,m_rincflo->finestLevel()),
                m_rincflo->boxArray(0,m_rincflo->finestLevel()),
                m_rincflo->DistributionMap(0,m_rincflo->finestLevel()),
                info_apply));
        m_reg_vel_apply_op->setMaxOrder(m_mg_maxorder);
        //	  m_reg_vel_apply_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,0),
        //				  m_rincflo->get_DiffuseVelocity_bc(Orientation::high,0));
        // We don't call setDomainBC here because we will need to call it separately for each component
        }
    }
}

void
DiffusionScalarOp::readParameters ()
{
    #define FUNC_NAME "Rincflo::DiffusionScalarOp::readParameters"
    BL_PROFILE(FUNC_NAME);

    ParmParse pp("diffusion");
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

    // Report parameters if verbose
    if (m_verbose > 0)
    {
        Print() << "\n*** diffusion parameters: ***\n";
        Print() << format("verbose                  : {:d}\n", m_verbose);
        Print() << format("mg_verbose               : {:d}\n", m_mg_verbose);
        Print() << format("mg_bottom_verbose        : {:d}\n", m_mg_bottom_verbose);
        Print() << format("mg_max_iter              : {:d}\n", m_mg_max_iter);
        Print() << format("mg_bottom_maxiter        : {:d}\n", m_mg_bottom_maxiter);
        Print() << format("mg_max_fmg_iter          : {:d}\n", m_mg_max_fmg_iter);
        Print() << format("mg_max_coarsening_level  : {:d}\n", m_mg_max_coarsening_level);
        Print() << format("mg_maxorder              : {:d}\n", m_mg_maxorder);
        Print() << format("mg_rtol                  : {:6.2e}\n", m_mg_rtol);
        Print() << format("mg_atol                  : {:6.2e}\n", m_mg_atol);
        Print() << format("bottom_solver_type       : {:s}\n", m_bottom_solver_type);
        Print() << format("num_pre_smooth           : {:d}\n", m_num_pre_smooth);
        Print() << format("num_post_smooth          : {:d}\n", m_num_post_smooth);
        Print() << format("num_final_smooth         : {:d}\n", m_num_final_smooth);
        Print() << format("num_bottom_smooth        : {:d}\n", m_num_bottom_smooth);
        //Print() << format("bottom_rtol          : {:6.2e}\n", m_bottom_rtol);
    }
    #undef FUNC_NAME
}

void
DiffusionScalarOp::DiffuseScalar (
    Vector<MultiFab*> const& conc,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt)
{
    //
    //      alpha a - beta div ( b grad )   <--->   1 - dt div ( D grad )
    // So the constants and variable coefficients are:
    //
    //      alpha: 1
    //      beta: dt
    //      a: 1
    //      b: D

    if (m_verbose > 0) 
    {
        Print() << "Diffusing scalars one at a time ..." << std::endl;
    }

    const int finest_level = m_rincflo->finestLevel();

    Vector<MultiFab> rhs(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        rhs[lev].define(conc[lev]->boxArray(), conc[lev]->DistributionMap(), 1, 0);
    }

    #if (AMREX_USE_EB)
    if (m_eb_scal_solve_op)
    {
        m_eb_scal_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
	        m_eb_scal_solve_op->setACoeffs(lev, *density[lev]);
	    }
    }
    else
    #endif
    {
        m_reg_scal_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
	        m_reg_scal_solve_op->setACoeffs(lev, *density[lev]);
	    }
    }

    for (int comp = 0; comp < conc[0]->nComp(); ++comp)
    {
        #if (AMREX_USE_EB)
        if (m_eb_scal_solve_op)
        {
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, comp, *eta[lev]);
                m_eb_scal_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);
            }
        }
        else
        #endif
        {
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, comp, *eta[lev]);
                m_reg_scal_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b));
            }
        }

        Vector<MultiFab> phi;
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            phi.emplace_back(*conc[lev], amrex::make_alias, comp, 1);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(rhs[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) 
            {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& rhs_a = rhs[lev].array(mfi);
                Array4<Real const> const& conc_a = conc[lev]->const_array(mfi,comp);
                Array4<Real const> const& rho_a = density[lev]->const_array(mfi);
                auto func_bx = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                rhs_a(i,j,k) = rho_a(i,j,k) * conc_a(i,j,k);
                };
                ParallelFor(bx, func_bx);
            }

            #if (AMREX_USE_EB)
            if (m_eb_scal_solve_op) 
            {
                m_eb_scal_solve_op->setLevelBC(lev, &phi[lev]);
            } 
            else
            #endif
            {
                m_reg_scal_solve_op->setLevelBC(lev, &phi[lev]);
            }
        }

        #if (AMREX_USE_EB)
        MLMG mlmg(m_eb_scal_solve_op ? static_cast<MLLinOp&>(*m_eb_scal_solve_op) : static_cast<MLLinOp&>(*m_reg_scal_solve_op));
        #else
        MLMG mlmg(*m_reg_scal_solve_op);
        #endif
	    mlmg.setHypreOptionsNamespace("diffusion_scalar.hypre");
	
        // The default bottom solver is BiCG
        if (m_bottom_solver_type == "smoother")
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::smoother);
        }
        else if (m_bottom_solver_type == "hypre")
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        }

        // Maximum iterations for MultiGrid / ConjugateGradients
        mlmg.setMaxIter(m_mg_max_iter);
        mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
        mlmg.setBottomMaxIter(m_mg_bottom_maxiter);
        
        // Verbosity for MultiGrid / bottomsolver
        mlmg.setVerbose(m_mg_verbose);
        mlmg.setBottomVerbose(m_mg_bottom_verbose);

	    mlmg.setPreSmooth(m_num_pre_smooth);
        mlmg.setPostSmooth(m_num_post_smooth);
	    mlmg.setFinalSmooth(m_num_final_smooth);
        mlmg.setBottomSmooth(m_num_bottom_smooth);
	
        mlmg.solve(GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    }
}

void
DiffusionScalarOp::diffuse_vel_components (Vector<MultiFab*> const& vel,
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

    if (m_verbose > 0) {
        Print() << "Diffusing velocity components one at a time ..." << std::endl;
    }

    AMREX_ASSERT(vel[0]->nComp() == SpaceDim);

    const int finest_level = m_rincflo->finestLevel();

    Vector<MultiFab> rhs(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        rhs[lev].define(vel[lev]->boxArray(), vel[lev]->DistributionMap(), 1, 0);
    }

    for (int comp = 0; comp < vel[0]->nComp(); ++comp)
    {
        int eta_comp = 0;

        #if (AMREX_USE_EB)
        if (m_eb_vel_solve_op)
        {
            // Because the different components may have different boundary conditions, we need to
            // reset these for each solve
            m_eb_vel_solve_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,comp),
                                           m_rincflo->get_DiffuseVelocity_bc(Orientation::high,comp));
            m_eb_vel_solve_op->setScalars(1.0, dt);
            for (int lev = 0; lev <= finest_level; ++lev) {
                m_eb_vel_solve_op->setACoeffs(lev, *density[lev]);
                m_eb_vel_solve_op->setEBHomogDirichlet(lev, *eta[lev]);
            }
    
            for (int lev = 0; lev <= finest_level; ++lev) {
                Array<MultiFab,SpaceDim> 
                    b = m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *eta[lev]);

		m_eb_vel_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);
            }
        }
        else
        #endif
        {
            // Because the different components may have different boundary conditions, we need to
            // reset these for each solve
            m_reg_vel_solve_op->setDomainBC(m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,comp),
                                            m_rincflo->get_DiffuseVelocity_bc(Orientation::high,comp));
    	    m_reg_vel_solve_op->setScalars(1.0, dt);
            for (int lev = 0; lev <= finest_level; ++lev) {
                m_reg_vel_solve_op->setACoeffs(lev, *density[lev]);
            }
    

            for (int lev = 0; lev <= finest_level; ++lev) {
                Array<MultiFab,SpaceDim> 
                    b = m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *eta[lev]);
                m_reg_vel_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b));
            }
        }

        Vector<MultiFab> phi;
        for (int lev = 0; lev <= finest_level; ++lev) {
	    vel[lev]->FillBoundary(m_rincflo->Geom(lev).periodicity());
            phi.emplace_back(*vel[lev], amrex::make_alias, comp, 1);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(rhs[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) 
            {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& rhs_a = rhs[lev].array(mfi);
                Array4<Real const> const& vel_a = vel[lev]->const_array(mfi,comp);
                Array4<Real const> const& rho_a = density[lev]->const_array(mfi);
                auto func_bx = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    rhs_a(i,j,k) = rho_a(i,j,k) * vel_a(i,j,k);
                };
                ParallelFor(bx, func_bx);
            }

            #if (AMREX_USE_EB)
            if (m_eb_vel_solve_op) 
            {
                m_eb_vel_solve_op->setLevelBC(lev, &phi[lev]);

            } 
            else
            #endif
            {
                m_reg_vel_solve_op->setLevelBC(lev, &phi[lev]);
            }
        }

        #if (AMREX_USE_EB)
        MLMG mlmg(m_eb_vel_solve_op ? static_cast<MLLinOp&>(*m_eb_vel_solve_op) : static_cast<MLLinOp&>(*m_reg_vel_solve_op));
        #else
        MLMG mlmg(*m_reg_vel_solve_op);
        #endif
	    mlmg.setHypreOptionsNamespace("diffusion_scalar.hypre");
	
        // The default bottom solver is BiCG
        if (m_bottom_solver_type == "smoother")
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::smoother);
        }
        else if (m_bottom_solver_type == "hypre")
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        }

        // Maximum iterations for MultiGrid / ConjugateGradients
        mlmg.setMaxIter(m_mg_max_iter);
        mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
        mlmg.setBottomMaxIter(m_mg_bottom_maxiter);
        
        // Verbosity for MultiGrid / Bottom solver
        mlmg.setVerbose(m_mg_verbose);
        mlmg.setBottomVerbose(m_mg_bottom_verbose);

        mlmg.setPreSmooth(m_num_pre_smooth);
        mlmg.setPostSmooth(m_num_post_smooth);
	    mlmg.setFinalSmooth(m_num_final_smooth);
        mlmg.setBottomSmooth(m_num_bottom_smooth);
	
        mlmg.solve(GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    }
}


void DiffusionScalarOp::ComputeLaplacian (
    Vector<MultiFab*> const& a_laps,
    Vector<MultiFab const*> const& a_scalar,
    Vector<MultiFab const*> const& a_density,
    Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("DiffusionScalarOp::ComputeLaplacian");

    int finest_level = m_rincflo->finestLevel();

    Vector<MultiFab> scalar(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        AMREX_ASSERT(a_scalar[lev]->nComp() == a_laps[lev]->nComp());
        scalar[lev].define(
            a_scalar[lev]->boxArray(),
            a_scalar[lev]->DistributionMap(),
            m_rincflo->m_nspec, 1, MFInfo(),
            a_scalar[lev]->Factory());
        MultiFab::Copy(scalar[lev], *a_scalar[lev], 0, 0, m_rincflo->m_nspec, 1);
    }

#if (AMREX_USE_EB)
    if (m_eb_scal_apply_op)
    {
        Vector<MultiFab> laps_tmp(finest_level+1);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            laps_tmp[lev].define(
                a_laps[lev]->boxArray(),
                a_laps[lev]->DistributionMap(),
                m_rincflo->m_nspec, 2, MFInfo(),
                a_laps[lev]->Factory());
            laps_tmp[lev].setVal(0.0);
        }

        // We want to return div (mu grad)) phi
        m_eb_scal_apply_op->setScalars(0.0, -1.0);

        // For when we use the stencil for centroid values
        // m_eb_scal_apply_op->setPhiOnCentroid();  

        // This should have no effect since the first scalar is 0
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_eb_scal_apply_op->setACoeffs(lev, *a_density[lev]);
        }

        for (int comp = 0; comp < m_rincflo->m_nspec; ++comp) 
        {
            int eta_comp = comp;

            Vector<MultiFab> laps_comp;
            Vector<MultiFab> scalar_comp;
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                laps_comp.emplace_back(laps_tmp[lev],amrex::make_alias,comp,1);
                scalar_comp.emplace_back(scalar[lev],amrex::make_alias,comp,1);

                Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *a_eta[lev]);

                m_eb_scal_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);
                m_eb_scal_apply_op->setLevelBC(lev, &scalar_comp[lev]);
            }

            MLMG mlmg(*m_eb_scal_apply_op);
            mlmg.apply(GetVecOfPtrs(laps_comp), GetVecOfPtrs(scalar_comp));
        }

        for(int lev = 0; lev <= finest_level; lev++)
        {
            amrex::single_level_redistribute(laps_tmp[lev], *a_laps[lev], 0, m_rincflo->m_nspec, m_rincflo->Geom(lev));
        }
    }
    else
    #endif
    {
        // We want to return div (mu grad)) phi
        m_reg_scal_apply_op->setScalars(0.0, -1.0);

        // This should have no effect since the first scalar is 0
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_reg_scal_apply_op->setACoeffs(lev, *a_density[lev]);
        }

        for (int comp = 0; comp < m_rincflo->m_nspec; ++comp) 
        {
            int eta_comp = comp;

            Vector<MultiFab> laps_comp;
            Vector<MultiFab> scalar_comp;
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                laps_comp.emplace_back(*a_laps[lev],amrex::make_alias,comp,1);
                scalar_comp.emplace_back(scalar[lev],amrex::make_alias,comp,1);
                Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *a_eta[lev]);

                m_reg_scal_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(b));
                m_reg_scal_apply_op->setLevelBC(lev, &scalar_comp[lev]);
            }

            MLMG mlmg(*m_reg_scal_apply_op);
            mlmg.apply(GetVecOfPtrs(laps_comp), GetVecOfPtrs(scalar_comp));
        }
    }
}

void DiffusionScalarOp::ComputeDivTau (
    Vector<MultiFab*> const& a_divtau,
    Vector<MultiFab const*> const& a_vel,
    Vector<MultiFab const*> const& a_density,
    Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("DiffusionScalarOp::ComputeDivTau");

    int finest_level = m_rincflo->finestLevel();

    AMREX_ASSERT(a_vel[0]->nComp()    == SpaceDim);
    AMREX_ASSERT(a_divtau[0]->nComp() == SpaceDim);

    Vector<MultiFab> vel(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        vel[lev].define(
            a_vel[lev]->boxArray(),
            a_vel[lev]->DistributionMap(),
            a_vel[lev]->nComp(), 1, MFInfo(),
            a_vel[lev]->Factory());
        MultiFab::Copy(vel[lev], *a_vel[lev], 0, 0, a_vel[lev]->nComp(), 1);
    }

#if (AMREX_USE_EB)
    if (m_eb_vel_apply_op)
    {
        Vector<MultiFab> divtau_tmp(finest_level+1);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            divtau_tmp[lev].define(
                a_divtau[lev]->boxArray(),
                a_divtau[lev]->DistributionMap(),
                a_divtau[lev]->nComp(), 2, MFInfo(),
                a_divtau[lev]->Factory());
            divtau_tmp[lev].setVal(0.0);
        }

        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_eb_vel_apply_op->setEBHomogDirichlet(lev, *a_eta[lev]);
        }

        // We want to return div (mu grad)) phi
        m_eb_vel_apply_op->setScalars(0.0, -1.0);

        // For when we use the stencil for centroid values
        // m_eb_vel_apply_op->setPhiOnCentroid();  

        // This should have no effect since the first scalar is 0
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_eb_vel_apply_op->setACoeffs(lev, *a_density[lev]);
        }

        int eta_comp = 0;

        for (int comp = 0; comp < a_divtau[0]->nComp(); ++comp) 
        {
            Vector<MultiFab> divtau_single;
            Vector<MultiFab>    vel_single;

            // Because the different components may have different boundary conditions, we need to
            // reset these for each solve
            m_eb_vel_apply_op->setDomainBC(
                m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,comp),
                m_rincflo->get_DiffuseVelocity_bc(Orientation::high,comp));


            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                divtau_single.emplace_back(divtau_tmp[lev],amrex::make_alias,comp,1);
                   vel_single.emplace_back(       vel[lev],amrex::make_alias,comp,1);
                m_eb_vel_apply_op->setLevelBC(lev, &vel_single[lev]);

                Array<MultiFab,SpaceDim> b =
                    m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *a_eta[lev]);
                m_eb_vel_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);
            }

            MLMG mlmg(*m_eb_vel_apply_op);

            mlmg.apply(GetVecOfPtrs(divtau_single), GetVecOfPtrs(vel_single));
        }

        for(int lev = 0; lev <= finest_level; lev++)
        {
            amrex::single_level_redistribute(
                divtau_tmp[lev], *a_divtau[lev], 0, a_divtau[lev]->nComp(), m_rincflo->Geom(lev));
        }
    }
    else
    #endif
    {
        // We want to return div (mu grad)) phi
        m_reg_vel_apply_op->setScalars(0.0, -1.0);

        // This should have no effect since the first scalar is 0
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            m_reg_vel_apply_op->setACoeffs(lev, *a_density[lev]);
        }

        int eta_comp = 0;
        Vector<MultiFab> divtau_single;
        Vector<MultiFab>    vel_single;

        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, eta_comp, *a_eta[lev]);
            m_reg_vel_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(b));
        }

        for (int comp = 0; comp < a_divtau[0]->nComp(); ++comp) 
        {
	    // Because the different components may have different boundary conditions, we need to
            // reset these for each solve
            m_reg_vel_apply_op->setDomainBC(
                m_rincflo->get_DiffuseVelocity_bc(Orientation::low ,comp),
				m_rincflo->get_DiffuseVelocity_bc(Orientation::high,comp));
	    
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                divtau_single.emplace_back(*a_divtau[lev],amrex::make_alias,comp,1);
                   vel_single.emplace_back(      vel[lev],amrex::make_alias,comp,1);

                m_reg_vel_apply_op->setLevelBC(lev, &vel_single[lev]);
            }

            MLMG mlmg(*m_reg_vel_apply_op);
            mlmg.apply(GetVecOfPtrs(divtau_single), GetVecOfPtrs(vel_single));
        }
    }
}
