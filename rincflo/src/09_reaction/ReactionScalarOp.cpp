#include <rincflo.H>
#include <ReactionScalarOp.H>

// *********************************************************************************************************************
ReactionScalarOp::ReactionScalarOp (Rincflo* a_rincflo, int a_ncomp)
    : m_rincflo(a_rincflo), m_ncomp(a_ncomp)
{
    #define FUNC_NAME "Rincflo::ReactionScalarOp::ReactionScalarOp"
    BL_PROFILE(FUNC_NAME);

    readParameters();

    // allocate operators/mlmg for each component
    m_eb_solve_op.resize(m_ncomp);
    m_mlmg.resize(m_ncomp);
    
    LPInfo info_solve;
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    if(m_bottom_solver_type == "hypre") 
        {info_solve.setMaxCoarseningLevel(0);}

    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);

    #if (AMREX_USE_EB)
    int finest_level = m_rincflo->finestLevel();
    Vector<EBFArrayBoxFactory const*> ebfact;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {ebfact.push_back(&(m_rincflo->EBFactory(lev)));}

    for(int n=0; n<m_ncomp; n++)
    {
        m_eb_solve_op[n].reset(
            new MLEBABecLap(
                m_rincflo->Geom(0,finest_level),
                m_rincflo->boxArray(0,finest_level),
                m_rincflo->DistributionMap(0,finest_level),
                info_solve, ebfact));
        m_eb_solve_op[n]->setMaxOrder(m_mg_maxorder);
        m_eb_solve_op[n]->setDomainBC(
            m_rincflo->get_react_scalar_bc(Orientation::low,n),
            m_rincflo->get_react_scalar_bc(Orientation::high,n));

        m_mlmg[n].reset(new MLMG(*m_eb_solve_op[n]));
    }

    // make this flag more understandable? 
    if (m_rincflo->need_Laplacian()) 
    {
        m_eb_scal_apply_op.reset(
            new MLEBABecLap(
                m_rincflo->Geom(0,finest_level),
                m_rincflo->boxArray(0,finest_level),
                m_rincflo->DistributionMap(0,finest_level),
                info_apply, ebfact));
        m_eb_scal_apply_op->setMaxOrder(m_mg_maxorder);
      // setDomainBC is called later
    }
    #endif
  
    setOptions();  
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void ReactionScalarOp::readParameters ()
{
    #define FUNC_NAME "Rincflo::ReactionScalarOp::readParameters"
    BL_PROFILE(FUNC_NAME);

    ParmParse pp("reaction");
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
    pp.query("bottom_rtol", m_bottom_rtol);

    // Report parameters if verbose
    if (m_verbose > 0)
    {
        Print() << "\n*** reaction parameters ***\n";
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
        Print() << format("bottom_rtol              : {:6.2e}\n", m_bottom_rtol);
    }
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void ReactionScalarOp::setOptions ()
{
    #define FUNC_NAME "Rincflo::ReactionScalarOp::setOptions"
    BL_PROFILE(FUNC_NAME);

    for(int n=0; n<m_ncomp; n++) 
    {
        m_mlmg[n]->setHypreOptionsNamespace("reaction.hypre");
        
        // The default bottom solver is BiCG
        if (m_bottom_solver_type == "smoother")
            {m_mlmg[n]->setBottomSolver(MLMG::BottomSolver::smoother);}
        else if (m_bottom_solver_type == "hypre")
            {m_mlmg[n]->setBottomSolver(MLMG::BottomSolver::hypre);}

        // Maximum iterations for MultiGrid / ConjugateGradients
        m_mlmg[n]->setMaxIter(m_mg_max_iter);
        m_mlmg[n]->setMaxFmgIter(m_mg_max_fmg_iter);
        m_mlmg[n]->setBottomMaxIter(m_mg_bottom_maxiter);
            
        // Verbosity for MultiGrid / Bottom solver
        m_mlmg[n]->setVerbose(m_mg_verbose);
        m_mlmg[n]->setBottomVerbose(m_mg_bottom_verbose);
        // Smoothing for MultiGrid
        m_mlmg[n]->setPreSmooth(m_num_pre_smooth);
        m_mlmg[n]->setPostSmooth(m_num_post_smooth);
        m_mlmg[n]->setFinalSmooth(m_num_final_smooth);
        m_mlmg[n]->setBottomSmooth(m_num_bottom_smooth);
        // Tolerance for MultiGrid
        m_mlmg[n]->setBottomTolerance(m_bottom_rtol);
    }
    #undef FUNC_NAME
}

//      alpha A - beta div ( B grad )   <--->   1 - dt div ( mu grad )  u = RHS
// So the constants and variable coefficients are:
//      alpha: (scalar const)             1
//      beta:  (scalar const)             dt
//      A:     (scalar field/const)       1
//      B:     (scalar field/const)       mu
//   RHS= conc+dt*conv+dt*source  (arising from time-stepping and Picard iteration) ; stored in conc

// *********************************************************************************************************************
void ReactionScalarOp::react_scalar (
    Vector<MultiFab*> const& conc,
    Vector<MultiFab*> const& rhs_arr,
    Vector<MultiFab const*> const& eta,
    Vector<MultiFab const*> const& acoeff,
    Real dt,
    int mode)
{
    #define FUNC_NAME "Rincflo::ReactionScalarOp::react_scalar"
    BL_PROFILE(FUNC_NAME);

    const int finest_level = m_rincflo->finestLevel();

    if(m_needs_update) // update operator(s)
    {
        for(int n=0; n<m_ncomp; n++) 
        {
            Vector<MultiFab> phi;
            if(mode==0) //e.g. lin 
                {m_eb_solve_op[n]->setScalars(1.0, dt);}
            else if (mode==1) //e.g. pic / just diffusion
                {m_eb_solve_op[n]->setScalars(1.0, dt);}

            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                if (mode==0) //e.g. lin
                    {m_eb_solve_op[n]->setACoeffs(lev, *acoeff[lev]);}
                else if (mode==1) //e.g. pic / just diffusion
                    {m_eb_solve_op[n]->setACoeffs(lev, 1.0);}

                if(m_rincflo->m_constant_Ds)
                    {m_eb_solve_op[n]->setBCoeffs(lev, m_rincflo->m_D_s[n]);}
                else
                {
                    Array<MultiFab,SpaceDim> b = m_rincflo->AverageScalarEtaToFaces(lev, n, *eta[lev]);      
                    m_eb_solve_op[n]->setBCoeffs(lev, GetArrOfConstPtrs(b));
                }
            }
        }
        // if fixed dt, nothing changes in between time steps
        m_needs_update = !(m_rincflo->m_fixed_dt > 0.0 && m_rincflo->m_constant_Ds);
    }

    // solve for each component
    for(int n=0; n<m_ncomp; n++) 
    { 
        if (m_verbose > 2) 
            {Print() << "reacting scalar " << n << std::endl;}

        // setup rhs and vector for solution
        Vector<MultiFab> phi;
        Vector<MultiFab> rhs;

        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            rhs.emplace_back(*rhs_arr[lev], amrex::make_alias, n, 1);
            phi.emplace_back(*conc[lev], amrex::make_alias, n, 1);
            m_eb_solve_op[n]->setLevelBC(lev, &phi[lev]);
        }

        // actual solve
        m_mlmg[n]->solve(GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    }
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// compute laplacian scalar
void ReactionScalarOp::ComputeLaplacian (
    Vector<MultiFab*> const& a_laps,
    Vector<MultiFab const*> const& a_scalar)
{
    #define FUNC_NAME "Rincflo::ReactionScalarOp::ComputeLaplacian"
    BL_PROFILE(FUNC_NAME);

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

        // We want to return div ( grad) phi
        m_eb_scal_apply_op->setScalars(0.0, -1.0);

        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            //  m_eb_scal_apply_op->setACoeffs(lev, 1.0); //not needed
            m_eb_scal_apply_op->setBCoeffs(lev, 1.0);
        }
      
        for (int comp = 0; comp < m_rincflo->m_nspec; ++comp) 
        {
            m_eb_scal_apply_op->setDomainBC(
                m_rincflo->get_react_scalar_bc(Orientation::low,comp),
                m_rincflo->get_react_scalar_bc(Orientation::high,comp));
            Vector<MultiFab> laps_comp;
            Vector<MultiFab> scalar_comp;
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                laps_comp.emplace_back(laps_tmp[lev],amrex::make_alias,comp,1);
                scalar_comp.emplace_back(scalar[lev],amrex::make_alias,comp,1);
                m_eb_scal_apply_op->setLevelBC(lev, &scalar_comp[lev]);
            }
            MLMG mlmg(*m_eb_scal_apply_op);
            mlmg.apply(GetVecOfPtrs(laps_comp), GetVecOfPtrs(scalar_comp));
        }

        for(int lev = 0; lev <= finest_level; lev++)
        {
            amrex::single_level_redistribute(
                laps_tmp[lev],
                *a_laps[lev], 0, m_rincflo->m_nspec,
                m_rincflo->Geom(lev));
        }
    }
    #endif
    #undef FUNC_NAME
}
