#include <rincflo.H>
#include <ReactionEliquidOp.H>

// *********************************************************************************************************************
ReactionEliquidOp::ReactionEliquidOp (Rincflo* a_rincflo)
  : m_rincflo(a_rincflo)
{
    readParameters();
  
    LPInfo info_solve;
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    if(m_bottom_solver_type == "hypre")
        {info_solve.setMaxCoarseningLevel(0);}

    int finest_level = m_rincflo->finestLevel();
    Vector<EBFArrayBoxFactory const*> ebfact;
    for (int lev = 0; lev <= finest_level; ++lev)
        {ebfact.push_back(&(m_rincflo->EBFactory(lev)));}
    m_eb_solve_op.reset(new MLEBABecLap(m_rincflo->Geom(0,finest_level),
        m_rincflo->boxArray(0,finest_level),
        m_rincflo->DistributionMap(0,finest_level),
        info_solve, ebfact));
    m_eb_solve_op->setMaxOrder(m_mg_maxorder);
    m_eb_solve_op->setDomainBC(m_rincflo->get_react_eliquid_bc(Orientation::low),
        m_rincflo->get_react_eliquid_bc(Orientation::high));

    m_mlmg.reset(new MLMG(*m_eb_solve_op));

    if (m_rincflo->do_Electromigration())
    {
        LPInfo info_apply;
        info_apply.setMaxCoarseningLevel(0);
      
        m_eb_apply_op.reset(new MLEBABecLap(m_rincflo->Geom(0,finest_level),
            m_rincflo->boxArray(0,finest_level),
            m_rincflo->DistributionMap(0,finest_level),
            info_apply, ebfact));
        m_eb_apply_op->setMaxOrder(m_mg_maxorder);
        m_eb_apply_op->setDomainBC(m_rincflo->get_react_eliquid_bc(Orientation::low),
            m_rincflo->get_react_eliquid_bc(Orientation::high));
    }
  
    setOptions();  
}

// *********************************************************************************************************************
void ReactionEliquidOp::readParameters ()
{
    #define FUNC_NAME "Rincflo::ReactionEliquidOp::readParameters"
    BL_PROFILE(FUNC_NAME);

    ParmParse pp("reaction_epot");
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
    pp.query("offset_solution", m_offset_solution);

    // Report parameters if verbose
    if (m_verbose > 0)
    {
        Print() << "\n*** reaction_epot parameters - shared liquid and solid ***\n";
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
        Print() << format("offset_solution          : {:d}\n", m_offset_solution);
    }
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void ReactionEliquidOp::setOptions ()
{
    m_mlmg->setHypreOptionsNamespace("reaction_epot.hypre");

    // The default bottom solver is BiCG
    if (m_bottom_solver_type == "smoother")
        {m_mlmg->setBottomSolver(MLMG::BottomSolver::smoother);}
    else if (m_bottom_solver_type == "hypre")
        {m_mlmg->setBottomSolver(MLMG::BottomSolver::hypre);}
  
    // Maximum iterations for MultiGrid / ConjugateGradients
    m_mlmg->setMaxIter(m_mg_max_iter);
    m_mlmg->setMaxFmgIter(m_mg_max_fmg_iter);
    m_mlmg->setBottomMaxIter(m_mg_bottom_maxiter);
            
    // Verbosity for MultiGrid / Bottom solver
    m_mlmg->setVerbose(m_mg_verbose);
    m_mlmg->setBottomVerbose(m_mg_bottom_verbose);

    m_mlmg->setPreSmooth(m_num_pre_smooth);
    m_mlmg->setPostSmooth(m_num_post_smooth);
    m_mlmg->setFinalSmooth(m_num_final_smooth);
    m_mlmg->setBottomSmooth(m_num_bottom_smooth);

    m_mlmg->setBottomTolerance(m_bottom_rtol);
}  

// *********************************************************************************************************************
// 
//      alpha A - beta div ( B grad )   <--->   0 - div ( kappa grad )  u = RHS
// So the constants and variable coefficients are:
//      alpha: (scalar const)             0
//      beta:  (scalar const)             1
//      A:     (scalar field/const)       0
//      B:     (scalar field/const)       kappa
//   RHS= source+div2conc ; stored in epotL
void ReactionEliquidOp::react_epot_liquid (
    Vector<MultiFab*> const& epotL, 
    Vector<MultiFab*> const& rhs_arr, 
    Vector<MultiFab const*> const& kappa)
{ 
    const int finest_level = m_rincflo->finestLevel();
 
    if (m_needs_update) 
    // update operator
    {
        m_eb_solve_op->setScalars(0.0, 1.0);      
        for (int lev = 0; lev <= finest_level; ++lev) 
        {    
            m_eb_solve_op->setACoeffs(lev, 0.0);

            if(m_rincflo->m_kappa_l > 0.0) 
                {m_eb_solve_op->setBCoeffs(lev, m_rincflo->m_kappa_l);}
            else 
            {
                Array<MultiFab,SpaceDim> b = m_rincflo->AverageEpotScalToFaces(lev, 0, *kappa[lev]);      
                m_eb_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCenter);
            }
        }
      
        // if kappa_l > 0 (i.e. constant) nothing changes in between time-steps...
        if (m_rincflo->m_kappa_l > 0.0)
            {m_needs_update = false;}
    }
  
    // check if we are updating bc in between time-steps due to membrane
    if(m_rincflo->need_MembreaneBcUpdate())  
    {
        m_eb_solve_op->setDomainBC(m_rincflo->get_react_eliquid_bc(Orientation::low), 
            m_rincflo->get_react_eliquid_bc(Orientation::high));
    }
  
    // prepare for solve
    if(m_verbose > 2)
        {Print() << "reacting epot liquid\n";}

    // vector for solution (and inhomog b.c., if any)
    Vector<MultiFab> phi;
    // setup rhs
    Vector<MultiFab> rhs;

    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        rhs.emplace_back(*rhs_arr[lev], amrex::make_alias, 0, 1);
        phi.emplace_back(*epotL[lev], amrex::make_alias, 0, 1);
        m_eb_solve_op->setLevelBC(lev, &phi[lev]);
    }

    // actual solve
    m_mlmg->solve(GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);    

    // possible ways to handle current-driven simulations with membrane...$
    // offset solution (if defined) [when problem has only Neumann BCs]
    if(m_offset_solution)
    {
        constexpr int dir {1};
        constexpr bool lo {true};
        constexpr int comp {0};
        constexpr bool local {false};
        constexpr bool phi_is_nodal {false};
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            // average value of low y-boundary (where membrane is)
            Real offset=m_rincflo->SumOverBoundaryCells(dir, lo, phi[lev], comp, phi_is_nodal, local); 
            AMREX_D_TERM(
            int Nx= m_rincflo->Geom(lev).Domain().length(0);,
            int Ny= m_rincflo->Geom(lev).Domain().length(1);,
            int Nz= m_rincflo->Geom(lev).Domain().length(2);)
            #if (AMREX_IS_2D)
            offset /= (Nx);
            #else
            offset /= (Nx*Nz);
            #endif
            phi[lev].plus(-offset, 0, 1, 0);
            EB_set_covered(phi[lev], 0.0);
            // exactly zeros the cells at low y-boundary (where membrane is)
            // m_rincflo->ZerosBoundaryCells(1,true,phi[lev],0); 
        }
    }
}

// *********************************************************************************************************************
// compute laplacian
void ReactionEliquidOp::compute_divcgpot (
    Vector<MultiFab*> const& a_divcgpot,
    Vector<MultiFab const*> const& a_epot,
    Vector<MultiFab const*> const& a_conc)
{
    BL_PROFILE("ReactionEliquidOp::compute_divcgpot");

    int finest_level = m_rincflo->finestLevel();
    Vector<MultiFab> epot(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        epot[lev].define(a_epot[lev]->boxArray(),
            a_epot[lev]->DistributionMap(),
            1, 1, MFInfo(),
            a_epot[lev]->Factory());
    }
  
    if (m_eb_apply_op)
    {
        Vector<MultiFab> divcgpot_tmp(finest_level+1);
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            AMREX_ASSERT(a_divcgpot[lev]->nComp() == a_conc[lev]->nComp());
            divcgpot_tmp[lev].define(a_divcgpot[lev]->boxArray(),
                 a_divcgpot[lev]->DistributionMap(),
                 m_rincflo->m_nspec, 2, MFInfo(),
                 a_divcgpot[lev]->Factory());
            divcgpot_tmp[lev].setVal(0.0);
        }

        // We want to return div c_j ( grad) phi
        m_eb_apply_op->setScalars(0.0, -1.0);

        for (int comp = 0; comp < m_rincflo->m_nspec; ++comp) 
        {
            Vector<MultiFab> divcgpot_comp;
    
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                MultiFab::Copy(epot[lev], *a_epot[lev], 0, 0, 1, 1);
                divcgpot_comp.emplace_back(divcgpot_tmp[lev],amrex::make_alias,comp,1);
            
                Array<MultiFab,SpaceDim> b = m_rincflo->AverageEpotScalToFaces(lev, comp, *a_conc[lev]);      
                m_eb_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCenter);
                m_eb_apply_op->setLevelBC(lev, &epot[lev]);
            }

            MLMG mlmg(*m_eb_apply_op);
            mlmg.apply(GetVecOfPtrs(divcgpot_comp), GetVecOfPtrs(epot));
        }
      
        for(int lev = 0; lev <= finest_level; lev++)
        {
            amrex::single_level_redistribute(divcgpot_tmp[lev],
                *a_divcgpot[lev], 0, m_rincflo->m_nspec, m_rincflo->Geom(lev));
            EB_set_covered(*a_divcgpot[lev], 0.0);
        }
    }
}
