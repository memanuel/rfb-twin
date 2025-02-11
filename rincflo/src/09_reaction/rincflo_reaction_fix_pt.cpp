#include <rincflo.H>

// *********************************************************************************************************************
// Fixed point iteration for reaction time stepping.
// Includes inner steps with Picard or Newton step
// *********************************************************************************************************************

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// solve reaction part                                                    //
// -----------------------------------------------------------------------//
bool Rincflo::ReactStep_pic (Real dt, int* tot_pic_it)
{
    #define FUNC_NAME "Rincflo::ReactStep_pic"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // INIT
    int max_it=m_fixpoint_max_iter;
    int pic_it=0; 
    // relax if omega<1.0
    bool relax_pic = m_fixpoint_relax; 
    // relaxation parameter
    Real omega_pic = m_fixpoint_omega; 
    bool do_iteration = true;
    bool converged = false;

    // allocate space for conductivity "kappaS" (if used need to be defined here to be visible)
    Vector<MultiFab> kappaS;
    const int nghost_kappa = 1;
    const int nghost_flxrhs = 0;
    if(m_reaction_model == ReactionModel::ButlerVolmer &&  m_solve_epotS)      
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }

    // Main iteration loop
    while ( do_iteration ) 
    {
        pic_it++;

        if(m_verbose > 3) 
            {Print() << "start Picard iteration: " << pic_it << std::endl ;}

        //Store quantities
        CopyNewToOldPicard_react();
        for(int lev = 0; lev <= finest_level; lev++) 
            {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc_op, nghost_state());}
        if(m_reaction_model == ReactionModel::ButlerVolmer) 
        {      
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                fillpatch_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL_op, nghost_state());
                if(m_solve_epotS)
                    {fillpatch_epotS(lev, m_t_new[lev], m_leveldata[lev]->epotS_op, nghost_state());}
                }
            }
        
            // Calculate source term
            CalcSourceTermAll();

            //(Calculate conductivity)
            if(m_reaction_model == ReactionModel::ButlerVolmer && m_kappa_l < 0.0)
                {CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);}
    
        //(Calculate conc flux)
        if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL) 
        {
            get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const()); //get div^2 conc
            CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
        }

        //Construct RHS and solve diff-react equations

        //  -> for concentration
        UpdateReactConc_pic(dt);
        
        //( -> for epot liquid )
        if(m_reaction_model == ReactionModel::ButlerVolmer) 
        {
            if(m_calc_flux_epotL)
                {Update_DiffReactEpotL(get_kappa_const(),get_flxrhs_const(),get_source_const());}
            else
                {Update_DiffReactEpotL(get_kappa_const(),get_source_const());}
        }

        //( -> for epot solid  )
        if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
            {UpdateDiffReactEpotS(GetVecOfConstPtrs(kappaS),get_source_const());}
        
        //(Relaxation)
        if(relax_pic) 
            {RelaxationFixPoint(omega_pic);}
    
        //Check convergence
        converged=CheckConvergenceFixPoint(m_verbose);
        
        if ( converged || pic_it > max_it )
            {do_iteration=false;}

      }//end of Picard loop

    if(m_verbose>1) 
        {Print() << "Picard iterations: " << pic_it << std::endl ;}

    if(converged==false) 
        {bool trash=CheckConvergenceFixPoint(100);}

    *tot_pic_it=pic_it;
  
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return converged;
}

// *********************************************************************************************************************
bool Rincflo::ReactStep_nwt (Real dt) 
{
    Abort("Newton method not implemented yet"); //$
    return false;
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// solve diffusion and reaction part                                      //
// -----------------------------------------------------------------------//
bool Rincflo::DiffReactStep_pic (Real dt, int* tot_it)
{
    #define FUNC_NAME "Rincflo::DiffReactStep_pic"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Initialize
    int max_it = m_fixpoint_max_iter;
    int pic_it = 0;
    // relax if omega<1.0
    bool relax_pic = m_fixpoint_relax; 
    Real omega_pic = m_fixpoint_omega; 
    bool do_iteration = true;
    bool converged = false;

    // allocate space for conductivity "kappaS" (if used need to be defined here to be visible)
    Vector<MultiFab> kappaS;
    const int nghost_kappa = 1;
    const int nghost_flxrhs = 0;
    if(m_reaction_model == ReactionModel::ButlerVolmer &&  m_solve_epotS)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }

    // Main loop
    while (do_iteration)
    {
        pic_it++;
        if(m_verbose > 3) 
            {Print() << format("start Picard iteration: {:d} \n", pic_it);}

        // Store quantities
        CopyNewToOldPicard_react();
        for(int lev = 0; lev <= finest_level; ++lev) 
            {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc_op, nghost_state());}
        if(m_reaction_model == ReactionModel::ButlerVolmer) 
        {
            for (int lev = 0; lev <= finest_level; ++lev) 
            {
                fillpatch_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL_op, nghost_state());
                if(m_solve_epotS)
                    {fillpatch_epotS(lev, m_t_new[lev], m_leveldata[lev]->epotS_op, nghost_state());}
            }
        }
    
        // Calculate source term
        CalcSourceTermAll();

        // (Calculate conductivity)
        if(m_reaction_model == ReactionModel::ButlerVolmer && m_kappa_l < 0.0)
            {CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);}
        
        // (Calculate conc flux)
        if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL)
        {
            //get div^2 conc
            get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const()); 
            CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
        }
        
        // Construct RHS and solve diff-react equations

        //  -> for concentration
        UpdateDiffReactConc_pic(dt);

        //( -> for epot liquid )
        if(m_reaction_model == ReactionModel::ButlerVolmer)
        {
            if(m_calc_flux_epotL)
                {Update_DiffReactEpotL(get_kappa_const(),get_flxrhs_const(),get_source_const());}
            else
                {Update_DiffReactEpotL(get_kappa_const(),get_source_const());}
        }

        //( -> for epot solid  )
        if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
            {UpdateDiffReactEpotS(GetVecOfConstPtrs(kappaS),get_source_const());}
        
        // (Relaxation)
        if(relax_pic) 
            {RelaxationFixPoint(omega_pic);}
        
        // Check convergence
        converged = CheckConvergenceFixPoint(m_verbose);
        
        if (converged || pic_it > max_it)
            {do_iteration=false;}
       
    } //end of Picard loop

    if(m_verbose>1) 
        {Print() << format("Picard iterations: {:d}.\n", pic_it);}

    if(converged==false) 
    {
        // print stuff to understand what did not converge 
        bool trash=CheckConvergenceFixPoint(5);
    } 

    *tot_it=pic_it;
  
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return converged;
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// solve diffusion and reaction part                                      //
// -----------------------------------------------------------------------//
bool Rincflo::DiffReactStep_nwt (Real dt)
{
    Abort("Newton method not implemented yet"); //$
    return false;
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// Relaxation update for quantities of fixed-point iteration              //
// -----------------------------------------------------------------------//
void Rincflo::RelaxationFixPoint (Real omega) 
{ 
    #define FUNC_NAME "Rincflo::RelaxationFixPoint"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Update variables: u = omega*u + (1-omega)*u_old 
    // dst = a*x+b*y  lincomb(dst,a,x,xcomp,b,y,ycomp,dstcomp,numcomp,nghost)

    if (m_verbose > 3)
        {Print() << "Relax with omega= " << omega << "\n" ;}
    
    for(int lev = 0; lev <= finest_level; lev++)
    {
        MultiFab& conc_new = m_leveldata[lev]->conc;
        MultiFab const& conc_old = m_leveldata[lev]->conc_op;     
        MultiFab::LinComb(conc_new,omega,conc_new,0,1.0-omega,conc_old,0,0,m_nspec,0);
      
        if(m_reaction_model == ReactionModel::ButlerVolmer) 
        {
            MultiFab& epotL_new = m_leveldata[lev]->epotL;
            MultiFab const& epotL_old = m_leveldata[lev]->epotL_op;      
            MultiFab::LinComb(epotL_new,omega,epotL_new,0,1.0-omega,epotL_old,0,0,1,0);

            if(m_solve_epotS) 
            {
                MultiFab& epotS_new = m_leveldata[lev]->epotS;
                MultiFab const& epotS_old = m_leveldata[lev]->epotS_op;          
                MultiFab::LinComb(epotS_new,omega,epotS_new,0,1.0-omega,epotS_old,0,0,1,0);
            }
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// check convergence for fixed-point iteration                            //
// -----------------------------------------------------------------------//
bool Rincflo::CheckConvergenceFixPoint(int verbose) 
{
    #define FUNC_NAME "Rincflo::CheckConvergenceFixPoint"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Alias the tolerance for concentration and epot in Picard iterations
    Real tol_conc_pic = m_fixpoint_conc_tol; 
    Real tol_epot_pic = m_fixpoint_epot_tol;

    // Are we running the ButlerVolmer model?
    bool is_BV = (m_reaction_model == ReactionModel::ButlerVolmer);

    // Apply fixpoint relaxation factor, fixpoint_omega, if using fixpoint relaxation (currently NOT used)
    if(m_fixpoint_relax) 
    {
        tol_conc_pic *= m_fixpoint_omega;
        tol_epot_pic *= m_fixpoint_omega;
    }

    // Flags - have we converged on each level?
    bool is_converged_conc[finest_level + 1];
    bool is_converged_epot[finest_level + 1];

    // Initialize change in concentration and epot with a large placeholder
    Real change_conc = 1.0E10;
    Real change_ep = is_BV ? 0.0 : 1.0E10;
 
    for(int lev = 0; lev <= finest_level; lev++)
    {    
        MultiFab& conc_new = m_leveldata[lev]->conc;
        MultiFab const& conc_old = m_leveldata[lev]->conc_op;

        MultiFab diff_tra;
        diff_tra.define(conc_new.boxArray(), conc_new.DistributionMap(), m_nspec, 0);
        MultiFab::Copy(diff_tra, conc_new, 0, 0, m_nspec, 0);
        MultiFab::Subtract(diff_tra, conc_old, 0, 0, m_nspec, 0);
        
        // Calculate the squared change in concentration of each species concentration
        Vector<Real> sq_change(m_nspec);
        for(int i=0; i < m_nspec; i++)
            {sq_change[i] = MultiFab::Dot(diff_tra, i, 1, 0);}
        
        // How many redox active species are in this simulation?
        int n_redox = min(m_nspec, 2);

        // Calculate the relative change in concentration of each redox active species
        // The denominator is the TOTAL concentration of both redox active species
        Vector<Real> rel_change(m_nspec);
        for(int i=0; i < n_redox; i++) 
            {rel_change[i] = sqrt(sq_change[i]) / m_conc_redox_tot;}

        // Get the largest relative change in concentration for any redox active species on this level
        change_conc = rel_change[0];
        for(int i=1; i < n_redox; i++)
            {change_conc = max(change_conc, rel_change[i]);}

        // Is the convergence criterion for concentration satisfied?
        is_converged_conc[lev] = (change_conc < tol_conc_pic);
      
        if(verbose > 1) 
        { 
            Print() << format("lev={:d}, c_conc={:5.3e}. relax={}. (omega={:5.2f}, tol_c={:5.2e}, fixpt_tol_c={:5.2e}.\n",
            lev, change_conc, m_fixpoint_relax, m_fixpoint_omega, tol_conc_pic, m_fixpoint_conc_tol);
        }
        
        if(verbose > 3) 
        { 
            for(int i=0; i<m_nspec; i++) 
            {
                Print() << format("(conc {:d}). relchange={:5.3e}. max={:5.3e}. min={:5.3e}. max_diff={:5.3e}. min_diff={:5.3e}\n",
                i, rel_change[i], conc_new.max(i), conc_new.min(i), diff_tra.max(i), diff_tra.min(i));
            }
        }
      
        // Get the largest relative change in epot on this level if applicable
        is_converged_epot[lev] = true;      
        // check only for reaction models with epot
        if(is_BV) 
        {
            MultiFab& epotL_new = m_leveldata[lev]->epotL;
            MultiFab const& epotL_old = m_leveldata[lev]->epotL_op;      
            MultiFab& epotS_new = m_leveldata[lev]->epotS;
            MultiFab const& epotS_old = m_leveldata[lev]->epotS_op;      

            MultiFab diff_epL;
            diff_epL.define(epotL_new.boxArray(), epotL_new.DistributionMap(), 1, 0);
            MultiFab::Copy(diff_epL, epotL_new, 0, 0, 1, 0);
            MultiFab::Subtract(diff_epL, epotL_old, 0, 0, 1, 0);
            Real sq_change = MultiFab::Dot(diff_epL, 0, 1, 0);
            Real sq_old=MultiFab::Dot(epotL_old,0,1,0);
            Real change_epL = 0.0;
            if(sq_change>0.0 && sq_old>0.0)
                {change_epL = sqrt(sq_change / sq_old);}
            else if(sq_change > 0.0)
                {change_epL = 1.0;}

            change_ep = change_epL;
            if(m_solve_epotS) 
            {
                MultiFab diff_epS;
                diff_epS.define(epotS_new.boxArray(), epotS_new.DistributionMap(), 1, 0);
                MultiFab::Copy(diff_epS,epotS_new, 0, 0, 1, 0);
                MultiFab::Subtract(diff_epS,epotS_old, 0, 0, 1, 0);
                sq_change=MultiFab::Dot(diff_epS, 0, 1, 0);
                sq_old=MultiFab::Dot(epotS_old, 0, 1, 0);
                Real change_epS = 0.0;
                if(sq_change>0.0 && sq_old>0.0)
                    {change_epS = sqrt(sq_change / sq_old);}
                else if(sq_change > 0.0)
                    {change_epS = 1.0;}
                change_ep = max(change_epL, change_epS);
            }
            // Is the convergence criterion for epotL satisfied?
            is_converged_epot[lev] = (change_ep < tol_epot_pic);
            
            if(verbose > 3) 
            {
                Print() << format("epL: max={:5.3e}. min={:5.3e}. max_diff={:5.3e}. min_diff={:5.3e}.\n",
                    epotL_new.max(0), epotL_new.min(0), diff_epL.max(0), diff_epL.min(0));
                if(m_solve_epotS) 
                    {Print() << format("epS: max={:5.3e}, min={:5.3e}\n", epotS_new.max(0), epotS_new.min(0));}
            }
            if(verbose > 1) 
            { 
                Print() << format("c_ep={:5.3e}, tol_ep={:5.3e}, fixpt_tol_ep={:5.3e}.\n", 
                change_ep, tol_epot_pic, m_fixpoint_epot_tol);
            }        
        } // if ButlerVolmer      
    } //lev
        
    // check if convergence has been achieved at all levels
    bool converged = true;
    for(int lev = 0; lev <= finest_level; lev++)      
        {converged = converged && is_converged_epot[lev] && is_converged_epot[lev];}

    if(converged && (m_verbose>1) ) 
        {Print() << format("Pic_conv: change_conc= {:5.3e}, change_ep={:5.3e}.\n", change_conc, change_ep);}
  
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return converged;
}
