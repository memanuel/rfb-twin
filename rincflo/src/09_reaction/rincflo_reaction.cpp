#include <rincflo.H>

// *********************************************************************************************************************
// reaction part of the simulation                                        //
// *********************************************************************************************************************
void Rincflo::React()
{
    #define FUNC_NAME "Rincflo::React"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // write header row on convergence_flow if necessary
    if (!cvg_react.tellp())
    {
        Print(cvg_react) << "StepNumber,SimTime,dt,WallTimeCum,WallTimeStep";
        for (int lev = 0; lev <= finest_level; lev++)
        {
            Print(cvg_react) << format(",ConcChange_{:d}", lev);
            if (m_reaction_model == ReactionModel::ButlerVolmer)
            {
                Print(cvg_react) << format(",PotChange_{:d}", lev);
            }
        }
        Print(cvg_react) << ",Current_mA,FluxDiff_mA\n";
    }

    Print() << "\n ============ START REACTION ============ \n";

    // Initialize variables for the reaction calculation
    InitReact();
    InitialRedistribution();

    // Calculate mac velocities (they will change only if fluid is evolved again)
    // First declare and sizes the mac velocities
    AMREX_D_TERM(
        Vector<MultiFab> u_mac;,
        Vector<MultiFab> v_mac;,
        Vector<MultiFab> w_mac;)
    AMREX_D_TERM(
        u_mac.resize(finest_level + 1);,
        v_mac.resize(finest_level + 1);,
        w_mac.resize(finest_level + 1);)

    // Now perform the mac calculation
    CalcMacVel(
        get_velocity_new_const(), get_density_new_const(),
        AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)));

    // Calculate the time step one time only in the Nernst model
    if (m_reaction_model == ReactionModel::Nernst)
    {
        ComputeDtNernst();
        Print() << format("Nernst model: set dt = {:6.2e}\n", m_dt);
    }

    // Start timing reaction simulation
    Real react_start_time = second();
    ReduceRealMax(react_start_time, IOProcessorNumber());
    // The starting value of the reaction step
    int strt_rstep = m_rstep;

    // Calculate the maximum current; this doesn't change during the reaction
    m_current_max = m_flow_out * m_conc_redox_tot * m_F_const * m_n_electrons;

    // Header for the profile line
    string profile_line_header = format("{:7s}, {:11s}, {:7s}, {:7s}, {:9s}",
        "Step", "SimTime", "dt", "WallTime", "StepTime");
    // Add the change in state of charge on each level and optionally in epot for BV model
    for(int lev = 0; lev <= finest_level; lev++)
    {
        profile_line_header += format(", SOC[{:d}]   ", lev);
        if(m_reaction_model == ReactionModel::ButlerVolmer) 
            {profile_line_header += format(", epot[{:d}]  ", lev);}
    }
    profile_line_header += ", Current";
    profile_line_header += ", Utilization";
    if (m_reaction_model == ReactionModel::Nernst)
        {profile_line_header += " , CurrDiff_MT";}
    profile_line_header += " , FluxDiff\n";
    Print() << profile_line_header;

    // Give the simulation a "warm start" by imposing the redox constraints and computing derived fields
    ImposeRedoxConcConstraint();
    CalcDerivedReact();

    // Variables to control the main time stepping loop

    // Have we taken too many steps?
    bool too_many_steps = (m_max_rsteps >= 0) && (m_rstep >= m_max_rsteps) && (m_rstep >= m_min_rsteps);
    // Should we continue simulating the reaction?
    bool do_react = !too_many_steps;
    // Cumulative time at the current step
    Real cum_time = 0.0;
    // Cumulative time at the previous step
    Real cum_time_prev = 0.0;
    // Time spent on most recent step
    Real step_time = 0.0;

    // Main time-stepping loop
    DEBUG_PRINT("Entering main reaction loop.\n");
    while (do_react)
    {
        // The reaction step counter; separate from the flow step counter m_step
        ++m_rstep;

        // Status
        if (m_verbose > 0)
            {Print() << "\n ============   NEW REACTION STEP " << m_rstep << " ============ \n";}

        // regrid if necessary
        if (m_regrid_react_int > 0 && (m_rstep > 1) && (m_rstep % m_regrid_react_int == 0))
        {
            if (m_verbose > 0)
                {Print() << "Regridding...\n";}

            // we want to be efficient and don't evolve the flow if no regriding...
            bool same_grids = true; // $ true if nothing changed, needs to be updated after regrid()

            regrid(0, m_cur_time);
            CalcGeometry();
            if (m_verbose > 0 && IOProcessor())
                {printGridSummary(OutStream(), 0, finest_level);}

            // if something changed need to evolve (automatically new time-step associated to velocity field...)
            if (!same_grids)
            {
                if (m_verbose > 0)
                    {Print() << "Evolving flow...\n";}
                m_do_flow = true;
                m_stop_time = -1;
                // don't save checkpoints during flow evolution
                m_check_int = -1;           
                //-1=don't plot during flow evolution
                m_plot_int = -1;            
                // this max of steps at most (anyway stops when steady-state reached again)
                m_max_step = m_nstep + 100; 
                // don't evolve conc (so we can evolve fluid without constraint on time-step from the reaction term)
                m_advect_conc = false;
                Evolve();
                m_advect_conc = true;

                // update data and fill ng (this would be automatically done if flow would be evolved)
                CopyNewToOld_velocity();
                CopyNewToOld_density();
                CopyNewToOld_conc();
                int ng = nghost_state();
                for (int lev = 0; lev <= finest_level; ++lev)
                {
                    fillpatch_velocity(lev, m_t_old[lev], m_leveldata[lev]->velocity_o, ng);
                    fillpatch_density(lev, m_t_old[lev], m_leveldata[lev]->density_o, ng);
                    fillpatch_conc(lev, m_t_old[lev], m_leveldata[lev]->conc_o, ng);
                    fillpatch_velocity(lev, m_t_new[lev], m_leveldata[lev]->velocity, ng);
                    fillpatch_density(lev, m_t_new[lev], m_leveldata[lev]->density, ng);
                    fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, ng);
                }
                // need to recompute mac_vel after regridding
                AMREX_D_TERM(
                    u_mac.resize(finest_level + 1);,
                    v_mac.resize(finest_level + 1);,
                    w_mac.resize(finest_level + 1);)
                CalcMacVel(
                    get_velocity_new_const(), get_density_new_const(),
                    AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)));
            } // if grid has changed
        }     // if regridding

        // Advance the reaction by one time step
        if (need_MembreaneBcUpdate())
            {SolveReactMem(AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)));}
        else
            {SolveReact(AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)));}

        // Impose redox constraints after every reaction time step. 
        ImposeRedoxConcConstraint();

        // Update the current; this is used in convergence testing and the profile line
        CalcCurrentFromSource();

        // Are we at steady state?
        bool is_steady_state = ReactionSteadyStateReached();

        // Cumulative time spent on the reaction simulation
        cum_time_prev = cum_time;
        cum_time = second() - react_start_time;
        ReduceRealMax(cum_time, IOProcessorNumber());
        // Time spent on most recent step
        step_time = cum_time - cum_time_prev;

        // Assemble a profile line to summarize progress in the reaction simulation
        // StepNumber, SimTime, dt, WallTimeCum, StepTime, SOC[0], ... SOC[n], Current_mA, Utilization, FluxDiffReal
        string profile_line = format("{: >7d}, {:11.8f}, {:7.2e}, {:7.0f}, {:9.6f}",
                                        m_rstep, m_cur_time, m_dt, cum_time, step_time);
        // Add the change in concentration on each level and optionally in epot for BV model
        for (int lev = 0; lev <= finest_level; lev++)
        {
            profile_line += format(", {:8.3e}", m_change_soc[lev]);
            if(m_reaction_model == ReactionModel::ButlerVolmer) 
                {profile_line += format(", {:7.2e}", m_change_epot[lev]);}
        }

        // Add the current and the flux-equivalent current; these are calculated in ReactionSteadyStateReached()
        constexpr Real a2ma = 1.0E3;
        // Report the total current in milliamps; always from BV kinetics on the wire even in Nernst model
        profile_line += format(", {:8.6f}", m_current*a2ma);
        // Report the utilization
        profile_line += format(", {:8.6f}", m_utilization);
        // In the Nernst model, also report relative difference between current_bv and current_mt
        if (m_reaction_model == ReactionModel::Nernst)
            {profile_line += format(", {:+8.3e}", m_curr_diff_rel);}
        // Report the relative difference between net flux at outlet vs. current on the wire
        profile_line += format(", {:+8.3e}\n", m_flux_diff_rel);

        // Write profile to file regardless of verbosity
        Print(cvg_react) << profile_line;

        // Terse output to console when verbosity is zero; only print every m_status_int lines OR on the last step
        if ((m_verbose ==0) && ((m_rstep % m_status_int == 0) || (!do_react)))
            {Print() << profile_line;}

        // Has it been too long since the last plot and checkpoint?
        if ((m_react_int_sec > 0) && (cum_time > m_react_status_time + m_react_int_sec))
        {
            WritePlotFile(CheckPointType::Reaction, m_rstep);
            WriteCheckPointFile(CheckPointType::Reaction);
            m_react_status_time = cum_time;
        }

        // Do we need a new plot due to steps taken?
        bool write_plot_now = (m_react_plot_int > 0) && (m_rstep % m_react_plot_int == 0) && (m_rstep > 0);
        if (write_plot_now)
        {
            WritePlotFile(CheckPointType::Reaction, m_rstep);
            m_last_plt = m_rstep;
            m_react_status_time = cum_time;
        }

        // Do we need a new checkpoint?
        bool write_chkp_now = (m_react_check_int > 0) && (m_rstep % m_react_check_int == 0) && (m_rstep > 0);
        if (write_chkp_now)
        {
            WriteCheckPointFile(CheckPointType::Reaction);
            m_last_chk = m_rstep;
        }

        // Do we need a numpy checkpoint?
        bool write_numpy_now = (m_react_numpy_int > 0) && (m_rstep % m_react_numpy_int == 0) && (m_rstep > 0);
        if (write_numpy_now)
            {WriteNumpyAll();}

        // Did we write anything extra to the console? Then we want to refresh the profile line header
        bool write_profile_header = write_plot_now || write_chkp_now || write_numpy_now;

        // Have we taken too few reaction steps?
        bool too_few_steps = (m_rstep < m_min_rsteps);
        // Have we taken too many reaction steps?
        too_many_steps = (m_max_rsteps >= 0) && (m_rstep >= m_max_rsteps);
        // Have we taken too many stagnant steps?
        bool too_many_stagnant_steps = (m_stagnant_steps >= m_stagnant_steps_max);
        // Should we terminate based on tolerance, stagnation and steps?
        bool stop_on_tol = (is_steady_state || too_many_steps || too_many_stagnant_steps) && (!too_few_steps);
        // Have we reached the fixed time limit?
        bool stop_on_time = (m_stop_time > 0) && (m_cur_time >= m_stop_time);
        // Stop if too many steps, steady state reached or stagnating
        // However, do NOT stop if we have taken too few steps
        do_react = !(stop_on_tol || stop_on_time);

        // Status message for type of termination
        if (is_steady_state && !do_react)
            {Print() << "Terminating reaction because steady state has been reached!\n";}
        else if (too_many_steps && !do_react)
            {Print() << format("Terminating reaction because too many steps ({:d}) have been taken.\n", m_rstep);}
        else if (too_many_stagnant_steps && !do_react)
            {Print() << format("Terminating reaction because too many stagnant steps ({:d}) have been taken.\n", m_stagnant_steps);}
        else if (stop_on_time && !do_react)
            {Print() << format("Terminating reaction simulation because stop time ({:10.6f}) reached.\n", m_stop_time);}
        else if (write_profile_header)
            {Print() << profile_line_header;}
    } // stop reaction

    // FINALIZE REACTION PART
    if (m_react_check_int > 0 && m_rstep != m_last_chk)
        {WriteCheckPointFile(CheckPointType::Reaction);}
    if (m_react_plot_int > 0 && m_rstep != m_last_plt)
        {WritePlotFile(CheckPointType::Reaction, m_rstep);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// perform one reaction simulation step                                   //
// *********************************************************************************************************************
void Rincflo::SolveReact(
    AMREX_D_DECL(
        Vector<MultiFab *> const &u_mac,
        Vector<MultiFab *> const &v_mac,
        Vector<MultiFab *> const &w_mac))
{
    #define FUNC_NAME "Rincflo::SolveReact"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Special case of the Nernst model has no source term and a fixed time step
    if (m_reaction_model == ReactionModel::Nernst)
    {
        // Run a single step in the Nernst model and return early
        NernstStep(AMREX_D_DECL(u_mac, v_mac, w_mac), m_dt, m_nernst_explicit_diffusion);
        return;
    }

    // Save previous reaction state variables: conc concentrations and optionally electric potentials
    StoreQuantities();

    // Calculate the time step in m_dt
    ComputeDtReaction();

    // Set time arrays for use in fillpatching
    UpdateTimeArrays();

    // Alias member variables for legibility
    int time_stepping_method = m_react_timestepping_method;
    int nonlin_method = m_react_nonlin_method;
    // Did the Picard iterations to solve diffusion and reaction converge?
    bool converged = false;
    // What was the total number of nonlinear iterations?
    int tot_nonlin_it = 0;

    // Multiplicative factor for a small increase in the max reactant consumption
    // constexpr Real mrc_factor = std::pow(2.0, 1.0 / 1024.0);
    constexpr Real mrc_factor = 1.00067713069306640783;

    // Fully explicit (it is NOT a good idea with EB!)
    if (time_stepping_method == 0)
    {
        // forward euler:
        AdvDiffReactStep_exp(AMREX_D_DECL(u_mac, v_mac, w_mac), m_dt, false, false);
        // Set the converged flag to true to simplify the downstream logic
        converged = true;
    }

    // [A - D(E)R] or [A/2 - D(E)R/2 - A/2]
    // time_stepping_method == 1: A - D(E)R
    // time_stepping_method == 2: A/2 - D(E)R - A/2
    else if ((time_stepping_method == 1) || (time_stepping_method == 2))
    {
        // Set the time step for advection based on the nonlinear method chosen
        Real dt_advect = (time_stepping_method == 1) ? m_dt : m_dt * 0.5;
        
        // Do the advection step before the reaction + diffusion stage
        AdvectionStep(AMREX_D_DECL(u_mac, v_mac, w_mac), dt_advect, false);

        // Attempt a diffusion + reaction step with the selected nonlinear method
        switch (nonlin_method)
        {
            case 1:
                converged = DiffReactStep_pic(m_dt, &tot_nonlin_it);
                break;
            case 2:
                converged = DiffReactStep_nwt(m_dt);
                break;
        }

        // Do a second half advection step if time_stepping_method == 2
        if (time_stepping_method==2)
            {AdvectionStep(AMREX_D_DECL(u_mac, v_mac, w_mac), dt_advect, false);}
    } // if time_stepping_method==1 or 2

    // predictor-corrector
    else if (time_stepping_method == 3)
    {
        // Predictor
        AdvectionStep(AMREX_D_DECL(u_mac, v_mac, w_mac), m_dt, true);

        int partial_nonlin_it = 0;
        switch (nonlin_method)
        {
            case 1:
                converged = DiffReactStep_pic(m_dt, &partial_nonlin_it);
                break;
            case 2:
                converged = DiffReactStep_nwt(m_dt);
                break;
        }
        tot_nonlin_it += partial_nonlin_it;

        // Corrector
        AdvectionStepCorrector(AMREX_D_DECL(u_mac, v_mac, w_mac), m_dt);
        switch (nonlin_method)
        {
            case 1:
                converged = converged && DiffReactStep_pic(m_dt, &partial_nonlin_it);
                break;
            case 2:
                converged = converged && DiffReactStep_nwt(m_dt);
                break;
        }

        tot_nonlin_it += partial_nonlin_it;
    } // if time_stepping_method==3

    // Did the diffusion + reaction solution converge?
    // If the loop converged, then we can slightly increase the maximum reactant consumption
    if (converged)
    {
        // Increase the maximum reactant consumption
        m_reactant_consumption_max = min(m_reactant_consumption_max * mrc_factor, m_reactant_consumption_max_limit);               ;

        // Set flag indicating a reaction step was not rolled back
        m_last_react_step_rollback = false;
    }
    // If if failed to converge, roll it back and tighten up the maximum reactant consumption
    else
    {
        // Reverse the bad time step
        StoreQuantities_undo();

        // Set flag indicating a reaction step was rolled back and increment error counter
        m_last_react_step_rollback = true;
        ++m_fail_count_picard_iter;

        // Tighten up the maximum reactant consumption
        m_reactant_consumption_max *= 0.5;

        // Report warning
        const char* nonlinear_method_cstr = (nonlin_method == 1) ? "Picard" : "Newton";
        Print() << format("WARNING! {:s} loop did not converge! (time_stepping_method=1.)\n", nonlinear_method_cstr);
        Print() << format("Rolled back reaction time step and set reaction_consumption_max={:9.6f}.\n", m_reactant_consumption_max);

        // Generate an error plot
        WritePlotFile(CheckPointType::Error, m_rstep);
    }

    // Report status about nonlinear iterations
    if (m_verbose > 0)
    {
        Print() << format("tot_nonlin_it= {:d}.\n", tot_nonlin_it);
        Print() << format("Failed picard iterations = {:d}.\n", m_fail_count_picard_iter);
    }

    // Fail if too many picard iterations have failed to converge
    if (m_fail_count_picard_iter >= 1000)
        {Abort("ABORT! Too many Picard iterations (1000) have failed to converge. ...\n");}
    
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// -------------------------------------------------------------------------------  //
// perform one reaction simulation step with iterative update of membrane b.c.    //
// -------------------------------------------------------------------------------  //
void Rincflo::SolveReactMem(
    AMREX_D_DECL(
        Vector<MultiFab *> const &u_mac,
        Vector<MultiFab *> const &v_mac,
        Vector<MultiFab *> const &w_mac))
{
    #define FUNC_NAME "Rincflo::SolveReactMem"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Save previous reaction state variables: conc concentrations and optionally electric potentials
    StoreQuantities();

    // Calculate the time step in m_dt
    ComputeDtReaction();

    // Set time arrays for use in fillpatching
    UpdateTimeArrays();

    int time_stepping_method = m_react_timestepping_method;
    int nonlin_method = m_react_nonlin_method;
    int tot_nonlin_it = 0;
    int step_nonlin_it = 0;

    if (time_stepping_method != 1 || nonlin_method != 1)
    {
        Abort("ABORT! SolveReactMem... only timestepping=1 and nonlin=1\n");
    }

    // operator splitting:  A + DR
    AdvectionStep(AMREX_D_DECL(u_mac, v_mac, w_mac), m_dt, false);

    // BC for membrane are updated iteratively (and used for solving the epotL equations)
    Real bc_mem = 0.0;
    Real bc_mem_old = 0.0;
    bool do_bcmem_it = true;
    int it_bcmem = 0;
    int max_it_bcmem = 100;
    bool bcmem_converged = false;
    Real tol_bcmem = 1e-5;

    // calc b.c.
    bc_mem = UpdateBCMem();

    if (m_verbose > 2)
        {Print() << "BCmem loop, init_bc= " << bc_mem << std::endl;}

    while (do_bcmem_it)
    {
        it_bcmem++;

        bc_mem_old = bc_mem;

        // DR is solve iteratively
        bool converged_pic = false;
        converged_pic = DiffReactStep_pic(m_dt, &step_nonlin_it);

        tot_nonlin_it += step_nonlin_it;

        if (converged_pic == false)
        {
            WritePlotFile(CheckPointType::Error, m_rstep);
            Print() << "WARNING! Picard loop did not converge! Step will be rolled back.\n";
            ++m_fail_count_picard_iter;
        }

        // calc new b.c.
        bc_mem = UpdateBCMem();

        // converged?
        if (std::fabs(bc_mem - bc_mem_old) < tol_bcmem * std::fabs(bc_mem))
        {
            do_bcmem_it = false;
            bcmem_converged = true;
        }

        // too many steps?
        if (it_bcmem > max_it_bcmem)
        {
            do_bcmem_it = false;
            bcmem_converged = false;
        }

        if (m_verbose > 2)
        {
            Print() << format("BCmem loop, done: {} current_bc= {} prev_bc= {}.\n", it_bcmem, bc_mem, bc_mem_old);
        }
    }

    if (bcmem_converged == false)
    {
        Print() << "WARNING! BCmem loop did not converge! Step will be rolled back.\n";
        // Save error plot, impose redox constraint, and increment error counter and continue
        ++m_fail_count_picard_iter;
        WritePlotFile(CheckPointType::Error, m_rstep);
    }

    if (m_verbose > 0)
    {
        Print() << format("BCmem_iterations: {} tot_Picard_iterations {}.\n", it_bcmem, tot_nonlin_it);
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// calculate advection term                                               //
// *********************************************************************************************************************
void Rincflo::AdvectionStep(
    AMREX_D_DECL(
    Vector<MultiFab *> const &u_mac,
    Vector<MultiFab *> const &v_mac,
    Vector<MultiFab *> const &w_mac),
    Real dt,
    bool store_old)
{
    #define FUNC_NAME "Rincflo::AdvectionStep"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The new rate of change in concentration due to advection
    Vector<MultiFab *> const& dconc_dt_mf = get_conv_dconc_dt_new();
    // The new concentration
    Vector<MultiFab *> const& conc_new_mf = get_conc_new();
    // The new concentration after update only for advection
    Vector<MultiFab *> const& conc_adv_mf = get_conc_adv_new();

    // Calculate convective term
    ComputeConvectiveTermConc(dconc_dt_mf, GetVecOfConstPtrs(conc_new_mf), AMREX_D_DECL(u_mac, v_mac, w_mac));

    // Copy the convected concentration from conc_adv to conc_adv_o if requested
    if (store_old)
        {CopyNewToOld_conc_adv();}

    // update conc
    int l_nspec = m_nspec;
    for (int lev = 0; lev <= finest_level; lev++)
    {
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*conc_new_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            // rate of change in concentration due to advection
            Array4<Real const> const &dconc_dt = dconc_dt_mf[lev]->const_array(mfi);
            // new concentration - to be updated
            Array4<Real> const &conc_new = conc_new_mf[lev]->array(mfi);        
            // concentration after update for advection only; will be the same as conc after this step
            Array4<Real> const &conc_adv = conc_adv_mf[lev]->array(mfi);

            auto func = 
            [=, this] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                for (int n = 0; n < l_nspec; ++n)
                {
                    conc_adv(i,j,k,n) = conc_new(i,j,k,n) + dt * dconc_dt(i,j,k,n);
                    conc_new(i,j,k,n) = conc_adv(i,j,k,n);
                } 
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::AdvectionStepCorrector(
    AMREX_D_DECL(
        Vector<MultiFab *> const &u_mac,
        Vector<MultiFab *> const &v_mac,
        Vector<MultiFab *> const &w_mac),
    Real dt)
{
    #define FUNC_NAME "Rincflo::AdvectionStepCorrector"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Vector<MultiFab *> const &dconc_dt_new_mf = get_conv_dconc_dt_new();
    Vector<MultiFab *> const &dconc_dt_old_mf = get_conv_dconc_dt_old();
    Vector<MultiFab *> const &conc_new_mf = get_conc_new();
    Vector<MultiFab *> const &conc_old_mf = get_conc_old();
    Vector<MultiFab *> const &conc_adv_mf = get_conc_adv_new();

    // Calculate convective term
    ComputeConvectiveTermConc(dconc_dt_new_mf, GetVecOfConstPtrs(conc_new_mf), AMREX_D_DECL(u_mac, v_mac, w_mac));

    // update conc
    int l_nspec = m_nspec;
    for (int lev = 0; lev <= finest_level; lev++)
    {
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*dconc_dt_new_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            // conc from previous timestep
            Array4<Real> const &conc_old = conc_old_mf[lev]->array(mfi);   
            // convective term that will store updated value
            Array4<Real> const &dconc_dt_new = dconc_dt_new_mf[lev]->array(mfi);     
            // convective term from predictor
            Array4<Real> const &dconc_dt_old = dconc_dt_old_mf[lev]->array(mfi); 
            // concentration after update for advection only
            Array4<Real> const &conc_adv = conc_adv_mf[lev]->array(mfi);
            auto func =
            [=, this] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
            {
                for (int n = 0; n < l_nspec; ++n) 
                    {conc_adv(i,j,k,n) = conc_old(i,j,k,n) + 0.5* dt * (dconc_dt_new(i,j,k,n) + dconc_dt_old(i,j,k,n));} 
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// solve diffusion part                                                   //
// *********************************************************************************************************************
void Rincflo::DiffStep(Real dt, bool set_rhs)
{
    #define FUNC_NAME "Rincflo::DiffStep"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Construct RHS and solve diff equations for concentration
    UpdateDiffConc(dt, set_rhs);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
