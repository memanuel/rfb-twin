#include <rincflo.H>

// *********************************************************************************************************************
Rincflo::Rincflo()
{
    // NOTE: Geometry on all levels has just been defined in the AmrCore
    // constructor. No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    #if (DEBUG_RFB)
    DebugPrintSetup();
    #endif

    // Read inputs file using ParmParse
    ReadParameters();

    // This is needed before initializing level MultiFab
    AMREX_EB_ONLY(MakeEBGeometry();)

    // Initialize memory for data-array internals
    ResizeArrays();

    // Set up boundary conditions and initial fluid flow steps
    InitBoundaryConditions();
    InitAdvection();
    SetBackgroundPressure(); 
}

// *********************************************************************************************************************
Rincflo::~Rincflo()
{}

// *********************************************************************************************************************
void Rincflo::InitData()
{
    #define FUNC_NAME "Rincflo::InitData"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Here we are starting from scratch
    if(!is_restart())
    {
        // This tells the Cluster routine to use the new chopping routine which rejects cuts if they don't improve the efficiency
        SetUseNewChop();

        // This is an AmrCore member function which recursively makes new levels with MakeNewLevelFromScratch.
        InitFromScratch(m_cur_time);
    }
    // Here we are restarting from a checkpoint file
    else
    {
        // Read parameters in namespace rincflo to control details of the restart
        ParmParse pp("rincflo");

        // Read starting configuration from chk file.
        ReadCheckPointFile();
        if (m_verbose > 0) 
            {Print() << "Rincflo constructor - read checkpoint file.\n";}

        // Is this restart an upsampling operation? If so, what level are we upsampling from?
        // AMReX calls this lbase; it's the first level that is rebuilt during the regridding operation.
        int lbase = 0;
        pp.query("upsample_from", lbase);

        // Do the initial regridding operation
        if (m_verbose > 0) 
            {Print() << format("Regridding after restart from lbase={:d}...\n", lbase);}
        regrid(lbase, m_cur_time);

        // Respect precedence of nstep and rstep in the input file (otherwise these get clobbered by checkpoint)
        pp.query("nstep", m_nstep);
        pp.query("rstep", m_rstep);

        // Scale velocity if applicable
        if (m_vel_scale_factor != 1.0)
            {ScaleInputVelocity();}

        // Analyze flow; only do this on a restart - no velocity field yet on a new simulation
        AnalyzeFlow();
    }

    AMREX_EB_ONLY(InitialRedistribution());

    // Compute the derived geometry
    CalcGeometry();

    // Compute derived react quantities if running a reaction model
    if (is_Reaction())
        {CalcDerivedReact();}

    // Display mean porosity on each level
    #if (AMREX_USE_EB)
    for(int lev=0; lev<= finest_level ; lev++)  
    {
        Real mean_vfrac = CalcAverageVolumeFraction(lev);
        Print() << format("mean vfrac (porosity) = {:0.6f} (lev {:d})\n", mean_vfrac, lev);
    }
    #endif

    // These operations are only without restart files: initial projection, initial iterations, initial checkpoint
    if (!is_restart())
    {
        if (m_do_initial_proj) 
            {InitialProjection();}
        if (m_initial_iterations > 0) 
            {InitialIterations();}
        if (!is_ActiveSimReact() && m_check_int > 0)
            {WriteCheckPointFile(CheckPointType::FluidFlow); }
    }

    // Write out EB surface if requested
    #if defined(AMREX_USE_EB) && defined(AMREX_IS_3D)
    ParmParse pp("rincflo");
    bool write_eb_surface = false;
    pp.query("write_eb_surface", write_eb_surface);
    if (write_eb_surface)
    {
        Print() << "Writing EB surface.\n";
        WriteMyEBSurface();
    }
    #endif
    
    // Print grid summary
    if (m_verbose>0 && IOProcessor()) 
        {printGridSummary(OutStream(), 0, finest_level);}

    // Write out grid summary to file
    printGridSummary(grid_info, 0, finest_level);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::UpdateTimeArrays()
{
    #define FUNC_NAME "Rincflo::UpdateTimeArrays"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Set new time on the arrays m_t_old and m_t_new to use in fillpatching
    for (int lev = 0; lev <= finest_level; lev++)
    {
        m_t_old[lev] = m_cur_time;
        m_t_new[lev] = m_cur_time + m_dt;
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::UpdateTime()
{
    #define FUNC_NAME "Rincflo::UpdateTime"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Set new time to correctly use in fillpatching
    UpdateTimeArrays();

    // Status
    if (m_verbose > 0)
    {
        Print() << format("from old_time {:12.8f} to new time {:12.8f} with dt = {:6.2e}.\n",
                          m_cur_time, m_cur_time + m_dt, m_dt);
    }

    // Apply the step dt to cur_time
    m_cur_time += m_dt;

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::Evolve()
{
    #define FUNC_NAME "Rincflo::Evolve()"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Write header row on convergence_flow if necessary
    if (!cvg_flow.tellp())
        {Print(cvg_flow) << "StepNumber,SimTime,dt,WallTimeCum,WallTimePerStep,VelChange\n";}

    // Flag indicating if we should terminate the main flow simulation loop
    bool do_not_evolve = ((m_max_step >= 0) && (m_nstep >= m_max_step)) || 
        ((m_stop_time > 0.0) && (m_cur_time >= m_stop_time - 1.e-12 * m_dt));
    
    // If we're not simulating the flow, there is no need to run any flow steps
    if (m_do_flow == false)
        {do_not_evolve = true;}

    // Flag indicating whether we've achieved the steady state condition
    bool is_steady_state = false;
    // Flag indicating if we've taken too many steps
    bool too_many_steps = false;
    // Flag indicating whether we've taken too many stagnant flow steps
    bool too_many_stagnant_flow_steps = false;
    // Flag indicating if we've reached the fixed time limit?
    bool stop_on_time = false;

    // Start timing flow simulation
    Real flow_start_time = second();
    ReduceRealMax(flow_start_time, IOProcessorNumber());
    // The starting value of the flow step
    int strt_nstep = m_nstep;

    // Main flow simulation loop
    while(!do_not_evolve)
    {
        if (m_verbose > 0)
            {Print() << "\n ============   NEW TIME STEP   ============ \n";}

        if ((m_regrid_flow_int > 0) && (m_nstep > 0) && (m_nstep % m_regrid_flow_int == 0))
        {
            if (m_verbose > 0) {Print() << "Regridding...\n";}
            regrid(0, m_cur_time);
            CalcGeometry();
            if (m_verbose > 0 && IOProcessor())
                {printGridSummary(OutStream(), 0, finest_level);}
        }

        // Advance to time t + dt
        Advance();
        m_nstep++;
        m_cur_time += m_dt;

        // Cumulative time spent
        Real cum_time = second() - flow_start_time;
        ReduceRealMax(cum_time, IOProcessorNumber());
        // Total number of steps simulated in this run
        int num_steps = m_nstep - strt_nstep;
        // Average time spent per step in seconds
        Real avg_step_time = cum_time / num_steps;        

        // Assemble profile line for printing to file and optionally to console
        // StepNumber,SimTime,dt,WallTimeCum,WallTimePerStep
        string profile_line = format("{: >6d}, {:10.8f}, {:.2e}, {:7.0f}, {:9.6f}", 
            m_nstep, m_cur_time, m_dt, cum_time, avg_step_time);
        // Add change in each level
        for(int lev = 0; lev <= finest_level; lev++)
            {profile_line += format(", {:6.3e}", m_change_vel[lev]);}
        profile_line += "\n";

        // Write profile to file regardless of verbosity
        Print(cvg_flow) << profile_line;

        // Terse output to console when verbosity is zero; only print every m_status_int steps OR on the last step
        if ((m_verbose == 0) && ((m_nstep % m_status_int == 0) || do_not_evolve))
            {Print() << profile_line;}

        // Write plotfile if necessary
        if (WriteNow())
        {
            WritePlotFile(CheckPointType::FluidFlow, m_nstep);
            m_last_plt = m_nstep;
        }

        // Write checkpoint if necessary
        if ((m_check_int > 0) && (m_nstep % m_check_int == 0))
        {
            WriteCheckPointFile(CheckPointType::FluidFlow);
            m_last_chk = m_nstep;
        }

        // Write numpy checkpoint if necessary
        if ((m_flow_numpy_int > 0) && (m_nstep % m_flow_numpy_int == 0))   
            {WriteNumpyAll();}

        // Should we stop because we've reached steady state *and* that has been set as a termination condition?
        // Put m_steady_state on RHS to avoid short circuit evaluation; always want to call SteadyStateReached()
        is_steady_state = SteadyStateReached() && m_steady_state;

        // Have we taken too many steps?
        too_many_steps = (m_max_step >= 0) && (m_nstep >= m_max_step);

        // Have we taken too many stagnant flow steps?
        too_many_stagnant_flow_steps = m_steady_state && (m_stagnant_flow_steps >= m_stagnant_flow_steps_max) 
            && (m_stagnant_flow_steps_max > 0);

        // Have we reached the fixed time limit?
        stop_on_time = (m_stop_time > 0) && (m_cur_time >= m_stop_time);

        // Mechanism to terminate Rincflo normally
        do_not_evolve = is_steady_state || too_many_steps || too_many_stagnant_flow_steps || stop_on_time;
    }
  
    // Indicate reason for termination
    if (is_steady_state)
        {Print() << "Terminating flow simulation because steady state reached.\n";}
    else if (too_many_steps)
        {Print() << format("Terminating flow simulation because max steps ({:d}) reached.\n", m_max_step);}
    else if (too_many_stagnant_flow_steps)
        {Print() << format("Terminating flow simulation because too many stagnant flow steps ({:d}) taken.\n", 
            m_stagnant_flow_steps);}
    else if (stop_on_time)
        {Print() << format("Terminating flow simulation because stop time ({:10.6f}) reached.\n", m_stop_time);}

    // Output at the final time
    bool write_final_chk = ((m_check_int > 0) && (m_nstep != m_last_chk)) || (m_check_int <= 0);
    if( write_final_chk) 
        {WriteCheckPointFile(CheckPointType::FluidFlow);}
    bool write_final_plt = 
        ((m_plot_int > 0 || m_plot_per_exact > 0 || m_plot_per_approx > 0) && (m_nstep != m_last_plt)) ||
        (m_plot_int <= 0);
    if( write_final_plt)
        {WritePlotFile(CheckPointType::FluidFlow, m_nstep); }
    
    // Close file with flow convergence  
    cvg_flow.close();

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
bool Rincflo::WriteNow()
{
    bool write_now = false;

    if ( m_plot_int > 0 && (m_nstep % m_plot_int == 0) ) 
        {write_now = true;}

    else if ( m_plot_per_exact  > 0 && (std::abs(std::remainder(m_cur_time, m_plot_per_exact)) < 1.e-12) ) 
        {write_now = true;}

    else if (m_plot_per_approx > 0.0)
    {
        // Check to see if we've crossed a m_plot_per_approx interval by comparing
        // the number of intervals that have elapsed for both the current
        // time and the time at the beginning of this timestep.

        int num_per_old = static_cast<int>(std::round((m_cur_time-m_dt) / m_plot_per_approx));
        int num_per_new = static_cast<int>(std::round((m_cur_time     ) / m_plot_per_approx));

        // Before using these, however, we must test for the case where we're
        // within machine epsilon of the next interval. In that case, increment the counter, 
        // because we have indeed reached the next m_plot_per_approx interval at this point.

        const Real eps = std::numeric_limits<Real>::epsilon() * 10.0 * std::abs(m_cur_time);
        const Real next_plot_time = (num_per_old + 1) * m_plot_per_approx;

        if ((num_per_new == num_per_old) && std::abs(m_cur_time - next_plot_time) <= eps)
            {num_per_new += 1;}

        // Similarly, we have to account for the case where the old time is within
        // machine epsilon of the beginning of this interval, so that we don't double
        // count that time threshold -- we already plotted at that time on the last timestep.

        if ((num_per_new != num_per_old) && std::abs((m_cur_time - m_dt) - next_plot_time) <= eps)
            {num_per_old += 1;}

        if (num_per_old != num_per_new)
            {write_now = true;}
    }

    return write_now;
}
