#include <rincflo.H>

// *********************************************************************************************************************
void Rincflo::ReadParameters()
{
    #define FUNC_NAME "Rincflo::ReadParameters"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    {
        // Variables with prefix Rincflo
        ParmParse pp("rincflo");

        // Which actions should Rincflo take?
        pp.query("do_flow",     m_do_flow);
        pp.query("do_reaction", m_do_reaction);
        pp.query("do_analysis", m_do_analysis);

        // Load from or write to numpy
        pp.query("read_numpy",  m_read_numpy);
        pp.query("write_numpy", m_write_numpy);

        // Directories with numpy data
        pp.query("numpy_input_dir_flow",   m_numpy_input_dir_flow);
        pp.query("numpy_input_dir_react",  m_numpy_input_dir_react);
        pp.query("numpy_output_dir",       m_numpy_output_dir);

        // How many levels of numpy data do we expect and what offset should we apply?
        pp.query("numpy_lev_flow",  m_numpy_lev_flow);
        pp.query("numpy_lev_offset_flow",  m_numpy_lev_offset_flow);
        pp.query("numpy_lev_react", m_numpy_lev_react);
        pp.query("numpy_lev_offset_react", m_numpy_lev_offset_react);

        // Verbosity of numpy output
        pp.query("numpy_verbose", m_numpy_verbose);

        // Velocity scale factor
        pp.query("vel_scale_factor", m_vel_scale_factor);

        // Interval between printing status update to console for each flow or reaction step
        pp.query("status_int", m_status_int);

        // Interval between numpy checkpoints
        pp.query("flow_numpy_int", m_flow_numpy_int);
        pp.query("react_numpy_int", m_react_numpy_int);

        // Optionally reset counters for fluid flow (nstep) or reaction (rstep)
        pp.query("nstep", m_nstep);
        pp.query("rstep", m_rstep);

        // Report actions taken
        PrintStars();
        Print() << "Actions to be taken by Rincflo:\n";
        if (do_flow())
            {Print() << format("Simulate flow.\n");}
        if (do_reaction())
            {Print() << format("Simulate reaction.\n");}
        if (do_ReadNumpy())
            {Print() << format("Read numpy data.\n");}
        if (do_WriteNumpy())
            {Print() << format("Write numpy data.\n");}
        if (do_Analysis())
            {Print() << format("Perform analysis of summary statistics.\n");}

        // Set flow options
        pp.query("stop_time", m_stop_time);
        pp.query("max_step", m_max_step);
        pp.query("steady_state", m_steady_state);
        pp.query("steady_state_tol", m_steady_state_tol);
        pp.query("tol_flow_rate_comp", m_tol_flow_rate_comp);
        pp.query("tol_flow_rate_mag", m_tol_flow_rate_mag);
        pp.query("tol_flow_relative", m_tol_flow_relative);
        pp.query("tol_flow_relativerate", m_tol_flow_relativerate);
        pp.query("stagnant_flow_steps_max", m_stagnant_flow_steps_max);

        // Check consistency of flow inputs
        if(m_tol_flow_relativerate) 
        {
            m_tol_flow_rate_comp=false; 
            m_tol_flow_rate_mag=false; 
            m_tol_flow_relative=false; 
        }
        else if (m_tol_flow_rate_comp) 
        {
            m_tol_flow_rate_mag=false; 
            m_tol_flow_relative=false; 
            m_tol_flow_relativerate=false;
        }
        else if (m_tol_flow_rate_mag) 
        {
            m_tol_flow_rate_comp=false; 
            m_tol_flow_relative=false; 
            m_tol_flow_relativerate=false;
        }
        else if (m_tol_flow_relative) 
        {
            m_tol_flow_rate_comp=false; 
            m_tol_flow_rate_mag=false; 
            m_tol_flow_relativerate=false;
        }
        else 
        { 
            // Nothing specified, use defaults
            m_tol_flow_rate_comp=false; 
            m_tol_flow_rate_mag=false; 
            m_tol_flow_relative=true; 
            m_tol_flow_relativerate=false;
        }
    }
    
    ReadIOParameters();
    ReadRheologyParameters();
    {   
        // Prefix amr
        ParmParse pp("amr");
        pp.query("regrid_flow_int", m_regrid_flow_int);
        #if (AMREX_USE_EB)
        pp.query("refine_cutcells", m_refine_cutcells);
        pp.query("ngrow", m_ngrow);
        pp.query("build_coarse_level_by_coarsening", m_build_coarse_level_by_coarsening);
        #endif
    }   // end prefix amr

    {
        // Prefix eb2
        ParmParse pp("eb2");
        // Set handling of multiply cut cells (this is done in EB2 subclass, but put a copy here too)
        pp.query("cover_multiple_cuts", m_cover_multiple_cuts);       
    }

    {   
        // Prefix Rincflo
        ParmParse pp("rincflo");
        pp.query("verbose", m_verbose);
        pp.query("initial_iterations", m_initial_iterations);
        pp.query("do_initial_proj", m_do_initial_proj);
        // The (flow) CFL inputs
        pp.query("cfl", m_cfl);
        pp.queryarr("cfl_sched", m_cfl_sched, 0, 2);
        // The fixed time step inputs
        pp.query("fixed_dt", m_fixed_dt);
        pp.queryarr("fixed_dt_sched", m_fixed_dt_sched, 0, 2);

        // Set flags describing time stepping
        m_using_cfl_single = (m_cfl > 0.0);
        m_using_cfl_sched = (m_cfl_sched[0] > 0.0) && (m_cfl_sched[1] > 0.0);
        m_using_cfl = m_using_cfl_single || m_using_cfl_sched;
        m_using_fixed_dt_single = (m_fixed_dt > 0.0);
        m_using_fixed_dt_sched = (m_fixed_dt_sched[0] > 0.0) && (m_fixed_dt_sched[1] > 0.0);
        m_using_fixed_dt = m_using_fixed_dt_single || m_using_fixed_dt_sched;

        if (m_verbose > 1)
            {Print() << "Read fluid initial iterations and CFL parameters.\n";}

        // Test for consistency in time-step specification
        if (m_using_cfl && m_using_fixed_dt)
            {Abort("specified both cfl and fixed_dt.");}
        if (m_using_cfl_single && m_using_cfl_sched)
            {Abort("specified both a single cfl and a cfl schedule.");}
        if (m_using_fixed_dt_single && m_using_fixed_dt_sched)
            {Abort("specified both a single fixed_dt and a fixed_dt schedule.");}
        
        // This will multiply the time-step in the very first step only
        pp.query("init_shrink", m_init_shrink);
        if (m_init_shrink > 1.0) 
            {Abort("We require m_init_shrink <= 1.0");}

        // Physics
        pp.queryarr("delp", m_delp, 0, SpaceDim);
        pp.queryarr("gravity", m_gravity, 0, SpaceDim);
        pp.query("constant_density", m_constant_density);
        pp.query("advect_conc", m_advect_conc);
        pp.query("test_species_conservation" , m_test_species_conservation);

        // Are we using MOL or Godunov?
        pp.query("advection_type"                   , m_advection_type);
        pp.query("use_ppm"                          , m_godunov_ppm);
        pp.query("godunov_use_forces_in_trans"      , m_godunov_use_forces_in_trans);
        pp.query("godunov_include_diff_in_forcing"  , m_godunov_include_diff_in_forcing);
        pp.query("use_mac_phi_in_godunov"           , m_use_mac_phi_in_godunov);

        if (m_verbose > 1)
            {Print() << "Read physics parameters and set advection type.\n";}

        // What type of redistribution algorithm; default is FluxRedistribution, options are
        // {NoRedist, FluxRedist, MergeRedist, StateRedist}
        #if (AMREX_USE_EB)
        pp.query("redistribution_type", m_redistribution_type);
        if (m_redistribution_type != "NoRedist" && 
            m_redistribution_type != "FluxRedist" &&
            m_redistribution_type != "MergeRedist" &&
            m_redistribution_type != "StateRedist")
            {Abort("redistribution type must be NoRedist, FluxRedist, MergeRedist, or StateRedist");}

        if (m_advection_type == "Godunov" && m_godunov_ppm) 
            {Abort("Cant use PPM with EBGodunov");}
        #endif

        if (m_advection_type == "MOL") 
            {m_godunov_include_diff_in_forcing = false;}

        if (m_advection_type != "MOL" and m_advection_type != "Godunov")
            {Abort("advection type must be MOL or Godunov");}

        // The default for diffusion_type is 2, i.e. the default m_diff_type is DiffusionType::Implicit
        // TODO: change diffusion_type from an integer to an enum
        int diffusion_type = 2;
        pp.query("diffusion_type", diffusion_type);
        if (diffusion_type == 0) 
            {m_diff_type = DiffusionType::Explicit;} 
        else if (diffusion_type == 1) 
            {m_diff_type = DiffusionType::Crank_Nicolson;} 
        else if (diffusion_type == 2) 
            {m_diff_type = DiffusionType::Implicit;} 
        else 
            {Abort("We currently require diffusion_type = 0 for explicit, 1 for Crank-Nicolson or 2 for implicit");}

        // Default is true; should we use tensor solve instead of separate solves for each component?
        pp.query("use_tensor_solve", use_tensor_solve);
        pp.query("use_tensor_correction", use_tensor_correction); //default=false

        if (use_tensor_solve && use_tensor_correction) 
            {Abort("We cannot have both use_tensor_solve and use_tensor_correction be true");}

        if (m_diff_type != DiffusionType::Implicit && use_tensor_correction) 
            {Abort("We cannot have use_tensor_correction be true and diffusion type not Implicit");}
    
        if (m_cfl > 1.0) 
            {Abort("We currently require cfl <= 1.0");}

        // Get the nominal height in 2D
        #if (AMREX_IS_2D)
        pp.query("height", m_height);
        #endif

        // Initial conditions
        pp.query("probtype", m_probtype);
        pp.query("ic_u", m_ic_u);
        pp.query("ic_v", m_ic_v);
        pp.query("ic_w", m_ic_w);
        pp.query("ic_p", m_ic_p);

        // Viscosity (if constant)
        pp.query("mu", m_mu);

        // Density (if constant)
        pp.query("ro_0", m_ro_0);
        AMREX_ALWAYS_ASSERT(m_ro_0 > 0.0);

        // Number of species
        pp.query("nspec", m_nspec);
        if (m_nspec < 2) 
            {Abort("We currently require at least two species");}

        // Scalar diffusion coefficients
        m_D_s.resize(m_nspec, 0.0);
        pp.queryarr("D_s", m_D_s, 0, m_nspec );
        m_ic_t.resize(m_nspec, 0.0);
        pp.queryarr("ic_t", m_ic_t, 0, m_nspec);
    }   // end prefix Rincflo

    {   
        // Prefix mac
        ParmParse pp("mac_proj");
        pp.query("bottom_solver"        , m_mac_bottom_solver);
        pp.query("verbose"              , m_mac_mg_verbose );
        pp.query("bottom_verbose"       , m_mac_mg_bottom_verbose );
        pp.query("rtol"                 , m_mac_mg_rtol );
        pp.query("atol"                 , m_mac_mg_atol );
        pp.query("maxiter"              , m_mac_mg_maxiter );
        pp.query("bottom_maxiter"       , m_mac_mg_bottom_maxiter );
        pp.query("max_coarsening_level" , m_mac_mg_max_coarsening_level );

        // query the hypre domain to suppress a false warning about unused ParmParse variables
        string throwaway_s = "";
        int throwaway_i = 0;
        Real throwaway_r = 0.0;
        ParmParse pph("mac_proj.hypre");
        pph.query("hypre_solver"         , throwaway_s);
        pph.query("num_krylov"           , throwaway_i);
        pph.query("max_iterations"       , throwaway_i);
        pph.query("rtol"                 , throwaway_r);
        pph.query("hypre_preconditioner" , throwaway_s);
        pph.query("bamg_max_levels"      , throwaway_i);
        pph.query("bamg_agg_num_levels"  , throwaway_i);
        pph.query("bamg_num_sweeps"      , throwaway_i);
        pph.query("bamg_relax_type"      , throwaway_i);
        pph.query("bamg_relax_order"     , throwaway_i);
        pph.query("bamg_coarsen_type"    , throwaway_i);
        pph.query("bamg_interp_type"     , throwaway_i);
    }

    if (m_verbose > 1)
        {Print() << "Read parameters for MAC solver.\n";}

    {   
        // Prefix nodal
        ParmParse pp("nodal_proj");
        pp.query("bottom_solver"         , m_nodal_bottom_solver);
        pp.query("verbose"               , m_nodal_mg_verbose );
        pp.query("bottom_verbose"        , m_nodal_mg_bottom_verbose );
        pp.query("rtol"                  , m_nodal_mg_rtol );
        pp.query("atol"                  , m_nodal_mg_atol );
        pp.query("maxiter"               , m_nodal_mg_maxiter );
        pp.query("bottom_maxiter"        , m_nodal_mg_bottom_maxiter );
        pp.query("max_coarsening_level"  , m_nodal_mg_max_coarsening_level );

        // query the hypre domain to suppress a false warning about unused ParmParse variables
        string throwaway_s = "";
        int throwaway_i = 0;
        Real throwaway_r = 0.0;
        ParmParse pph("nodal_proj.hypre");
        pph.query("hypre_solver"         , throwaway_s);
        pph.query("num_krylov"           , throwaway_i);
        pph.query("max_iterations"       , throwaway_i);
        pph.query("rtol"                 , throwaway_r);
        pph.query("hypre_preconditioner" , throwaway_s);
        pph.query("bamg_max_levels"      , throwaway_i);
        pph.query("bamg_agg_num_levels"  , throwaway_i);
        pph.query("bamg_num_sweeps"      , throwaway_i);
        pph.query("bamg_relax_type"      , throwaway_i);
        pph.query("bamg_relax_order"     , throwaway_i);
        pph.query("bamg_coarsen_type"    , throwaway_i);
        pph.query("bamg_interp_type"     , throwaway_i);
    }
    
    if (m_verbose > 1)
        {Print() << "Read parameters for nodal solver.\n";}

    // Summarize the fluid model parameters
    PrintFluidModelSummary();

    // Read the reaction parameters; these will be reported to screen as appropriate
    ReadReactionParameters();

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::ReadIOParameters()
{
    #define FUNC_NAME "Rincflo::ReadIOParameters"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Prefix amr
    ParmParse pp("amr");

    pp.query("check_file", m_check_file);
    pp.query("check_int", m_check_int);
    pp.query("check_int_sec", m_check_int_sec);
    pp.query("restart", m_restart_file);

    pp.query("plot_file", m_plot_file);
    pp.query("plot_int", m_plot_int);
    pp.query("plot_int_sec", m_plot_int_sec);
    pp.query("plot_per_exact" , m_plot_per_exact);
    pp.query("plot_per_approx", m_plot_per_approx);

    if ( (m_plot_int       > 0 && m_plot_per_exact  > 0) ||
         (m_plot_int       > 0 && m_plot_per_approx > 0) ||
         (m_plot_per_exact > 0 && m_plot_per_approx > 0) )
        {Abort("Must choose only one of plot_int or plot_per_exact or plot_per_approx");}

    // Which variables to write to plotfile
    pp.query("plt_cell_type",  m_plt_cell_type);
    pp.query("plt_vfrac",      m_plt_vfrac);
    pp.query("plt_afrac",      m_plt_afrac);
    pp.query("plt_volume",     m_plt_volume);
    pp.query("plt_area",       m_plt_area);

    pp.query("plt_velmag",     m_plt_velmag);
    pp.query("plt_velx",       m_plt_velx);
    pp.query("plt_vely",       m_plt_vely);
    pp.query("plt_velz",       m_plt_velz);
    pp.query("plt_p",          m_plt_p);

    pp.query("plt_conc",       m_plt_conc);
    pp.query("plt_soc",        m_plt_soc);

    pp.query("plt_epotL",      m_plt_epotL);
    pp.query("plt_epotS",      m_plt_epotS);
    pp.query("plt_potdiff",    m_plt_potdiff);
    pp.query("plt_overpot",    m_plt_overpot);
    pp.query("plt_curr",       m_plt_curr);
    pp.query("plt_curr_dens",  m_plt_curr_dens);
    pp.query("plt_src",        m_plt_src);

    pp.query("plt_kappaL",     m_plt_kappaL);
    pp.query("plt_gpx",        m_plt_gpx);
    pp.query("plt_gpy",        m_plt_gpy);
    pp.query("plt_gpz",        m_plt_gpz);
    pp.query("plt_rho",        m_plt_rho);
    pp.query("plt_laps",       m_plt_laps);
    pp.query("plt_cnvt",       m_plt_cnvt);
    pp.query("plt_vort",       m_plt_vort);
    pp.query("plt_flxrhs",     m_plt_flxrhs);
    pp.query("plt_divcgpot",   m_plt_divcgpot  );

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Perform initial pressure iterations
void Rincflo::InitialIterations()
{
    #define FUNC_NAME "Rincflo::InitialIterations"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    CopyNewToOld_velocity();
    CopyNewToOld_density();
    CopyNewToOld_conc();

    int initialisation = 1;
    bool explicit_diffusion = (m_diff_type == DiffusionType::Explicit);
    ComputeDt(initialisation, explicit_diffusion);
    if (m_verbose)
        {Print() << format("Doing initial pressure iterations with dt = {:f}.\n", m_dt);}

    auto mac_phi = get_mac_phi();

    for (int lev = 0; lev <= finest_level; ++lev) 
        {m_t_old[lev] = m_t_new[lev];}
    for (int lev = 0; lev <= finest_level; ++lev) 
        {mac_phi[lev]->setVal(0.);}

    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        fillpatch_velocity(lev, m_t_old[lev], m_leveldata[lev]->velocity_o, ng);
        fillpatch_density(lev, m_t_old[lev], m_leveldata[lev]->density_o, ng);
        if (do_AdvectConc()) 
            {fillpatch_conc(lev, m_t_old[lev], m_leveldata[lev]->conc_o, ng);}
    }

    for (int iter = 0; iter < m_initial_iterations; ++iter)
    {
        if (m_verbose) 
            {Print() << format("\nIn initial_iterations: iter = {:d}\n", iter);}

        ApplyPredictor(true);
        CopyOldToNew_velocity();
        CopyOldToNew_density();
        CopyOldToNew_conc();
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Project velocity field to make sure initial velocity is divergence-free
void Rincflo::InitialProjection()
{
    #define FUNC_NAME "Rincflo::InitialProjection"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Real time = 0.0;
    if (m_verbose > 0)
        {Print() << "Initial projection:\n";}

    Real dummy_dt = 1.0;
    bool incremental = false;
    ApplyProjection(get_density_new_const(), m_cur_time, dummy_dt, incremental);   
        
    // We set pressure and gradp back to zero (p0 may still be still non-zero)
    for (int lev = 0; lev <= finest_level; lev++)
    {
        m_leveldata[lev]->pressure.setVal(0.0);
        m_leveldata[lev]->gradp.setVal(0.0);
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
#if (AMREX_USE_EB)
void 
Rincflo::InitialRedistribution()
{
    #define FUNC_NAME "Rincflo::InitialRedistribution"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    if (m_verbose > 1)
        {Print() << "Starting InitialRedistribution() ... \n";}

    // Next we must redistribute the initial solution if we are going to use 
    // MergeRedist or StateRedist redistribution schemes
    if ( m_redistribution_type == "StateRedist" || m_redistribution_type == "MergeRedist" )
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];

            // We use the "old" data as the input here
            // We must fill internal ghost values before calling redistribution
            // We also need any physical boundary conditions imposed if we are
            // calling state redistribution (because that calls the slope routine)

            EB_set_covered(ld.velocity, 0, SpaceDim, ld.velocity.nGrow(), 0.0);
            ld.velocity.FillBoundary(geom[lev].periodicity());
            MultiFab::Copy(ld.velocity_o, ld.velocity, 0, 0, SpaceDim, ld.velocity.nGrow());
            fillpatch_velocity(lev, m_t_new[lev], ld.velocity_o, 3);

            if (!m_constant_density) 
            {
                EB_set_covered(ld.density, 0, 1, ld.density.nGrow(), 0.0);
                ld.density.FillBoundary(geom[lev].periodicity());
                MultiFab::Copy(ld.density_o, ld.density, 0, 0, 1, ld.density.nGrow());
                fillpatch_density(lev, m_t_new[lev], ld.density_o, 3);
            }
            if (do_AdvectConc()) 
            {
                EB_set_covered(ld.conc, 0, m_nspec, ld.conc.nGrow(), 0.0);
                ld.conc.FillBoundary(geom[lev].periodicity());
                MultiFab::Copy(ld.conc_o, ld.conc, 0, 0, 1, ld.conc.nGrow());
                fillpatch_conc(lev, m_t_new[lev], ld.conc_o, 3);
            }

            for (MFIter mfi(ld.density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.validbox();
                auto const& fact = EBFactory(lev);

                EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
                Array4<EBCellFlag const> const& flag = flagfab.const_array();

                if ( (flagfab.getType(grow(bx,1)) != FabType::covered) && (flagfab.getType(grow(bx,1)) != FabType::regular) )
                {
                    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), ccc, vfrac, AMREX_D_DECL(apx, apy, apz);
                    AMREX_D_TERM(
                    fcx = fact.getFaceCent()[0]->const_array(mfi);,
                    fcy = fact.getFaceCent()[1]->const_array(mfi);,
                    fcz = fact.getFaceCent()[2]->const_array(mfi);)
                    ccc = fact.getCentroid().const_array(mfi);
                    AMREX_D_TERM(
                    apx = fact.getAreaFrac()[0]->const_array(mfi);,
                    apy = fact.getAreaFrac()[1]->const_array(mfi);,
                    apz = fact.getAreaFrac()[2]->const_array(mfi);)
                    vfrac = fact.getVolFrac().const_array(mfi);

                    int ncomp = SpaceDim;
                    redistribution::RedistributeInitialData(
                        bx,ncomp, ld.velocity.array(mfi), ld.velocity_o.array(mfi), flag, 
                        AMREX_D_DECL(apx, apy, apz), 
                        vfrac, 
                        AMREX_D_DECL(fcx, fcy, fcz), 
                        ccc,geom[lev],m_redistribution_type);
                    if (!m_constant_density)
                    {
                        ncomp = 1;
                        redistribution::RedistributeInitialData(
                            bx,ncomp, ld.density.array(mfi), ld.density_o.array(mfi), flag, 
                            AMREX_D_DECL(apx, apy, apz), 
                            vfrac, 
                            AMREX_D_DECL(fcx, fcy, fcz), 
                            ccc,geom[lev],m_redistribution_type);
                    }
                    if (do_AdvectConc()) 
                    {
                        ncomp = m_nspec;
                        redistribution::RedistributeInitialData(
                            bx, ncomp, ld.conc.array(mfi), ld.conc_o.array(mfi), flag, 
                            AMREX_D_DECL(apx, apy, apz), 
                            vfrac, 
                            AMREX_D_DECL(fcx, fcy, fcz), 
                            ccc,geom[lev],m_redistribution_type);
                    }
                }
            }

            // We fill internal ghost values after calling redistribution
            ld.velocity.FillBoundary(geom[lev].periodicity());
            ld.density.FillBoundary(geom[lev].periodicity());
            ld.conc.FillBoundary(geom[lev].periodicity());
        }
    }

    if (m_verbose > 1)
        {Print() << "Completed Initial redistribution()\n";}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
#endif

// *********************************************************************************************************************
void Rincflo::PrintFluidModelSummary( )
{
    PrintStars();
    Print() << "Parameters for Fluid Flow:\n";

    // The diffusion type; this is always relevant even if we're doing only a reaction with no fluid flow
    string diff_type_str {""};
    switch (m_diff_type)
    {
        case Rincflo::DiffusionType::Invalid:
            diff_type_str = "Invalid";
            break;
        case Rincflo::DiffusionType::Explicit:
            diff_type_str = "Explicit";
            break;
        case Rincflo::DiffusionType::Crank_Nicolson:
            diff_type_str = "Crank_Nicolson";
            break;
        case Rincflo::DiffusionType::Implicit:
            diff_type_str = "Implicit";
    } 
    Print() << format("Diffusion type: {:s}.\n", diff_type_str.c_str());

    // Report the diffusion coefficients
    Print() << "Scalar diffusion coefficients " << std::endl;
    for (int i = 0; i < m_nspec; i++) 
        {Print() << format("    Species {:d}: {}\n", i, m_D_s[i]);}

    // Summarize the fluid flow if applicable
    if (m_do_flow)
    {
        Print() << format("const density: {}\n", m_constant_density);
        Print() << format("ro_0: {}.\n", m_ro_0);
        Print() << format("mu: {}.\n", m_mu);
        Print() << format("advect_conc: {}.\n", m_advect_conc);
        Print() << format("test_species_conservation: {}.\n", m_test_species_conservation);
        Print() << format("need_DivTau? {}.\n", need_DivTau());
        Print() << format("solve until steady_state? {}.\n", m_steady_state);
        if(m_steady_state) 
        { 
            string tol_str {""};
            if(m_tol_flow_relativerate) 
                {tol_str = "||V-Vo||/||Vo||*dt";}
            else if (m_tol_flow_rate_comp) 
                {tol_str = "max(|du|+|dv|+|dw|)/dt";}
            else if (m_tol_flow_rate_mag) 
                {tol_str = "||V-Vo||/dt";}
            else if (m_tol_flow_relative) 
                {tol_str = "||V-Vo||/||Vo||";}
            Print() << format("tol = {} criterion based on {}.\n", m_steady_state_tol, tol_str.c_str());
        }
    }
}
