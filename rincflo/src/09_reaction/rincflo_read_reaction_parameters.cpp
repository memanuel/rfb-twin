#include <rincflo.H>

void Rincflo::ReadReactionParameters()
{
    #define FUNC_NAME "Rincflo::ReadReactionParameters"
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Reaction parameters with prefix Rincflo
    ParmParse pp("rincflo");

    // Read the reaction model as a string
    m_reaction_model = ReactionModel::none;
    std::string reaction_model_s = "none";
    pp.query("reaction_model", reaction_model_s);

    // Convert the reaction model from string to enum
    if(reaction_model_s == "none")
        {m_reaction_model = ReactionModel::none;}    
    else if(reaction_model_s == "Nernst")
        {m_reaction_model = ReactionModel::Nernst;}
    else if(reaction_model_s == "SpecialRedox")
        {m_reaction_model = ReactionModel::SpecialRedox;}
    else if(reaction_model_s == "ButlerVolmer")
        {m_reaction_model = ReactionModel::ButlerVolmer;}
    else
        {Abort("Unknown reaction_model! Choose either none, linear, Nernst, SpecialRedox, ButlerVolmer ...");}

    // If there is no reaction model, quit early.
    if(m_reaction_model == ReactionModel::none)
    {
        Print() << "No reaction model specified; skip parsing of reaction parameters.\n";
        return;
    }

    // Check that cell spacing is uniform (i.e. grid cells are squares / cubes)
    // This is required b/c source term calculation assumes cell area / cell volume = afrac / vfrac * dx_inv
    // See Rincflo::CalcSourceTerm()
    #if (AMREX_USE_EB)
    AMREX_D_TERM(
    const Real dx = Geom(0).CellSize(0);,
    const Real dy = Geom(0).CellSize(1);,    
    const Real dz = Geom(0).CellSize(2);)

    // The absolute tolerance for spacing differences
    const Real spacing_min = min(AMREX_D_DECL(dx, dy, dz));
    const Real spacing_max = max(AMREX_D_DECL(dx, dy, dz));

    // Is the spacing uniform, i.e. are dx, dy, and dz suitably close?
    bool spacing_is_uniform = amrex::almostEqual(spacing_min, spacing_max);
    // The spacing must be uniform; this is imposed upstream by AMReX in AMReX_EB_Utils.cpp / apply_eb_redistribution()
    if (!spacing_is_uniform)
    {
        std::string msg = AMREX_D_PICK(
            "",
            format("Error! Grid spacing must be uniform. dx={:f}, dy={:f}.\n", dx, dy),
            format("Error! Grid spacing must be uniform. dx={:f}, dy={:f}, dz={:f}.\n", dx, dy, dz)
        );
        Abort(msg);
    }
    #endif

    // Regridding during reaction
    pp.query("regrid_react_int", m_regrid_react_int);

    // Physics/reaction models options
    pp.query("electromigration", m_electromigration);

    // Initial proton concentration (as reference level for overpotential calculation)
    pp.query("init_protons", m_init_protons);  
    pp.query("fix_protons_mem", m_fix_protons_mem);
    pp.query("mass_transfer", m_mass_transfer);

    // The equilibrium state of charge from the Nernst equation
    Real soc_eq = 0.0;

    // Read in the reaction model parameters
    switch (m_reaction_model)
    {
        case ReactionModel::Nernst:
            // Check number of species
            if(m_nspec != 2) 
                {Abort("We need exactly two species for Nernst model");}
            // Rate constant (used to convert source term back to overpotential)
            pp.query("k_0", m_k_0);
            // Applied potential in volts
            pp.query("DVapp", m_DVapp);
            // Electrons exchanged in the redox reaction
            pp.query("n_electrons", m_n_electrons);
            // Relaxation parameter for updating overpot in the Nernst model
            pp.query("nernst_omega", m_nernst_omega);
            // Should we use explicit diffusion for Nernst model time stepping?
            pp.query("nernst_explicit_diffusion", m_nernst_explicit_diffusion);
            break;

        case ReactionModel::SpecialRedox:
            // Check number of species
            if(m_nspec<2) 
                {Abort("We need at least two species for SpecialRedox model");}     
            // Rate constant
            pp.query("k_0", m_k_0);
            // Applied potential in volts
            pp.query("DVapp", m_DVapp);
            // Electrons exchanged in the redox reaction
            pp.query("n_electrons", m_n_electrons);
            // Simulate electromigration?
            m_electromigration = false;
            // Set the flag for symmetric charge transfer
            m_same_alpha = true;
            break;

        case ReactionModel::ButlerVolmer:
            // Check number of species
            if(m_nspec<2) 
                {Abort("We need at least two species for ButlerVolmer model.");}      
            // Rate constant
            pp.query("k_0", m_k_0);
            // Applied potential in volts; net reducing voltage with zero volts = 0.5 SOC at equilibrium
            pp.query("DVapp", m_DVapp);

            // Applied voltage on the membrane is the boundary condition for epotL for the bottom (zlo)
            ParmParse ppm("zlo");
            ppm.query("epotL", m_V_membrane);
            // Applied voltage on the electrode; calculated from DVapp and V_membrane
            m_V_electrode = m_V_membrane - m_DVapp;

            // Check that both of these applied voltages are set
            if (isnan(m_V_electrode) || isnan(m_V_membrane))
                {Abort("ButlerVolmer expects V_electrode and V_membrane = zlo.epotL to be set.");}

            // Electrons exchanged in the redox reaction
            pp.query("n_electrons", m_n_electrons);
            // Protons generated in the redox reaction (becomes source prefactor for species 3)
            pp.query("n_protons", m_n_protons);
            // Charge transfer coeff species 0
            pp.query("alpha_0", m_alpha_0);
            // Charge transfer coeff species 1
            pp.query("alpha_1", m_alpha_1);
            // Check that alphas sum to 1 in BV
            if(std::fabs(m_alpha_0 + m_alpha_1 - 1.0) > 1.0E-10) 
                {Abort("ButlerVolmer expects alpha0 + alpha1 = 1.0.");}
            // Set the flag for symmetric charge transfer
            m_same_alpha = (std::fabs(m_alpha_0 - m_alpha_1) < 1.0E-6);
            break;
    }

    // Product RT
    static constexpr Real RT_const = m_R_const * m_T_const;

    // Calculate constants that depend on the number of electrons transferred
    m_nF_const = m_n_electrons * m_F_const;
    m_RT_over_nF_const = RT_const / m_nF_const; // in V
    m_nF_over_RT_const = m_nF_const / RT_const; // in 1/V

    // Compute the equilibrium SOC and bounds on it when applicable (models with an applied voltage difference)
    switch (m_reaction_model)
    {
        case ReactionModel::Nernst:
        case ReactionModel::SpecialRedox:
        case ReactionModel::ButlerVolmer:
            // Default to running for 2.0 times the time it takes for reactant to flow across domain
            m_t_min = 2.0 * m_t_flow;
            // Optional override to the convergence criterion that simulation time is at least 1.5x t_flow
            pp.query("t_min", m_t_min);
            // Report flow time
            Print() << format("Time for reactant to flow across domain {:5.3f}. Minimum time = {:5.3f} sec.\n", m_t_flow, m_t_min);

            // Total concentration of oxidized plus reducied species in millimoloar; this is conserved
            m_conc_redox_tot = m_ic_t[0] + m_ic_t[1];

            // Prefactor for voltages in exponential terms; in units of 1/V
            m_nF_over_RT_const = (m_F_const*m_n_electrons)/(m_R_const*m_T_const);
            // Relative concentration of reduced species to oxidized species at equilibrium per Nernst equation
            soc_eq = 1.0 / (1.0 + exp(-m_nF_over_RT_const * m_DVapp));

            // Minimum SOC over the whole simulation
            // Real x_min = exp(m_nF_over_RT_const * m_E_0);
            // m_soc_min = x_min / (1.0 + x_min);
            m_soc_min = m_ic_t[1] / m_conc_redox_tot;
            // Minimum concentration of reduced species
            m_conc_red_min = m_conc_redox_tot * m_soc_min;
            // Maximum concentration of oxidized species
            m_conc_ox_max = m_conc_redox_tot * (1.0 - m_soc_min);

            // Maximum SOC over the whole simulation is the same as the Nernst equilibrium concentration
            m_soc_max = soc_eq;
            // Maximum concentration of reduced species
            m_conc_red_max = m_conc_redox_tot * m_soc_max;
            // Minimum concentration of oxidized species
            m_conc_ox_min = m_conc_redox_tot * (1.0 - m_soc_max);
            break;
    }
  
    // Charge of each species
    m_z_s.resize(m_nspec, 0.0);
    pp.queryarr("z_s", m_z_s, 0, m_nspec);

    // Properties of "extra" species (their transport is not simulated explicitly)
    pp.query("D_ex", m_D_ex);
    if(m_D_ex > 0.0)
        {m_extra_species = true;}    
    pp.query("z_ex", m_z_ex);
    
    // Conductivity solid
    pp.query("kappa_s", m_kappa_s);

    // Conductivity of liquid
    pp.query("kappa_l", m_kappa_l);

    // Options to solve epotS
    pp.query("solve_epotS", m_solve_epotS);
    
    // Option to include concentration-dependent ionic flux in epotL equation
    pp.query("calc_flux_epotL", m_calc_flux_epotL);

    // Membrane options, compensate current flowing out of cell by updating b.c. at membrane
    m_membc_type = MembraneBCType::none;
    std::string membc_type_s;
    pp.query("membc_type", membc_type_s);
    if(membc_type_s == "potential")
        {m_membc_type = MembraneBCType::potential;}
    else if(membc_type_s == "current")
        {m_membc_type = MembraneBCType::current;}

    // Write out status message
    PrintStars();
    switch (m_reaction_model)
    {
        case ReactionModel::Nernst:
            Print() << "Nernst model parameters:\n";
            Print() << format("Applied Voltage    = {:+8.2f} millivolts\n",  m_DVapp*1000.0);
            Print() << format("k_0                = {:8.1e}\n",  m_k_0);
            Print() << format("n_electrons        = {:d}\n", m_n_electrons);
            Print() << format("m_conc_redox_tot   = {:6.3f} millimolar.\n", m_conc_redox_tot);
            Print() << format("Minimum SOC        = {:8.3e}\n", m_soc_min);
            Print() << format("Maximum SOC        = {:8.3e}\n", m_soc_max);
            Print() << format("Relaxation omega   = {:8.6f}\n", m_nernst_omega);
            Print() << format("Explicit Diffusion = {}\n", m_nernst_explicit_diffusion);
            break;

        case ReactionModel::SpecialRedox:
            Print() << "Special Redox model parameters:\n";
            Print() << format("Applied Voltage  = {:+8.2f} millivolts\n",  m_DVapp*1000.0);
            Print() << format("k_0              = {:6.1e}\n",  m_k_0);
            Print() << format("n_electrons      = {:d}\n",     m_n_electrons);
            Print() << format("m_conc_redox_tot = {:6.3f} millimolar.\n", m_conc_redox_tot);
            Print() << format("Minimum SOC      = {:8.3e}\n", m_soc_min);
            Print() << format("Maximum SOC      = {:8.3e}\n", m_soc_max);
            break;

        case ReactionModel::ButlerVolmer:
            Print() << "ButlerVolmer model parameters:\n";
            Print() << format("Applied Voltage  = {:+8.2f} millivolts\n",  m_DVapp * 1000.0);
            Print() << format("V_membrane       = {:+8.2f} millivolts\n",  m_V_membrane * 1000.0);
            Print() << format("V_electrode      = {:+8.2f} millivolts\n",  m_V_electrode * 1000.0);
            Print() << format("k_0              = {:6.1e}\n",  m_k_0);
            Print() << format("n_electrons      = {:d}\n",     m_n_electrons);
            if(m_nspec>2) 
                {Print() << format("n_protons        = {:d} (prefactor for species 3)\n", m_n_protons);}
            Print() << format("m_conc_redox_tot = {:6.3f} millimolar.\n", m_conc_redox_tot);
            Print() << format("alpha_0          = {:8.6f}\n",  m_alpha_0);
            Print() << format("alpha_1          = {:8.6f}\n",  m_alpha_1);
            Print() << format("Minimum SOC      = {:8.3e}\n", m_soc_min);
            Print() << format("Maximum SOC      = {:8.3e}\n", m_soc_max);

            if (m_membc_type == MembraneBCType::potential)
                {Print() << "Membrane BC: potential\n";}
            if (m_membc_type == MembraneBCType::current)
                {Print() << "Membrane BC: current\n";}
            Print() << format("Electromigration = {:d}.\n", do_Electromigration());
            Print() << format("Mass transfer    = {:d}.\n", m_mass_transfer);
            for (int i=0; i<m_nspec ; i++)
            {
                string extra_msg = "";
                if (i == 0) {extra_msg = " (reduced)";}
                if (i == 1) {extra_msg = " (oxidized)";}
                Print() << format("Charge of species {:d} is {:d} {:s}.\n", i, m_z_s[i], extra_msg);
            }
        
            if(m_extra_species) 
            {
                Print() << format("Diffusivity of extra species = {:8.3e}.\n", m_D_ex);
                Print() << format("Charge of extra species = {:d}.\n", m_z_ex );;
            }

            if(m_kappa_l > 0.0) 
                {Print() << format("Conductivity of liquid = {:8.3f}.\n", m_kappa_l);}
            if(!m_calc_flux_epotL)
                {Print() << "Do NOT calculate flux term in epotL eq.\n"; }
            if(!m_solve_epotS) 
                {Print() << format("Do NOT solve potential in the solid; fixed value V_electrode = {:+8.2f} millivolts.\n", 
                                    m_V_electrode * 1000.0);}
            else 
                {Print() << format("Conductivity of solid = {:6.3f}.\n", m_kappa_s);}

            if (m_init_protons > 0.0)
                {Print() << format("Initial proton concentration = {:8.3f}. Use this to adjust overpotential.\n", m_init_protons);}
            else
                {Print() << "Do NOT adjust overpotential for proton concentration.\n";}
            if (m_fix_protons_mem)
                {Print() << "Fix proton concentration at membrane.\n";}
            else
                {Print() << "Do NOT fix proton concentration at membrane.\n";}
            if (m_same_alpha)
                {Print() << "alpha_0 = alpha_1 = 0.5, so we will simplify BV exp to 2 sinh (eta/2).\n";}
            break;
    }

    // Restarting options
    PrintStars();
    Print() << format("Restarting options:\n");
    Print() << format("Restart file: {:s}\n", m_restart_file.c_str());
    // Reinitializion of concentrations
    pp.query("react_reinit_all_conc", m_react_reinit_all_conc);
    Print() << format("reinitialize all concentrations: {}.\n", m_react_reinit_all_conc);
    pp.query("react_reinit_some_conc", m_react_reinit_some_conc);
    Print() << format("reinitialize some concentrations: {}.\n", m_react_reinit_some_conc);
    pp.query("react_reinit_conc_from", m_react_reinit_conc_from);
    if (m_react_reinit_some_conc)
        {Print() << format("react_reinit_conc_from: {:d}.\n", m_react_reinit_conc_from);}
    m_react_init_t.resize(m_nspec, 0.0);
    pp.queryarr("react_init_t", m_react_init_t, 0, m_nspec);
    if(m_react_reinit_all_conc)
    {
        for (int i = 0; i < m_nspec ; ++i)    
            {Print() << format("Reinit for species {:d} concentration is {:15.12f} mM.\n", i, m_react_init_t[i]);}
    }

    // Reinitialization of potential
    pp.query("react_reinit_epotS", m_react_reinit_epotS);
    pp.query("react_reinit_epotL", m_react_reinit_epotL);
    if(m_reaction_model == ReactionModel::ButlerVolmer &&  (m_react_reinit_epotS || m_react_reinit_epotL) )   
    {
        Print() << format("Reinit for epotS? {:s}\n", (m_react_reinit_epotS ? "yes" : "no"));
        Print() << format("Reinit for epotL? {:s}\n", (m_react_reinit_epotL ? "yes" : "no"));
    }

    // Initialization of reaction step counter
    pp.query("react_zero_step_counter", m_react_zero_step_counter);
    Print() << format("zero step counter? {}.\n", m_react_zero_step_counter);
    pp.query("react_init_step_counter", m_react_init_step_counter);
    if (m_react_init_step_counter > 0)
        {Print() << format("reset step counter to {:d}.\n", m_react_init_step_counter);}
    // Check consistency of reaction step counter
    if (m_react_zero_step_counter && (m_react_init_step_counter > 0))
        {Abort("Error on reaction counter intialization! Must specify at most one of react_zero_step_counter and react_init_step_counter");}
  
    // Time-stepping, max number of steps, and steady-state definition
    pp.query("cfl_react", m_cfl_react);
    pp.queryarr("cfl_react_sched", m_cfl_react_sched, 0, 2);
    // CFL must be at most one
    if ((m_cfl_react > 1.0))
        {Abort("We require cfl_react <= 1.0");}

    // Set flags describing reaction time stepping
    m_using_cfl_react_single = (m_cfl_react>0.0);
    m_using_cfl_react_sched = (m_cfl_react_sched[0]>0.0) && (m_cfl_react_sched[1]>0.0);
    m_using_cfl_react = m_using_cfl_react_single || m_using_cfl_react_sched;

    // Revise default parameter values for reactant consumption in the Nernst model
    if (m_reaction_model == ReactionModel::Nernst)
    {
        m_reactant_consumption_max = 1.0;
        m_reactant_consumption_max_limit = 1.0;
    }

    // The type of time stepping and range of time steps
    pp.query("react_timestepping_method", m_react_timestepping_method);
    pp.query("min_rsteps", m_min_rsteps);
    pp.query("max_rsteps", m_max_rsteps);
    pp.query("react_tol_soc", m_react_tol_soc);
    pp.query("react_tol_epot", m_react_tol_epot);
    pp.query("react_tol_flux", m_react_tol_flux);
    pp.query("react_tol_mt", m_react_tol_mt);
    pp.query("overpot_max", m_overpot_max);
    pp.query("reactant_consumption_max", m_reactant_consumption_max);
    pp.query("reactant_consumption_max_limit", m_reactant_consumption_max_limit);
    
    // Maximum number of stagnant time steps before terminating early
    pp.query("stagnant_steps_max", m_stagnant_steps_max);

    // Test for consistency in reaction time stepping
    if (m_using_cfl_react && m_using_fixed_dt)
        {Abort("specified both cfl_react and fixed_dt (either single or schedule).");}
    if (m_using_cfl_react_single && m_using_cfl_react_sched)
        {Abort("specified both a single cfl_react and a cfl_react schedule.");}      
    
    // Print info
    PrintStars();
    Print() << "Reaction configuration\n";
    Print() << format("react_timestepping_method {}.\n", m_react_timestepping_method);
    if (m_using_cfl_react_single)
        {Print() << format("single CFL {:0.6f}.\n", m_cfl_react);}
    if (m_using_cfl_react_sched)
        {Print() << format("CFL schedule {:0.6f} - {:0.6f}.\n", m_cfl_react_sched[0], m_cfl_react_sched[1]);}
    if (m_using_fixed_dt_single)
        {Print() << format("single fixed_dt {:6.3e}.\n", m_fixed_dt);}
    if (m_using_fixed_dt_sched)
        {Print() << format("fixed_dt schedule {:6.3e} - {:6.3e}.\n", m_fixed_dt_sched[0], m_fixed_dt_sched[1]);}
    Print() << format("overpot_max {:+8.2f} millivolts.\n", m_overpot_max * 1000.0);
    Print() << format("reactant_consumption_max {:10.8f} = 1/{:.1f}.\n", 
                    m_reactant_consumption_max, std::round(1.0 / m_reactant_consumption_max));
    Print() << format("reactant_consumption_max_limit {:10.8f} = 1/{:.1f}.\n", 
                    m_reactant_consumption_max_limit, std::round(1.0 / m_reactant_consumption_max_limit));
    if (m_min_rsteps > 0)
        {Print() << format("min_rsteps {:d}.\n", m_min_rsteps);}
    if (m_max_rsteps > 0)
        {Print() << format("max_rsteps {:d}.\n", m_max_rsteps);}
    Print() << format("steady-state tolerances:");
    if (m_react_tol_soc > 0.0)
        {Print() << format("concentration: {:6.3e}.\n", m_react_tol_soc);}
    if (m_reaction_model == ReactionModel::ButlerVolmer && m_react_tol_epot > 0.0)
        {Print() << format("Epot: {:6.3e}.\n", m_react_tol_epot);}
    if (m_react_tol_flux > 0.0)
        {Print() << format("(current - net flux) / current: {:6.3e}.\n", m_react_tol_flux);}
    if (m_reaction_model == ReactionModel::Nernst && m_react_tol_mt > 0.0)
        {Print() << format("(current_bv - current_mt) / current_bv: {:6.3e}.\n", m_react_tol_mt);}

    // The type of convergence test
    string tol_type_soc_s = "abs_rate";
    pp.query("tol_type_soc", tol_type_soc_s);

    if (tol_type_soc_s == "abs_step")
        {m_tol_type_soc = ToleranceType::abs_step;}
    else if (tol_type_soc_s == "abs_rate")
        {m_tol_type_soc = ToleranceType::abs_rate;}
    else
        {Abort("Unknown tol_type_soc! Choose either abs_step, abs_rate...");}

    switch (m_tol_type_soc)
    {
        case ToleranceType::abs_step:
            Print() << "SOC convergence type: RMS(Delta SOC) < tol\n"; 
            break;
        case ToleranceType::abs_rate:
            Print() << "SOC convergence type: RMS(Delta SOC) * (tau / dt) < tol\n"; 
            break;
    }

    // plotting / saving options
    pp.query("react_plot_int", m_react_plot_int);
    pp.query("react_check_int", m_react_check_int);
    pp.query("react_int_sec", m_react_int_sec);

    // non-linearity (and relaxation) options  
    pp.query("react_nonlin_method", m_react_nonlin_method);
    pp.query("fixpoint_conc_tol", m_fixpoint_conc_tol);
    pp.query("fixpoint_epot_tol", m_fixpoint_epot_tol);
    pp.query("fixpoint_max_iter", m_fixpoint_max_iter);
    pp.query("fixpoint_omega", m_fixpoint_omega);

    // Report additional reaction options
    if (m_reaction_model==ReactionModel::SpecialRedox || m_reaction_model==ReactionModel::ButlerVolmer)
    {
        Print() << "Nonlinear methods available: 0: linear, 1: picard, 2: newton; default 1.\n";
        Print() << format("Nonlinear method selected: {:d}.\n", m_react_nonlin_method);
        Print() << format("Fixed point tolerance on conc: {:5.2e}.\n", m_fixpoint_conc_tol);
    }
    if (m_reaction_model==ReactionModel::ButlerVolmer)
        {Print() << format("Fixed point tolerance on Epot: {:5.2e}.\n", m_fixpoint_epot_tol);}

    if(m_fixpoint_omega < 1.0) 
    {
        m_fixpoint_relax = true;
        Print() << format("Using relaxation with fixpoint_omega: {}.\n", m_fixpoint_omega);
    }
    
    // End of status for reaction setup
    Print() << "********************************************************************************\n";

    // Read in eb2 parameters
    ParmParse pp_eb2("eb2");
    // the eb2 threshold for a cell with a small volume fraction
    pp_eb2.query("small_volfrac", m_eb2_small_volfrac);
    // apply corresponding update to m_eb2_large_volfrac
    m_eb2_large_volfrac = 1.0 - m_eb2_small_volfrac;

    // allocate space for prefactor of source term for each species
    m_s_pref_conc.resize(m_nspec, 0.0);
  
    // initialize prefactors depending on operating conditions
    switch (m_reaction_model)
    {
        case ReactionModel::Nernst:
        case ReactionModel::SpecialRedox:
            // The oxidized species is consumed (1 per reaction)
            m_s_pref_conc[0] = -1.0;
            // The reduced species is produced (1 per reaction)
            m_s_pref_conc[1] = 1.0;
            break;

        case ReactionModel::ButlerVolmer:
            // The oxidized species is consumed (1 per reaction)
            m_s_pref_conc[0] = -1.0;
            // The reduced species is produced (1 per reaction)
            m_s_pref_conc[1] = 1.0;
            // The number of protons in the reaction is an input; convert from int to Real
            if( m_nspec > 2 )
                {m_s_pref_conc[2] = m_n_protons*1.0;}
            break;
    }

    // Intialize auxiliary constants for special redox model
    if(m_reaction_model == ReactionModel::SpecialRedox)
    {
        m_nF_over_RT_const = (m_F_const*m_n_electrons)/(m_R_const*m_T_const); //in 1/V
    }

    // Initialize auxiliary constants for ButlerVolmer model
    if(m_reaction_model == ReactionModel::ButlerVolmer)   
    {
        m_s_pref_epotL =  m_nF_const;       // -kappa div^2 pot = pref*source (+ flx...)
        m_s_pref_epotS = -m_nF_const;       // -kappa div^2 pot = pref*source

        m_Dz.resize(m_nspec, 0.0);
        for (int n=0; n < m_nspec; ++n) 
            {m_Dz[n]=m_D_s[n] * m_z_s[n];}
    
        m_Dzsq.resize(m_nspec, 0.0);
        for (int n=0; n < m_nspec; ++n) 
            {m_Dzsq[n]=m_D_s[n] * m_z_s[n] * m_z_s[n];}
    
        if(m_extra_species) 
        {
            m_Dz_ex.resize(m_nspec, 0.0);
            for (int n=0; n < m_nspec; ++n) 
                {m_Dz_ex[n] = m_z_s[n] * (m_D_s[n] - m_D_ex);}
        
            m_Dzsq_ex.resize(m_nspec, 0.0);
            for (int n=0; n < m_nspec; ++n) 
                {m_Dzsq_ex[n]=m_z_s[n] * (m_z_s[n]*m_D_s[n] - m_z_ex*m_D_ex);}

            m_zDFoverRT.resize(m_nspec, 0.0);
            for (int n=0; n < m_nspec; ++n) 
                {m_zDFoverRT[n]= (m_D_s[n] * m_z_s[n] * m_F_const) / (m_R_const*m_T_const);}
        }

        // Check if all species have zero charge, stuff still makes sense
        bool all_zero = true;
        for (int n=0;n<m_nspec;n++) 
        {
            if(m_z_s[n] != 0) 
                {all_zero = false;}
        }
        if (m_extra_species==true && m_z_ex != 0) 
            {all_zero = false;}
        if (all_zero) 
        {
            if(m_calc_flux_epotL == true) 
            {
                m_calc_flux_epotL = false;
                Print() << "All charges are zero, so we will not calculate concentration-dependent flux in epotL eq.\n";    
            }
            if (m_kappa_l<0.0) 
                {Abort("All charges are zero, but effective conductivity needs to be calculated");}
        }
    }  

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
