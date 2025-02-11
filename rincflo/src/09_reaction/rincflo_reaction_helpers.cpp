#include <rincflo.H>

// *********************************************************************************************************************
// Self contained helper functions for reaction calculations.
// *********************************************************************************************************************

// *********************************************************************************************************************
// from new to old (previous timestep)                                    //
// *********************************************************************************************************************
void Rincflo::StoreQuantities()
{
    #define FUNC_NAME "Rincflo::StoreQuantities"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Concentrations (conc)
    CopyNewToOld_conc();
    CopyNewToOld_soc();
    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev)
        {fillpatch_conc(lev, m_t_old[lev], m_leveldata[lev]->conc_o, ng);}

    // Electric potential
    if (m_reaction_model == ReactionModel::ButlerVolmer)
    {
        // copy old epot
        CopyNewToOld_epotL();
        if (m_solve_epotS)
            {CopyNewToOld_epotS();}
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            fillpatch_epotL(lev, m_t_old[lev], m_leveldata[lev]->epotL_o, ng);
            if (m_solve_epotS)
                {fillpatch_epotS(lev, m_t_old[lev], m_leveldata[lev]->epotS_o, ng);}
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// from old to new (for undoing a failed reaction timestep)               //
// *********************************************************************************************************************
void Rincflo::StoreQuantities_undo()
{
#define FUNC_NAME "Rincflo::StoreQuantities_undo"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Concentrations (conc)
    CopyOldToNew_conc();
    CopyOldToNew_soc();
    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev)
        {fillpatch_conc(lev, m_cur_time, m_leveldata[lev]->conc_o, ng);}

    // Electric potential
    if (m_reaction_model == ReactionModel::ButlerVolmer)
    {
        // copy old epot
        CopyOldToNew_epotL();
        if (m_solve_epotS)
            {CopyOldToNew_epotS();}
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            fillpatch_epotL(lev, m_cur_time, m_leveldata[lev]->epotL_o, ng);
            if (m_solve_epotS)
                {fillpatch_epotS(lev, m_cur_time, m_leveldata[lev]->epotS_o, ng);}
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// compute time-step to use in the reaction part                          //
// *********************************************************************************************************************
void Rincflo::ComputeDtReaction()
{
    #define FUNC_NAME "Rincflo::ComputeDtReaction"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // This should not be called when using the Nernst model, which has a special logic flow
    if (m_reaction_model == Rincflo::ReactionModel::Nernst)
        {Abort("ComputeDtReaction() should not be called when using the Nernst model. Use ComputeDtNernst() instead.\n");}

    // Store the past two dt
    m_prev_prev_dt = m_prev_dt;
    m_prev_dt = m_dt;
    // The default value for m_stop_time translates to infinity
    const Real inf = std::numeric_limits<Real>::infinity();    
    // The time step is at most the time remaining to the fixed end time
    Real dt_to_stop = (m_stop_time > 0) ? m_stop_time - m_cur_time : inf;

    // Expedited process when using a single fixed time step
    if (m_using_fixed_dt_single)
    {
        // Set the time step and time and return early
        m_dt = min(m_fixed_dt, dt_to_stop);
        UpdateTime();
        return;
    }

    // Are we using any kind of a schedule for the reaction time stepping?
    bool using_react_sched = m_using_fixed_dt_sched || m_using_cfl_react_sched;

    // Get the interpolated position beta in the schedule
    // The weights are alpha on the start of the schedule, beta on the end of the schedule
    Real sched_alpha{0.0}, sched_beta{0.0};
    if (using_react_sched)
    {
        Real s{Real(m_rstep)};
        Real s0{Real(m_start_rsteps)};
        Real s1{Real(m_max_rsteps)};
        sched_beta = (s - s0) / (s1 - s0);
        sched_alpha = 1.0 - sched_beta;
    }

    // If we are using a fixed_dt schedule, we can return early without expensive CFL calculations
    if (m_using_fixed_dt_sched)
    {
        // Do a geometric interpolation of m_fixed_dt
        m_fixed_dt = std::pow(m_fixed_dt_sched[0], sched_alpha) * std::pow(m_fixed_dt_sched[1], sched_beta);
        // Set the time step and time and return early
        m_dt = min(m_fixed_dt, dt_to_stop);
        UpdateTime();
        return;
    }

    // The derived fields must be up to date; this is done by the calling function

    // If we are using a cfl_react schedule, do a geometric interpolation of m_cfl_react here
    // We still need to update the CFL calculation below
    if (m_using_cfl_react_sched)
        {m_cfl_react = std::pow(m_cfl_react_sched[0], sched_alpha) * std::pow(m_cfl_react_sched[1], sched_beta);}

    // Is diffusion explicit?
    bool explicit_diffusion = (m_react_timestepping_method == 0);

    // Initialize CFL time scales for convection, diffusion, and reaction
    Real conv_cfl = 0.0;
    Real diff_cfl = 0.0;
    Real react_cfl = 0.0;

    // Maximum timestep that will consume at most a set fraction of reactant in any cell
    Real dt_max_react = dt_to_stop;
    // Are we covering multiply cut cells?
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    // Go through the levels and do calculations for CFL
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Extract information about this level
        auto const dx_inv = geom[lev].InvCellSizeArray();
        MultiFab const &vel = m_leveldata[lev]->velocity;
        MultiFab const &conc = m_leveldata[lev]->conc;
        MultiFab const &source = m_leveldata[lev]->source;

        // Embedded boundary and related
        const auto &fact = EBFactory(lev);
        const auto &flags = fact.getMultiEBCellFlagFab();

        // Get contribution to CFL from advection
        Real conv_lev = 0.0;
        auto func_max = 
        [=] AMREX_GPU_HOST_DEVICE(
        Box const &b, Array4<Real const> const &v, Array4<EBCellFlag const> const &f) -> Real
        {
            Real mx = -1.0;
            auto func_b = 
            [=, &mx](int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                {
                    mx = max(AMREX_D_TERM(
                          std::abs(v(i,j,k,0))*dx_inv[0], 
                        + std::abs(v(i,j,k,1))*dx_inv[1], 
                        + std::abs(v(i,j,k,2))*dx_inv[2])
                        , mx);
                }
            };
            Loop(b, func_b);
            return mx;
        };
        conv_lev = ReduceMax(vel, flags, 0, func_max);

        // Cumulative conv_cfl term over levels
        conv_cfl = max(conv_cfl, conv_lev);

        // Get contribution to CFL from explicit diffusion if applicable
        Real diff_lev = 0.0;
        if (explicit_diffusion)
        {
            Real max_diff_coeff = 0.0;
            for (int n = 0; n < m_nspec; n++)
            {
                max_diff_coeff = max(max_diff_coeff, m_D_s[n]);
            }

            diff_lev = max_diff_coeff * 2.0 * (AMREX_D_TERM(dx_inv[0] * dx_inv[0], +dx_inv[1] * dx_inv[1], +dx_inv[2] * dx_inv[2]));
            // Cumulative diff_cfl term over levels
            diff_cfl = max(diff_cfl, diff_lev);
        }

        auto func_min = 
        [=, this] 
        AMREX_GPU_DEVICE(Box const &b, Array4<Real const> const &conc,       
        Array4<Real const> const &source, Array4<EBCellFlag const> const &flags) -> Real
        {
            Real dt_react = inf;
            auto func_b = 
            [=, &dt_react, this](int i, int j, int k) noexcept
            {
                // Only need to check boundary cells; boundary = not (regular or covered)
                if (flags(i,j,k).isBoundary(cover_multiple_cuts))
                {
                    // The maximum time step in this cell based on max allowed consumption of either reactant
                    Real dt_react_step = -m_reactant_consumption_max *
                        min(conc(i,j,k,0) / m_s_pref_conc[0] / source(i,j,k), 
                            conc(i,j,k,1) / m_s_pref_conc[1] / source(i,j,k));

                    // Account for possible rapid change in the BV prefactor due to the exchange current j0
                    Real s = conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));   // state of charge
                    // Only apply the adjustment if the state of charge is small
                    if (s < 1.0E-4)
                    {
                        // The targeted state of charge at the end of the time step
                        Real t = s + m_reactant_consumption_max;
                        // j0 is proportional to sqrt(s * (1-s)) at the start and sqrt(t * (1-t)) at the end
                        Real dt_adj_factor = sqrt(s * (1.0 - s) / (t * (1.0-t)));
                        // Shrink the time step to offset possible increase in j0
                        dt_react_step *= dt_adj_factor;
                    }

                    // Update dt_react to the cumulative minimum dt_react_step seen so far
                    dt_react = min(dt_react, dt_react_step);
                } 
            };
            Loop(b, func_b);
            return dt_react;
        };

        // Get maximum timestep that does not consume too much reactant in any cell
        Real dt_max_react_lev = ReduceMin(conc, source, flags, 0, func_min);

        // Cumulative dt_max_react over levels
        dt_max_react = std::min(dt_max_react, dt_max_react_lev);
    } // loop over levels lev

    // Get maximum CFLs of each type
    ReduceRealMax(conv_cfl);
    ReduceRealMax(diff_cfl);
    ReduceRealMax(react_cfl);
    // Get maximum time step from consuming reactant (dt_max_react)
    ReduceRealMin(dt_max_react);

    // cd_cfl combines convection and diffusion CFLs
    Real cd_cfl = conv_cfl + diff_cfl;
    cd_cfl *= 2.0; // needed for the advection implementation (see flow part)
    // Combined CFL includes convection, diffusion, and reaction
    // react_cfl has a placeholder value of zero; reaction enters through dt_max_react
    Real comb_cfl = cd_cfl + react_cfl;
    // The new time step using CFL; based on the input cfl_react parameter (dimensionless) and current combined CFL
    Real dt_cfl = m_cfl_react / comb_cfl;
    // The new time step is the minimum of two methods: CFL and reactant consumption
    Real dt_auto = std::min(dt_cfl, dt_max_react);
    // The new time step should never increase by a factor greater than m_max_dt_ratio
    dt_auto = std::min(dt_auto, m_max_dt_ratio * m_prev_dt);

    // Use the automatic time step computed from CFL and apply the time update
    m_dt = min(dt_auto, dt_to_stop);
    UpdateTime();

    DEBUG_PRINT(format("{:s}: set reaction time dt={:5.3e}.\n", FUNC_NAME, m_dt));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::ImposeRedoxConcConstraint(int lev)
{
    Vector<MultiFab*> const& conc_mf = get_conc_new();
    Vector<MultiFab*> const& soc_mf = get_soc_new();
  
    // Constant NaN so we can record missing values
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    #if (AMREX_USE_EB)
    for (MFIter mfi(*conc_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {           
        // Embedded boundaries
        const auto& fact = EBFactory(lev);
        const auto& flags_mf = fact.getMultiEBCellFlagFab();
        const EBCellFlagFab& flags_box = flags_mf[mfi];

        // The concentration and soc as non-const arrays
        Array4<Real> conc = conc_mf[lev]->array(mfi);
        Array4<Real> soc = soc_mf[lev]->array(mfi);

        // Box for iteration
        Box const& bx = mfi.tilebox();
        // Get the type of this box (e.g. regular, single valued, covered)
        auto const& typ = flags_box.getType(bx);
        bool is_boundary = fabIsBoundary(typ);

        // Regular cells (all liquid)
        if (typ == FabType::regular)
        {
            auto func =
            [this, conc, soc] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Calculate the state of charge and clamp it into the allowed range
                soc(i,j,k) = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0)+conc(i,j,k,1)), m_soc_min, m_soc_max);
                // Harmonize concentration of oxidized and reduced species to be consistent with this SOC
                conc(i,j,k,0) = m_conc_redox_tot * (1.0 - soc(i,j,k));
                conc(i,j,k,1) = m_conc_redox_tot * soc(i,j,k);
            };
            ParallelFor(bx, func);
        }
        // Covered cells can be skipped; SOC initialized to NaN and never changes
        // Boundary cells
        else if (is_boundary)
        {
            const auto& flag = flags_box.const_array();
            // These cells may be liquid, boundary or solid; must check all possibilities.
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // SOC on cells that are not covered
                if ( !flag(i,j,k).isCovered() )
                {
                    soc(i,j,k) = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0)+conc(i,j,k,1)), m_soc_min, m_soc_max);
                    // Harmonize concentration of oxidized and reduced species to be consistent with this SOC
                    conc(i,j,k,0) = m_conc_redox_tot * (1.0 - soc(i,j,k));
                    conc(i,j,k,1) = m_conc_redox_tot * soc(i,j,k);
                }
                else 
                    {soc(i,j,k) = NaN;}
            };
            ParallelFor(bx, func);
        }
    }
    #else
    for (MFIter mfi(*conc_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {           
        Array4<Real> conc = conc_mf[lev]->array(mfi);
        Array4<Real> soc = soc_mf[lev]->array(mfi);
        Box const& bx = mfi.tilebox();

        ParallelFor(bx, 
        [this, conc, soc] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Calculate the state of charge and clamp it into the allowed range
            soc(i,j,k) = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0)+conc(i,j,k,1)), m_soc_min, m_soc_max);
            // Harmonize concentration of oxidized and reduced species to be consistent with this SOC
            conc(i,j,k,0) = m_conc_redox_tot * (1.0 - soc(i,j,k));
            conc(i,j,k,1) = m_conc_redox_tot * soc(i,j,k);
        });
    #endif
}

// *********************************************************************************************************************
void Rincflo::ImposeRedoxConcConstraint()
{
    for (int lev = 0; lev <= finest_level; lev++)
        {ImposeRedoxConcConstraint(lev);}
}

// *********************************************************************************************************************
void Rincflo::CalcCurrentFromSource()
{
    #if (AMREX_USE_EB)
    // Are we covering multiple cuts?
    const bool cover_multiple_cuts = m_cover_multiple_cuts;
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto& current_mf = m_leveldata[lev]->current;
        const auto& source_mf = m_leveldata[lev]->source;
        const auto& volume_mf = m_leveldata[lev]->volume;
    
        for (MFIter mfi(current_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {           
            // Embedded boundaries
            const auto& fact = EBFactory(lev);
            const auto& flags_mf = fact.getMultiEBCellFlagFab();
            const EBCellFlagFab& flags_box = flags_mf[mfi];

            // The current as a non-const array
            Array4<Real> current = current_mf.array(mfi);
            // The source as a const array
            Array4<const Real> source = source_mf.const_array(mfi);
            // The volume on this level as a const array
            Array4<const Real> volume = volume_mf.const_array(mfi);

            // Box for iteration
            Box const& bx = mfi.tilebox();
            // Get the type of this box (e.g. regular, single valued, covered)
            auto const& typ = flags_box.getType(bx);
            bool is_boundary = fabIsBoundary(typ);

            // Only boundary cells have nonzero current; the rest can be skipped
            if (is_boundary)
            {
                const auto& flag = flags_box.const_array();
                // These cells may be liquid, boundary or solid; must check all possibilities.
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Current on boundary cells
                    if ( flag(i,j,k).isBoundary(cover_multiple_cuts) )
                    {
                        current(i,j,k) = m_nF_const * source(i,j,k) * volume(i,j,k);
                    }
                };
                ParallelFor(bx, func);
            }
        }
    }

    #endif
}