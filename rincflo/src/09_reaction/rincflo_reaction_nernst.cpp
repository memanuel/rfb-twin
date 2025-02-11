#include <rincflo.H>

// *********************************************************************************************************************
// one step with advection and diffusion in the Nernst model
void Rincflo::NernstStep (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt,
    bool explicit_diffusion)
{
    // Copy concentrations from new to old and fill patches
    StoreQuantities();

    // Apply the time step dt directly rather than via ComputeDtReaction()
    UpdateTime();

    // We always save the old nernst concentrations
    constexpr bool save_old_nernst = true;

    // Apply the boundary condition; this incorporates the overpotential
    NernstApplyBoundaryConc(save_old_nernst);

    // Do a single explicit time step for advection and diffusion if requested
    if (explicit_diffusion)
        AdvDiffStep_exp(AMREX_D_DECL(u_mac, v_mac, w_mac), dt);

    // Otherwise do implicit diffusion; advection is always explicit
    else
    {
        // Perform an advection step; don't store old values of concentrations
        constexpr bool store_old = false;
        AdvectionStep(AMREX_D_DECL(u_mac, v_mac, w_mac), dt, store_old);

        // Perform a diffusion step with Picard iteration
        int tot_nonlin_it = 0;
        bool converged = DiffStep_pic(dt, &tot_nonlin_it);

        // Check that we converged
        if (!converged)
            {Abort("Rincflo::NernstStep() - DiffStep_pic failed to converge!\n");}
    }

    // Update overpotential; we always want to update the source term here
    constexpr bool update_source = true;
    CalcOverpotNernstAll(m_nernst_omega, update_source);
}

// *********************************************************************************************************************
// compute time-step to use in the Nernst reaction model
void Rincflo::ComputeDtNernst()
{
    #define FUNC_NAME "Rincflo::ComputeDtNernst"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Store the past two dt
    m_prev_prev_dt = m_prev_dt;
    m_prev_dt = m_dt;
    
    // Expedited process when using a single fixed time step
    if (m_using_fixed_dt_single)
    {
        // Set the time step and time and return early
        m_dt = m_fixed_dt;
        return;
    }

    // Initialize CFL time scale for convection
    Real conv_cfl = 0.0;
    // Nernst model doesn't use explicit diffusion, so set diff_cfl to zero
    Real diff_cfl = 0.0;

    // Go through the levels and do calculations for CFL
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Extract information about this level
        auto const dx_inv = geom[lev].InvCellSizeArray();
        MultiFab const& vel = m_leveldata[lev]->velocity;
        MultiFab const& conc = m_leveldata[lev]->conc;

        // Embedded boundary and related
        const auto& fact = EBFactory(lev);
        const auto& flags = fact.getMultiEBCellFlagFab();

        // Get contribution to CFL from advection
        Real conv_lev = 0.0;       
        auto func = 
        [=] AMREX_GPU_HOST_DEVICE
        (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mx = -1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                {
                    mx = max((AMREX_D_TERM(
                              std::abs(v(i,j,k,0))*dx_inv[0], 
                            + std::abs(v(i,j,k,1))*dx_inv[1], 
                            + std::abs(v(i,j,k,2))*dx_inv[2])), 
                            mx);
                }
            };
            Loop(b, func_b);
            return mx;
        };
        conv_lev = ReduceMax(vel, flags, 0, func);
        // Cumulative conv_cfl term over levels
        conv_cfl = max(conv_cfl, conv_lev);
    }   // loop over levels lev

    // Get maximum convection CFLs of each type
    ReduceRealMax(conv_cfl);

    // cd_cfl combines convection and diffusion CFLs
    Real cd_cfl = conv_cfl + diff_cfl;    
    // double combined CFL; this is needed for the advection implementation
    cd_cfl *= 2.0; 
    // The new time step using CFL; based on the input cfl_react parameter (dimensionless) and combined CFL   
    Real dt_auto = m_cfl_react / cd_cfl;

    // Use the automatic time step computed from CFL
    m_dt = dt_auto;

    DEBUG_PRINT(format("{:s}: set reaction time dt={:5.3e}.\n", FUNC_NAME, m_dt));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// NernstApplyBoundaryConc- apply the concentration of active species implied by the Nernst equation
void Rincflo::NernstApplyBoundaryConc(bool save_old_nernst)
{
    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        const auto& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            const bool cover_multiple_cuts = m_cover_multiple_cuts;
            Array4<Real> const& conc    = ld.conc.array(mfi);
            Array4<Real> const& conc_o  = ld.conc_on.array(mfi);
            Array4<Real> const& conc_on = ld.conc_on.array(mfi);
            Array4<Real const> const& overpot = ld.overpot.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Only apply Nernst concentrations on boundary cells
                if(flag(i,j,k).isBoundary(cover_multiple_cuts)) 
                {
                    // The kinetically adjusted voltage difference
                    Real dv_adj = m_DVapp - overpot(i,j,k);
                    // The implied soc at equilibrium from the Nernst equation
                    Real soc_eq = 1.0 / (1.0 + exp(-m_nF_over_RT_const * dv_adj));
                    // Clamp soc_eq to the range [m_soc_min, m_soc_max]
                    soc_eq = std::clamp(soc_eq, m_soc_min, m_soc_max);
                    // The implied concentration of the active species
                    conc(i,j,k,0) = m_conc_redox_tot * (1.0 - soc_eq);
                    conc(i,j,k,1) = m_conc_redox_tot * (soc_eq);
                    // Also write to conc_o (required for advection)
                    conc_o(i,j,k,0) = conc(i,j,k,0);
                    conc_o(i,j,k,1) = conc(i,j,k,1);
                    // Save the nernst BC values onto conc_on if applicable
                    if (save_old_nernst)
                    {
                        conc_on(i,j,k,0) = conc(i,j,k,0);
                        conc_on(i,j,k,1) = conc(i,j,k,1);
                    }
                }
            };
            ParallelFor(bx, func);
        }
    }
}

// *********************************************************************************************************************
void Rincflo::CalcOverpotNernstAll(Real omega, bool update_source)
{
    // The mutltifabs we're writing to
    Vector<MultiFab*> const& source = get_source();
    Vector<MultiFab*> const& overpot = get_overpot(); 

    // The multifabs we're reading from
    Vector<MultiFab const*> const& conc = get_conc_new_const();
    Vector<MultiFab const*> const& conc_on = get_conc_old_nernst_const();
 
    // For each level, build the boundary factory and delegate to CalcOverpotNernst where necessary
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // The cell volume from the geometry
        const auto& mf_volume = m_leveldata[lev]->volume;
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*source[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto& ld = *m_leveldata[lev];
            Box const& bx = mfi.tilebox();
            auto const& fact = EBFactory(lev);
            EBCellFlagFab const& flag_fab = fact.getMultiEBCellFlagFab()[mfi];            
            // Does this box contain boundary cells?
            auto typ = flag_fab.getType(bx);
            if (fabIsBoundary(typ))
            {
                Array4<Real const> const& area_over_volume = ld.area_over_volume.const_array(mfi);
                Array4<Real const> const& volume = ld.volume.const_array(mfi);
                Array4<EBCellFlag const> const& flag = flag_fab.const_array();
                CalcOverpotNernst(
                        lev, bx,
                        overpot[lev]->array(mfi),
                        source[lev]->array(mfi),
                        conc[lev]->const_array(mfi),
                        conc_on[lev]->const_array(mfi),
                        area_over_volume,
                        flag, omega, update_source);
            }
        }
    }
}

// *********************************************************************************************************************
void Rincflo::CalcOverpotNernst (
        int lev, Box const& bx,
        Array4<Real> const& overpot,
        Array4<Real> const& source,
        Array4<Real const> const& conc,
        Array4<Real const> const& conc_on,
        Array4<Real const> const& area_over_volume,
        Array4<EBCellFlag const> const& flag,
        Real omega,
        bool update_source)
{
    const bool cover_multiple_cuts = m_cover_multiple_cuts;
    auto func = 
    [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // There is only a source term on single-valued boundary cells
        if (flag(i,j,k).isBoundary(cover_multiple_cuts))
        {
            // The source term is the rate of change in the concentration of the reduced species (millimolarity / sec)
            // Why is it conc_o minus conc? NernstApplyBoundaryConc saves the starting concentrations in conc_on
            // Then the batch of time step is simulated and the reduced species concentration DROPS by the source term
            // The true source term therefore is the opposite of this quantity, which would conserve mass
            // Impose the constraint that the source term is always positive (known to be true at steady state)
            // source(i,j,k) = max((conc_on(i,j,k,1) - conc(i,j,k,1)) / m_dt, 0.0);

            // Only update the source term when requested
            if (update_source)
                {source(i,j,k) = (conc_on(i,j,k,1) - conc(i,j,k,1)) / m_dt;}

            // The exchange current when alpha0 = alpha1 = 0.5; use the old concentrations
            Real j0 = m_k_0 * sqrt(conc_on(i,j,k,0) * conc_on(i,j,k,1));

            // Solve for the overpotential in terms of the source term EWMA
            // Equivalent to inverting the equation below and solving for overpot(i,j,k) in terms of source(i,j,k)
            // source(i,j,k) = 2.0 * j0 * sinh(0.5 * m_nF_over_RT_const * overpot(i,j,k)) * area_over_volume(i,j,k);
            // overpot(i,j,k) = (2.0 / m_nF_over_RT_const) * asinh(source(i,j,k) / (2.0 * j0 * area_over_volume(i,j,k)));
            Real op_new = (2.0 / m_nF_over_RT_const) * asinh(source(i,j,k) / (2.0 * j0 * area_over_volume(i,j,k)));
            // Clamp the new overpotential
            op_new = std::clamp(op_new, 0.0, m_overpot_max);

            // Take weighted average of new and old overpotentials using relaxation parameter omega
            overpot(i,j,k) += omega * (op_new - overpot(i,j,k));
        }
    };
    ParallelFor(bx, func);
}


// *********************************************************************************************************************
Real Rincflo::CalcTotalCurrent_BV() const
{
    // Only the finest level is used
    const int lev = finest_level;

    // Embedded boundary related
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    // The total concentration of redox species
    const Real T0 = m_conc_redox_tot;
    // The rate constant
    const Real k0 = m_k_0;
    // Faraday's constant times number of electrons
    const Real nF = m_nF_const;
    // Range for clamping the overpotential
    const Real overpot_max = m_overpot_max;
    // Prefactor to multiply eta
    const Real eta_pre = 0.5 * m_nF_over_RT_const;    
        
    // Accumulate and return the total current
    Real current = 0.0;

    for (MFIter mfi(m_leveldata[lev]->soc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const &bx = mfi.tilebox();
        auto typ = flags[mfi].getType(bx);
        bool is_boundary = fabIsBoundary(typ);
        if (is_boundary)
        {
            // The flags in this box
            Array4<EBCellFlag const> const& flag = flags[mfi].const_array();
            // The state of charge
            Array4<Real const> const& soc = m_leveldata[lev]->soc.const_array(mfi);
            // The overpotential
            Array4<Real const> const& overpot = m_leveldata[lev]->overpot.const_array(mfi);
            // The area area of this cut cell
            Array4<Real const> const& area = m_leveldata[lev]->area.const_array(mfi);

            // Function to accumulate the current
            auto func =
            [=, &current] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
            {
                // Only boundary cells have current
                if (flag(i,j,k).isBoundary(cover_multiple_cuts))
                {
                    // The exchange current density
                    Real j0 = k0 * T0 * nF * sqrt(soc(i,j,k) * (1.0 - soc(i,j,k)));
                    // The clamped overpotential
                    Real eta = std::clamp(overpot(i,j,k), -overpot_max, overpot_max);
                    // The Butler-Volmer equation
                    current += 2.0 * j0 * sinh(eta_pre * eta) * area(i,j,k);
                }
            };
            Loop(bx, func);
        }
    }

    ReduceRealSum(current);
    return current;
}

// *********************************************************************************************************************
// apply diffusion for one time step using Picard iteration
bool Rincflo::DiffStep_pic (Real dt, int* tot_it)
{
    #define FUNC_NAME "Rincflo::DiffStep_pic"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Initialize variables for Picard iteration
    int max_it = m_fixpoint_max_iter;
    int pic_it = 0;
    bool do_iteration = true;
    bool converged = false;

    // Main loop
    while ( do_iteration )
    {
        ++pic_it;

        // Store quantities
        CopyNewToOldPicard_react();
        for(int lev = 0; lev <= finest_level; lev++) 
            {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc_op, nghost_state());}

        // Construct RHS and solve diffusion equation; RHS includes advective term
        constexpr bool set_rhs = true;
        UpdateDiffConc(dt, set_rhs);

        // Check convergence
        converged = CheckConvergenceFixPoint(m_verbose);
        do_iteration = !(converged || (pic_it > max_it));
    }

    if(m_verbose > 1) 
        {Print() << format("Picard iterations: {:d}.\n", pic_it);}

    *tot_it = pic_it;
  
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return converged;
}
