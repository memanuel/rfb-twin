#include <rincflo.H>

// *********************************************************************************************************************
// Initialization before initial solve.
// Update of marker and cell calculations (face centered velocities) after regridding.
// Also include the linear reaction model here.
// *********************************************************************************************************************

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// Initialize some variable                                               //
// -----------------------------------------------------------------------//
void Rincflo::InitReact()
{
    #define FUNC_NAME "Rincflo::InitReact"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Zero step counters if requested
    if (m_react_zero_step_counter)
    {
        m_rstep = 0;
        m_start_rsteps = 0;
        m_cur_time = 0.0;
    }
    // Manually reset reaction counter if requested
    else if (m_react_init_step_counter> 0)
    {
        m_rstep = m_react_init_step_counter;
        m_start_rsteps = m_react_init_step_counter;
    }

    // Step of last plot file and checkpoint
    m_last_plt = -1;
    m_last_chk = -1;
    // Always advect species concentration in reaction models
    m_advect_conc = 1;

    // Save flow fields including concentrations
    CopyNewToOld_velocity();
    CopyNewToOld_density();
    CopyNewToOld_conc();
    // Initialize the time arrays based on m_cur_time and m_dt
    UpdateTimeArrays();

    // Refine flow fields including concentrations (first old, then new)
    const int ng = nghost_state();
    // Refine electric potential fields in the BV model (old and new)
    if(m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        CopyNewToOld_epotL();
        CopyNewToOld_epotS();
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            auto& ld = m_leveldata[lev];
            fillpatch_epotS(lev, m_t_old[lev], ld->epotS_o, ng);
            fillpatch_epotL(lev, m_t_old[lev], ld->epotL_o, ng);
            fillpatch_epotS(lev, m_t_new[lev], ld->epotS, ng);
            fillpatch_epotL(lev, m_t_new[lev], ld->epotL, ng);
        }
        if (m_verbose > 1)
            {"InitReact() - completed extra fillpatch for BV.\n";}
    }
 
    // Re-initialize concentrations as required
    // Check for consistency of conc initialization flags
    if (m_react_reinit_all_conc && m_react_reinit_some_conc)
        {Abort("Error! Cannot have both ReinitConcAll and ReinitConcSome, they are mutually exclusive.\n");}
    // Reinitialize all the concentrations if requested
    if(m_react_reinit_all_conc)   
    {
        // Print() << "InitReact() - Reinitialize all concentrations.\n";
        ReinitConcAll();
    }
    // Reinitialize some of the concentrations if requested
    else if(m_react_reinit_some_conc)   
    {
        // Print() << format("InitReact() - Reinitialize some concentrations (from {:d} to last.\n", m_react_reinit_conc_from);
        ReinitConcSome();
    }
    // Do fillpatch for velocity and density
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        // Writable copy of level data on this level
        auto& ld = m_leveldata[lev];
        // Patch the old velocity and density; concentration was already patched above
        fillpatch_velocity(lev, m_t_old[lev], ld->velocity_o, ng);
        fillpatch_density(lev, m_t_old[lev], ld->density_o, ng);
        // Patch the new velocity and density; concentration was already patched above
        fillpatch_velocity(lev, m_t_new[lev], ld->velocity, ng);
        fillpatch_density(lev, m_t_new[lev], ld->density, ng);
    }
    if (m_verbose > 1)
        {Print() << "InitReact() - completed fillpatch.\n";}

    // Initialize source and overpot to zero if this isn't a restart
    if (!is_restart())
    {
        for (int lev = 0; lev <= finest_level; ++lev)   
        {
            auto& ld = *m_leveldata[lev];
            ld.source.setVal     (0.0, 0, 1, 0);
            ld.overpot.setVal    (0.0, 0, 1, 0);
        }
    }
    
    // Initialize pot_diff
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Embedded boundaries
        #if (AMREX_USE_EB)
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
        #endif

        // The potential difference MultiFab
        auto& ld_pot_diff = m_leveldata[lev]->pot_diff;
        // The potential difference is zero everywhere except boundary cells which will be written below
        ld_pot_diff.setVal(0.0, 0, 1, 0);

        // Iterate over multifabs
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for(MFIter mfi(ld_pot_diff, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            #if (AMREX_USE_EB)
            // Get the type of this box (e.g. regular, single valued, covered)
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_boundary = fabIsBoundary(typ);
            // Writable array of potential difference
            Array4<Real> const& pot_diff = ld_pot_diff.array(mfi);
            // Local alias for cover_multiple_cuts
            const bool cover_multiple_cuts = m_cover_multiple_cuts;

            // Only single-valued and multi-valued boxes can contain boundary cells
            // Don't waste time initializing pot_diff in ButlerVolmer; it's set in InitEpot() below.
            if (is_boundary && (m_reaction_model != ReactionModel::ButlerVolmer))
            {
                // These cells may be liquid, boundary or solid; must check all possibilities.
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (flag(i,j,k).isBoundary(cover_multiple_cuts))
                    {
                        // Set the potential difference equal to the applied voltage
                        pot_diff(i,j,k) = m_DVapp;
                    }
                };
                ParallelFor(bx, func);
            }
            #endif
        }
        if (m_verbose > 1)
            {Print() << format("InitReact() - Initialized pot_diff on level {:d}.\n", lev);}
    }

    // If we're not solving for the electric potential in the solid, we always need to reinitialize it
    if(!m_solve_epotS )
        {m_react_reinit_epotS = true;}
    // Initialize potential in solid and liquid and the potential difference
    // Always call this even if not reinitializing epotS or epotL to ensure that pot_diff is initialized
    if(m_reaction_model == ReactionModel::ButlerVolmer)   
        {InitEpot();}

    // The source term will be needed for first dt estimate
    CalcSourceTermAll();

    // Calc other quantities (mostly for plotting purpose)
    Vector<MultiFab> kappaS;
    const int nghost_kappa=1;
    const int nghost_flxrhs=0;

    // Initialize conductivity in the solid if necessary
    if(m_reaction_model == ReactionModel::ButlerVolmer &&  m_solve_epotS)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }

    // Calculate conductivity
    if(m_reaction_model == ReactionModel::ButlerVolmer)
        {CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);}

    //(Calculate conc flux)
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL) 
    {
        get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());
        CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
    }

    // Calculate the nominal flow rate along the x-axis for the nominal flow time
    constexpr int lev {0};
    constexpr int dir {0};
    constexpr bool lo {false};
    const auto& vel = m_leveldata[lev]->velocity;
    constexpr int comp {0};
    constexpr bool local {false};
    constexpr bool vel_is_nodal {false};
    const long Ny = Geom(lev).Domain().length(1);
    const long Nz = AMREX_D_PICK(1, 1, Geom(lev).Domain().length(2));
    Real u_sum = SumOverBoundaryCells(dir, lo, vel, comp, vel_is_nodal, local);
    Real u_nominal = u_sum / (Ny * Nz);
    m_t_flow = geom[0].ProbLength(0) / u_nominal;

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------------------------- //
// apply MAC projection to velocity (get face-centered velocities to be used in advection)  //
// ---------------------------------------------------------------------------------------- //
void Rincflo::CalcMacVel (
    Vector<MultiFab const*> const& vel,
    Vector<MultiFab const*> const& density,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac)
    )
{
    int ngmac = nghost_mac();
    // Go through the levels
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        AMREX_D_TERM( 
        u_mac[lev]->define(convert(grids[lev],IntVect::TheDimensionVector(0)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));,
        v_mac[lev]->define(convert(grids[lev],IntVect::TheDimensionVector(1)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));,
        w_mac[lev]->define(convert(grids[lev],IntVect::TheDimensionVector(2)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));)
    } 
    // Set boundary values if ghost cells being used
    if(ngmac > 0) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            AMREX_D_TERM(
            u_mac[lev]->setBndry(0.0);,
            v_mac[lev]->setBndry(0.0);,
            w_mac[lev]->setBndry(0.0);)
        }
    }

    // This will hold (1/rho) on faces
    AMREX_D_TERM(
    Vector<MultiFab> inv_rho_x(finest_level+1);,
    Vector<MultiFab> inv_rho_y(finest_level+1);,
    Vector<MultiFab> inv_rho_z(finest_level+1);)

    Vector<Array<MultiFab*,SpaceDim> > inv_rho(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(
        inv_rho[lev][0] = &inv_rho_x[lev];,
        inv_rho[lev][1] = &inv_rho_y[lev];,
        inv_rho[lev][2] = &inv_rho_z[lev];)
    }

    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        AMREX_D_TERM(
        inv_rho_x[lev].define(u_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));,
        inv_rho_y[lev].define(v_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));,
        inv_rho_z[lev].define(w_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));)
    
        #if (AMREX_USE_EB)
        const EBFArrayBoxFactory* ebfact = &EBFactory(lev);
        EB_interp_CellCentroid_to_FaceCentroid (*density[lev], inv_rho[lev], 0, 0, 1, geom[lev], get_density_bcrec());
        #else
        average_cellcenter_to_face(inv_rho[lev], *density[lev], geom[lev]);
        #endif
    
        for (int idim = 0; idim < SpaceDim; ++idim) 
            {inv_rho[lev][idim]->invert(1.0, 0);}
    }

    // Forcing terms (needed only when calling Godunov...)
    Vector<MultiFab> vel_forces;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        vel_forces.emplace_back(grids[lev], dmap[lev], SpaceDim, nghost_force(), MFInfo(), Factory(lev));
        vel_forces[lev].setVal(0.0);
    }

    // Is the following really needed ? 
    if (m_advection_type != "MOL") 
    {        
        bool include_pressure_gradient = !(m_use_mac_phi_in_godunov);
        ComputeVelForces(GetVecOfPtrs(vel_forces), vel, density, density, density, include_pressure_gradient);
        
        if (m_godunov_include_diff_in_forcing)
        {
            for (int lev = 0; lev <= finest_level; ++lev)
                {MultiFab::Add(vel_forces[lev], m_leveldata[lev]->divtau_o, 0, 0, SpaceDim, 0);}
        }
        
        if (nghost_force() > 0)
            {fillpatch_force(m_cur_time, GetVecOfPtrs(vel_forces), nghost_force());}
    }
        
    Compute_MacProjectedVelocities(
        vel,  
        AMREX_D_DECL(u_mac,v_mac,w_mac), 
        AMREX_D_DECL(GetVecOfPtrs(inv_rho_x), GetVecOfPtrs(inv_rho_y), GetVecOfPtrs(inv_rho_z)), 
        GetVecOfPtrs(vel_forces), 0.0);    
}
