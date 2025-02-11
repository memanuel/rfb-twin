#include <rincflo.H>
#include <Convection_conc_only.H>

// *********************************************************************************************************************
// reinitialize all concentrations
void Rincflo::ReinitConcAll ()
{
    #define FUNC_NAME "Rincflo::ReinitConcAll"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Delegate to ReinitConcImpl
    ReinitConcImpl(0, m_nspec);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// reinitialize selected concentrations as loaded from reaction parameters
void Rincflo::ReinitConcSome()
{
    #define FUNC_NAME "Rincflo::ReinitConcSome"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Delegate to ReinitConcImpl
    ReinitConcImpl(m_react_reinit_conc_from, m_nspec);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Reinitialize selected concentrations
void Rincflo::ReinitConcImpl (int n0, int n1)
{
    #define FUNC_NAME "Rincflo::ReinitConcImpl"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    for (int lev = 0; lev <= finest_level; lev++)
    {
        #if (AMREX_USE_EB)
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
        #endif
        auto& ld = *m_leveldata[lev];

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            // Embedded boundaries
            #if (AMREX_USE_EB)
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_liquid = fabIsLiquid(typ);
            bool is_boundary = fabIsBoundary(typ);
            #endif

            // non-const array of concentrations
            Array4<Real> const& conc = ld.conc.array(mfi);
            // non-const array of state of charge
            Array4<Real> const& soc = ld.soc.array(mfi);
            
            // Check that m_react_init_t has been initialized
            if (m_react_init_t.size() < 2)
                {Abort("m_react_init_t has not been initialized. Size must be at least 2!");}
            // The state of charge in this initialization
            const Real soc_init = m_react_init_t[1] / (m_react_init_t[0] + m_react_init_t[1]);
            // Is the SOC being initialized here to the value soc_init above?
            const bool is_soc_init = (n0 == 0) && (n1 > 0);

            #if (AMREX_USE_EB)
            // The box contains only regular cells (all liquid)
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int n = n0; n < n1; ++n) 
                        {conc(i,j,k,n) = m_react_init_t[n];}
                    soc(i,j,k) = is_soc_init ? soc_init : conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
                };
                ParallelFor(bx, func);
            }
            // The box contains cells of multiple types including some boundary cells
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if(!flag(i,j,k).isCovered()) 
                    {
                        for (int n = n0; n < n1; ++n) 
                            {conc(i,j,k,n) = m_react_init_t[n];}
                        soc(i,j,k) = is_soc_init ? soc_init : conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
                    }
                    else 
                    {
                        for (int n = n0; n < n1; ++n) 
                            {conc(i,j,k,n) = 0.0;}
                        soc(i,j,k) = NaN;
                    }
                };
                ParallelFor(bx, func);
            }
            // The box contains only covered cells (all solid)
            else
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int n = n0; n < n1; ++n) 
                        {conc(i,j,k,n) = 0.0;}
                    soc(i,j,k) = NaN;
                };
                ParallelFor(bx, func);
            }

            // When there are no embedded boundaries, all cells are regular
            #else
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int n = n0; n < n1; ++n) 
                    {conc(i,j,k,n) = m_react_init_t[n];}
                soc(i,j,k) = is_soc_init ? soc_init : conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
            };
            ParallelFor(bx, func);
            #endif
        }
    }

    // We also want to intialize conc_o so the first time step is correct
    CopyNewToOld_conc();

    // Patch the concentration
    const int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc  , ng);
        fillpatch_conc(lev, m_t_old[lev], m_leveldata[lev]->conc_o, ng);
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// calc convective term for conc only (not multiplied by rho!)
void Rincflo::ComputeConvectiveTermConc (
    Vector<MultiFab*> const& dconc_dt,
    Vector<MultiFab const*> const& conc,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac)
    )
{
    #define FUNC_NAME "Rincflo::ComputeConvectiveTermConc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    #if (AMREX_USE_EB)
    if(m_verbose>1)
        {Print() << format("RedistributionType {:s} \n", m_redistribution_type);}
    #endif
    auto l_advection_type = m_advection_type;

    // This will hold fluxes on faces
    AMREX_D_TERM(
    Vector<MultiFab> flux_x(finest_level+1);,
    Vector<MultiFab> flux_y(finest_level+1);,
    Vector<MultiFab> flux_z(finest_level+1);)
    DEBUG_PRINT("Created empty vectors of flux multifabs.\n");

    // Make one flux MF at each level to hold all the fluxes of the species
    const int n_flux_comp = m_nspec;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        AMREX_D_TERM(
        flux_x[lev].define(u_mac[lev]->boxArray(), dmap[lev], n_flux_comp, 0, MFInfo(), Factory(lev));,
        flux_y[lev].define(v_mac[lev]->boxArray(), dmap[lev], n_flux_comp, 0, MFInfo(), Factory(lev));,
        flux_z[lev].define(w_mac[lev]->boxArray(), dmap[lev], n_flux_comp, 0, MFInfo(), Factory(lev));)
    }
    
    Vector<Array<MultiFab*, SpaceDim> > fluxes(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(
        fluxes[lev][0] = &flux_x[lev];,
        fluxes[lev][1] = &flux_y[lev];,
        fluxes[lev][2] = &flux_z[lev];)
    }
    
    // Forcing terms (needed only when calling Godunov...)
    Vector<MultiFab> tra_forces;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        tra_forces.emplace_back(grids[lev], dmap[lev], SpaceDim, nghost_force(), MFInfo(), Factory(lev));
        tra_forces[lev].setVal(0.0);
    }
    
    int ngmac = nghost_mac();

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (ngmac > 0) 
        {
            AMREX_D_TERM(
            u_mac[lev]->FillBoundary(geom[lev].periodicity());,
            v_mac[lev]->FillBoundary(geom[lev].periodicity());,
            w_mac[lev]->FillBoundary(geom[lev].periodicity());)
        }

        // MultiFab divu(conc[lev]->boxArray(),conc[lev]->DistributionMap(), 1, 4);
        MultiFab divu(conc[lev]->boxArray(),conc[lev]->DistributionMap(), 1, conc[lev]->nGrow());
        divu.setVal(0.);
        if (l_advection_type != "MOL") 
        {
            Array<MultiFab const*, SpaceDim> u;
            AMREX_D_TERM(
            u[0] = u_mac[lev];,
            u[1] = v_mac[lev];,
            u[2] = w_mac[lev];)
      
            #if (AMREX_USE_EB)
            auto const& fact = EBFactory(lev);
            if (fact.isAllRegular())
                {computeDivergence(divu,u,geom[lev]);}
            else
                {EB_computeDivergence(divu,u,geom[lev],true);}
            #else
            computeDivergence(divu,u,geom[lev]);
            #endif
            divu.FillBoundary(geom[lev].periodicity());
        }
    
        #if (AMREX_USE_EB)
        const EBFArrayBoxFactory* ebfact = &EBFactory(lev);
        #endif

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*conc[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            convection::compute_fluxes_conc(
                bx, mfi,
                conc[lev]->const_array(mfi),
                divu.const_array(mfi),
                AMREX_D_DECL(
                u_mac[lev]->const_array(mfi),
                v_mac[lev]->const_array(mfi),
                w_mac[lev]->const_array(mfi)),
                AMREX_D_DECL(
                flux_x[lev].array(mfi),
                flux_y[lev].array(mfi),
                flux_z[lev].array(mfi)),
                tra_forces[lev].const_array(mfi),
                get_conc_bcrec(), 
                get_conc_bcrec_device_ptr(),
                get_conc_iconserv_device_ptr(),
                m_advection_type,
                m_nspec,
                m_godunov_ppm, m_godunov_use_forces_in_trans,
                #if (AMREX_USE_EB)
                ebfact,
                m_cover_multiple_cuts,
                #endif
                geom[lev],m_dt);
        }
        // DEBUG_PRINT(format("Completed convection::compute_fluxes_conc for level {:d}.\n", lev));
    }

    // In order to enforce conservation across coarse-fine boundaries we must be sure 
    // to average down the fluxes before we use them
    for (int lev = finest_level; lev > 0; --lev)
    {
        IntVect rr  = geom[lev].Domain().size() / geom[lev-1].Domain().size();
        #if (AMREX_USE_EB)
        EB_average_down_faces(GetArrOfConstPtrs(fluxes[lev]), fluxes[lev-1], rr, geom[lev-1]);
        #else
        average_down_faces(GetArrOfConstPtrs(fluxes[lev]), fluxes[lev-1], rr, geom[lev-1]);
        #endif
    }
    // DEBUG_PRINT("Completed EB_average_down_faces.\n");
    
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        #if (AMREX_USE_EB)
        MultiFab dconc_dt_tmp(conc[lev]->boxArray(), dmap[lev], m_nspec, 3, MFInfo(), Factory(lev)); 
        const EBFArrayBoxFactory* ebfact = &EBFactory(lev);
        // Must initialize to zero because not all values may be set, e.g. outside the domain.
        dconc_dt_tmp.setVal(0.);
        #endif

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*conc[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            convection::compute_convective_term_conc_impl(bx, mfi,
            #if (AMREX_USE_EB)
            dconc_dt_tmp.array(mfi),
            #else
            dconc_dt[lev]->array(mfi),
            #endif
            AMREX_D_DECL(
            u_mac[lev]->const_array(mfi),
            v_mac[lev]->const_array(mfi),
            w_mac[lev]->const_array(mfi)),
            AMREX_D_DECL(
            flux_x[lev].const_array(mfi),
            flux_y[lev].const_array(mfi),
            flux_z[lev].const_array(mfi)),
            get_conc_iconserv_device_ptr(),
            m_advection_type,
            m_nspec,
            #if (AMREX_USE_EB)
            ebfact,
            m_cover_multiple_cuts,
            #endif
            geom[lev]);
        }
    
        #if (AMREX_USE_EB)
        // We only filled these on the valid cells so we fill same-level interior ghost cells here. 
        // (We don't need values outside the domain or at a coarser level so we can call just FillBoundary)
        dconc_dt_tmp.FillBoundary(geom[lev].periodicity());
        
        for (MFIter mfi(*conc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            convection::redistribute_convective_term_conc (
                bx, mfi, conc[lev]->const_array(mfi), dconc_dt_tmp.array(mfi), dconc_dt[lev]->array(mfi),
                m_redistribution_type, m_nspec, ebfact, m_cover_multiple_cuts, geom[lev], m_dt);
        }
        #endif
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// construct RHS and solve diff equation for conc
// *********************************************************************************************************************
void Rincflo::UpdateDiffConc(Real dt, bool set_rhs)
{
    #define FUNC_NAME "Rincflo::UpdateDiffConc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Vector<MultiFab> rhs_arr;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {rhs_arr.emplace_back(grids[lev], dmap[lev], m_nspec, 0, MFInfo(), Factory(lev));}

    // Construct RHS = conc_adv, the concentrations after advection
    int l_nspec = m_nspec;
    if(set_rhs) 
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto const& fact = EBFactory(lev);
            auto& ld = *m_leveldata[lev];
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(rhs_arr[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                Box const& bx = mfi.tilebox();
                Array4<Real> const& rhs = rhs_arr[lev].array(mfi);
                // updated convective term
                Array4<Real const> const& conc_adv = ld.conc_adv.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int n = 0; n < l_nspec; ++n) 
                        {rhs(i,j,k,n) = conc_adv(i,j,k,n); }
                };
                ParallelFor(bx, func);
            }    
        }
    }

    // Apply boundary condiditions using 1 ghost cell for diffusion
    constexpr int ng_diff = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillphysbc_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, ng_diff);}

    // Calculate coefficients for the equation
    Vector<MultiFab> species_coeff;
    if(!m_constant_Ds) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {species_coeff.emplace_back(grids[lev], dmap[lev], m_nspec, ng_diff, MFInfo(), Factory(lev));}
        GetDiffCoeffSpecies(GetVecOfPtrs(species_coeff), ng_diff);
    }

    Vector<MultiFab> acoeff;
    // Solve equation (in: rhs, conc as initial guess ;  out: updated conc)
    get_reaction_scalar_op()->react_scalar(
        get_conc_new(), GetVecOfPtrs(rhs_arr), GetVecOfConstPtrs(species_coeff), GetVecOfConstPtrs(acoeff), dt, 1);

    // Run fillpatching with the full number of ghost cells (4 for EB)
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, nghost_state());}

    // Copy conc to conc_adv that will be used as RHS in next step
    if(set_rhs) 
    {
        for(int lev = 0; lev <= finest_level; lev++)      
        {    
            MultiFab const& conc_new = m_leveldata[lev]->conc;
            MultiFab& conc_adv = m_leveldata[lev]->conc_adv;
            MultiFab::Copy(conc_adv, conc_new, 0, 0, m_nspec, 0);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// react update for conc (Picard)                                               //
void Rincflo::UpdateReactConc_pic(Real dt)
{
    #define FUNC_NAME "Rincflo::UpdateReactConc_pic"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // conc = conc + dt * prefactor * source
    int l_nspec = m_nspec;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            // conc to update -> RHS
            Array4<Real> const& conc   = ld.conc.array(mfi);                        
            // contain previously updated conc
            Array4<Real const> const& conc_adv = ld.conc_adv.const_array(mfi);    
            // source (reaction) term
            Array4<Real const> const& st = ld.source.const_array(mfi);              
            auto func =
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int n = 0; n < l_nspec; ++n) 
                    {conc(i,j,k,n) = conc_adv(i,j,k,n) + dt * m_s_pref_conc[n] * st(i,j,k);}
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// construct RHS and solve diff-react equation for conc (Picard)
// *********************************************************************************************************************
void Rincflo::UpdateDiffReactConc_pic(Real dt)
{
    #define FUNC_NAME "Rincflo::UpdateDiffReactConc_pic"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Vector<MultiFab> rhs_arr;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {rhs_arr.emplace_back(grids[lev], dmap[lev], m_nspec, 0, MFInfo(), Factory(lev));}
    // construct RHS = conv_tra + dt * prefactor * source
    // if electromigration:
    // RHS = conv_tra + dt * prefactor * source + dt * E
    // E = zDF/RT * div * (c*gradpot)

    int l_nspec = m_nspec;

    if(do_Electromigration()) 
        {get_reaction_eliquid_op()->compute_divcgpot(get_divcgpot(),get_epotL_new_const(),get_conc_new_const());}

    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto const& fact = EBFactory(lev);
        auto& ld = *m_leveldata[lev];
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(rhs_arr[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            Box const& bx = mfi.tilebox();
            Array4<Real> const& rhs = rhs_arr[lev].array(mfi);
            // concentration updated only for advection
            Array4<Real const> const& conc_adv  = ld.conc_adv.const_array(mfi);
            // source (reaction) term
            Array4<Real const> const& st = ld.source.const_array(mfi);

            if(do_Electromigration()) 
            {
                // div (c gradpot)
                Array4<Real const> const& dcgp = ld.divcgpot.const_array(mfi);     
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for ( int n = 0; n < l_nspec; ++n) 
                        {rhs(i,j,k,n) = conc_adv(i,j,k,n) + dt*(m_s_pref_conc[n]*st(i,j,k) + m_zDFoverRT[n]*dcgp(i,j,k,n) );}
                };
                ParallelFor(bx, func);
            }
            else 
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int n = 0; n < l_nspec; ++n) 
                        {rhs(i,j,k,n) = conc_adv(i,j,k,n) + dt * m_s_pref_conc[n] * st(i,j,k);}
                };
                ParallelFor(bx, func);
            }
        }
        // DEBUG_PRINT(format("UpdateDiffReactConc_pic - Update rhs for level {:d}.\n", lev));
    }
    DEBUG_PRINT("UpdateDiffReactConc_pic - Completed update rhs for all levels.\n");

    // adjust number ghost cells
    constexpr int ng_diffreact = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillphysbc_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, ng_diffreact);}
    DEBUG_PRINT("UpdateDiffReactConc_pic - Completed fillphysbc_conc for all levels.\n");

    // calc/setup coefficients for the equation
    Vector<MultiFab> species_coeff;
    if(!m_constant_Ds) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {species_coeff.emplace_back(grids[lev], dmap[lev], l_nspec, ng_diffreact, MFInfo(), Factory(lev));}
        GetDiffCoeffSpecies(GetVecOfPtrs(species_coeff), ng_diffreact);
    }

    Vector<MultiFab> acoeff;

    // solve equation (in: rhs, conc as initial guess ;  out: updated conc)
    get_reaction_scalar_op()->react_scalar(get_conc_new(),GetVecOfPtrs(rhs_arr),GetVecOfConstPtrs(species_coeff),GetVecOfConstPtrs(acoeff),dt,1);
    DEBUG_PRINT("UpdateDiffReactConc_pic - Solved linear equation in get_reaction_scalar_op().\n");

    // re-update number ghost cells
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, nghost_state());}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// get diffusion coefficients for species
void Rincflo::GetDiffCoeffSpecies (Vector<MultiFab*> const& species_coeff, int nghost)
{
    #define FUNC_NAME "Rincflo::GetDiffCoeffSpecies"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    for (auto mf : species_coeff) 
    {
        for (int n = 0; n < m_nspec; ++n) 
            {mf->setVal(m_D_s[n], n, 1, nghost);}
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// get reaction operator                                                  //
ReactionScalarOp* Rincflo::get_reaction_scalar_op ()
{
    if (!m_reaction_scalar_op) 
        {m_reaction_scalar_op.reset(new ReactionScalarOp(this, m_nspec));}
    return m_reaction_scalar_op.get();
}

// *********************************************************************************************************************
// get reaction operator b.c.                                             //
Array<LinOpBCType,SpaceDim>
Rincflo::get_react_scalar_bc (Orientation::Side side, int n) const noexcept
{
    #define FUNC_NAME "Rincflo::get_react_scalar_bc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Array<LinOpBCType,SpaceDim> r;
    for (int dir = 0; dir < SpaceDim; ++dir) 
    {
        if (Geom(0).isPeriodic(dir)) 
            {r[dir] = LinOpBCType::Periodic;} 
        else 
        {
            auto bc = m_bc_type[Orientation(dir,side)];
            switch (bc)
            {
                case BC::mix_wall_potL:
                {
                    // for protons this is the membrane where we fix proton concentration
                    if(m_fix_protons_mem && n==2) 
                        {r[dir] = LinOpBCType::Dirichlet;}
                    else
                        {r[dir] = LinOpBCType::Neumann;}
                    break;
                }
                case BC::pressure_outflow:
                case BC::slip_wall:
                case BC::no_slip_wall:
                case BC::charging_wall_pot:
                case BC::charging_wall_cur:
                case BC::mix_wall_potS:
                case BC::mix_wall_potL_currS:
                case BC::mix_wall_currL_potS:
                {
                    r[dir] = LinOpBCType::Neumann;
                    break;
                }
                case BC::pressure_inflow:
                case BC::mass_inflow:
                {
                    r[dir] = LinOpBCType::Dirichlet;
                    break;
                }
                default:
                    Abort("get_react_scalar_bc: undefined BC type");
            };
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return r;
}
