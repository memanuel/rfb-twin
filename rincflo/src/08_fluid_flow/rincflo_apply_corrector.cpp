#include <rincflo.H>

//
// Apply corrector:
//
//  Output variables from the predictor are labelled _pred
//
//  1. Use u = vel_pred to compute
//
//      conv_u  = - u grad u
//      conv_r  = - u grad rho
//      conv_t  = - u grad trac
//      eta     = viscosity
//      divtau  = div( eta ( (grad u) + (grad u)^T ) ) / rho
//
//      conv_u  = 0.5 (conv_u + conv_u_pred)
//      conv_r  = 0.5 (conv_r + conv_r_pred)
//      conv_t  = 0.5 (conv_t + conv_t_pred)
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau  = divtau at new_time using (*) state
//      else
//         divtau  = 0.0
//      eta     = eta at new_time
//
//     rhs = u + dt * ( conv + divtau )
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho )
//
//      Note that in order to add the pressure gradient terms divided by rho,
//      we convert the velocity to momentum before adding and then convert them back.
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho * div ( eta grad ) ) u* = u^n + dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//                                                      + dt * ( g - grad(p + p0) / rho )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      u** = u* + dt * grad p / rho
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho
//
void Rincflo::ApplyCorrector()
{
    #define FUNC_NAME "Rincflo::ApplyCorrector"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // We use the new time value for things computed on the "*" state
    Real new_time = m_cur_time + m_dt;

    // *************************************************************************************
    // Allocate space for the MAC velocities
    // *************************************************************************************
    Vector<MultiFab> AMREX_D_DECL(u_mac(finest_level+1), v_mac(finest_level+1), w_mac(finest_level+1));
    int ngmac = nghost_mac();
    DEBUG_PRINT("Created empty multifab vectors for u_mac, v_mac, w_mac.\n");

    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        AMREX_D_TERM(
        u_mac[lev].define(convert(grids[lev],IntVect::TheDimensionVector(0)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));,
        v_mac[lev].define(convert(grids[lev],IntVect::TheDimensionVector(1)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));,
        w_mac[lev].define(convert(grids[lev],IntVect::TheDimensionVector(2)), dmap[lev], 1, ngmac, MFInfo(), Factory(lev));)
        if (ngmac > 0) 
        {
            AMREX_D_TERM(
            u_mac[lev].setBndry(0.0);,
            v_mac[lev].setBndry(0.0);,
            w_mac[lev].setBndry(0.0);)
        }
    }
    DEBUG_PRINT("Initialized u_mac, v_mac, w_mac.\n");

    // *************************************************************************************
    // Allocate space for half-time density
    // *************************************************************************************
    Vector<MultiFab> density_nph;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        density_nph.emplace_back(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
    }
    // DEBUG_PRINT("Initialized density_nph.\n");

    // **********************************************************************************************
    // We only reach the corrector if ! advection_type==MOL which means we don't use the forces
    //    in constructing the advection term
    // **********************************************************************************************
    Vector<MultiFab> vel_forces, tra_forces;
    Vector<MultiFab> vel_eta, tra_eta;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        vel_forces.emplace_back(grids[lev], dmap[lev], SpaceDim, nghost_force(), MFInfo(), Factory(lev));
        if (do_AdvectConc()) 
            {tra_forces.emplace_back(grids[lev], dmap[lev], m_nspec, nghost_force(), MFInfo(), Factory(lev));}
        vel_eta.emplace_back(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));
        if (do_AdvectConc()) 
            {tra_eta.emplace_back(grids[lev], dmap[lev], m_nspec, 1, MFInfo(), Factory(lev));}
    }
    // DEBUG_PRINT("Initialized tra_forces and tra_eta.\n");

    // **********************************************************************************************
    // Compute the explicit "new" advective terms R_u^(n+1,*), R_r^(n+1,*) and R_t^(n+1,*)
    // Note that "get_conv_dconc_dt_new" returns div(rho u conc)
    // *************************************************************************************
    ComputeConvectiveTermFull(
        get_conv_dv_dt_new(), get_conv_drho_dt_new(), get_conv_dconc_dt_new(),
        get_velocity_new_const(), get_density_new_const(), get_conc_new_const(),
        AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)), 
        {}, {}, new_time);

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    ComputeViscosity(GetVecOfPtrs(vel_eta), get_density_new(), get_velocity_new(), new_time, 1);
    ComputeSpeciesDiffCoeff(GetVecOfPtrs(tra_eta),1);

    // Here we create divtau of the (n+1,*) state that was computed in the predictor;
    //      we use this laps only if DiffusionType::Explicit
    if ( (m_diff_type == DiffusionType::Explicit) || use_tensor_correction ) 
    {
        ComputeDivTau(get_divtau_new(), get_velocity_new_const(),
            get_density_new_const(), GetVecOfConstPtrs(vel_eta));
    }

    if (do_AdvectConc() && m_diff_type == DiffusionType::Explicit) 
        {ComputeLaplacian(get_laps_new(), get_conc_new_const(), get_density_new_const(), GetVecOfConstPtrs(tra_eta));}

    // *************************************************************************************
    // Define local variables for lambda to capture.
    // *************************************************************************************
    Real l_dt = m_dt;
    bool l_constant_density = m_constant_density;
    int l_nspec = do_AdvectConc() ? m_nspec : 0;

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (l_constant_density)
    {
        for (int lev = 0; lev <= finest_level; lev++)
            {MultiFab::Copy(density_nph[lev], m_leveldata[lev]->density_o, 0, 0, 1, 0);}
    } 
    else 
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real const> const& rho_o  = ld.density_o.const_array(mfi);
                Array4<Real> const& rho_n        = ld.density.array(mfi);
                Array4<Real> const& rho_nph      = density_nph[lev].array(mfi);
                Array4<Real const> const& drdt_o = ld.conv_drho_dt_o.const_array(mfi);
                Array4<Real const> const& drdt   = ld.conv_drho_dt.const_array(mfi);

                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const Real rho_old = rho_o(i,j,k);
                    Real rho_new = rho_old + l_dt * 0.5*(drdt(i,j,k)+drdt_o(i,j,k));
                    rho_nph(i,j,k) = 0.5 * (rho_old + rho_new);
                    rho_n  (i,j,k) = rho_new;
                };
                ParallelFor(bx, func);
            } // mfi
        } // lev
    } // not constant density

    // *************************************************************************************
    // Compute the conc forcing terms (forcing for (rho s), not for s)
    // *************************************************************************************
    if (do_AdvectConc())
        {ComputeTraForces(GetVecOfPtrs(tra_forces),  GetVecOfConstPtrs(density_nph));}

    // *************************************************************************************
    // Update the conc next (note that dtdt already has rho in it)
    // (rho trac)^new = (rho trac)^old + dt * (
    //                   div(rho trac u) + div (mu grad trac) + rho * f_t
    // *************************************************************************************
    if (do_AdvectConc())
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real const> const& conc_o  = ld.conc_o.const_array(mfi);
                Array4<Real const> const& rho_o   = ld.density_o.const_array(mfi);
                Array4<Real      > const& conc    = ld.conc.array(mfi);
                Array4<Real const> const& rho     = ld.density.const_array(mfi);
                Array4<Real const> const& dtdt_o  = ld.conv_dconc_dt_o.const_array(mfi);
                Array4<Real const> const& dtdt    = ld.conv_dconc_dt.const_array(mfi);
                Array4<Real const> const& tra_f   = 
                    (l_nspec > 0) ? tra_forces[lev].const_array(mfi) : Array4<Real const>{};

                if (m_diff_type == DiffusionType::Explicit) 
                {
                    Array4<Real const> const& laps_o = 
                        (l_nspec > 0) ? ld.laps_o.const_array(mfi) : Array4<Real const>{};
                    Array4<Real const> const& laps   = 
                        (l_nspec > 0) ? ld.laps.const_array(mfi) : Array4<Real const>{};
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt * (
                                  0.5*(  dtdt(i,j,k,n) + dtdt_o(i,j,k,n))
                                 +0.5*(laps_o(i,j,k,n) +   laps(i,j,k,n))
                                   +    tra_f(i,j,k,n) );

                            conc_new /= rho(i,j,k);
                            conc(i,j,k,n) = conc_new;
                        }
                    };
                    ParallelFor(bx, func);
                } 
                else if (m_diff_type == DiffusionType::Crank_Nicolson) 
                {
                    Array4<Real const> const& laps_o = 
                        (l_nspec > 0) ? ld.laps_o.const_array(mfi) : Array4<Real const>{};
                    auto func =
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt * (
                                  0.5*(  dtdt(i,j,k,n) + dtdt_o(i,j,k,n))
                                 +0.5*(laps_o(i,j,k,n)                  )
                                   +    tra_f(i,j,k,n) );

                            conc_new /= rho(i,j,k);
                            conc(i,j,k,n) = conc_new;
                        }
                    };
                    ParallelFor(bx, func);
                } 
                else if (m_diff_type == DiffusionType::Implicit) 
                {
                    auto func =
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt * (
                                  0.5*(  dtdt(i,j,k,n)+dtdt_o(i,j,k,n))
                                 +      tra_f(i,j,k,n) );

                            conc_new /= rho(i,j,k);
                            conc(i,j,k,n) = conc_new;
                        }
                    };
                    ParallelFor(bx, func);
                }
            } // mfi
        } // lev
    } // if (do_AdvectConc)

    // *************************************************************************************
    // Solve diffusion equation for conc
    // *************************************************************************************
    if ( do_AdvectConc() &&
        (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit) )
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev) 
            {fillphysbc_conc(lev, new_time, m_leveldata[lev]->conc, ng_diffusion);}

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : 0.5*m_dt;
        DiffuseScalar(get_conc_new(), get_density_new(), GetVecOfConstPtrs(tra_eta), dt_diff);
    }

    // *************************************************************************************
    // Define the forcing terms to use in the final update (using half-time density)
    // *************************************************************************************
    ComputeVelForces(GetVecOfPtrs(vel_forces), get_velocity_new_const(), 
                       GetVecOfConstPtrs(density_nph), 
                       get_conc_old_const(), get_conc_new_const());

    // *************************************************************************************
    // Update velocity
    // *************************************************************************************
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto& ld = *m_leveldata[lev];

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& vel = ld.velocity.array(mfi);
            Array4<Real const> const& vel_o = ld.velocity_o.const_array(mfi);
            Array4<Real const> const& dvdt = ld.conv_dv_dt.const_array(mfi);
            Array4<Real const> const& dvdt_o = ld.conv_dv_dt_o.const_array(mfi);
            Array4<Real const> const& vel_f = vel_forces[lev].const_array(mfi);

            if (m_diff_type == DiffusionType::Explicit) 
            {
                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                Array4<Real const> const& divtau   = ld.divtau.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int idim = 0; idim < SpaceDim; ++idim) {
                        vel(i,j,k,idim) = vel_o(i,j,k,idim) + l_dt * (
                             0.5*(  dvdt_o(i,j,k,idim)+  dvdt(i,j,k,idim))
                            +0.5*(divtau_o(i,j,k,idim)+divtau(i,j,k,idim))
                            +        vel_f(i,j,k,idim) );
                    }
                };
                ParallelFor(bx, func);
            } 
            else if (m_diff_type == DiffusionType::Crank_Nicolson)
            { 
                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for (int idim = 0; idim < SpaceDim; ++idim) {
                        vel(i,j,k,idim) = vel_o(i,j,k,idim) + l_dt * (
                            0.5*(  dvdt_o(i,j,k,idim)+dvdt(i,j,k,idim))
                           +0.5*(divtau_o(i,j,k,idim)                 )
                           +        vel_f(i,j,k,idim) );
                    }
                };
                ParallelFor(bx, func);
            }
            else if (m_diff_type == DiffusionType::Implicit)
            {
                if (use_tensor_correction) 
                {
                    Array4<Real const> const& divtau   = ld.divtau.const_array(mfi);
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        for (int idim = 0; idim < SpaceDim; ++idim) 
                        {
                            // Here divtau is the difference of tensor and scalar divtau!
                            vel(i,j,k,idim) = vel_o(i,j,k,idim) + l_dt * (
                                 0.5*(  dvdt_o(i,j,k,idim)+dvdt(i,j,k,idim))
                                +        vel_f(i,j,k,idim) + divtau(i,j,k,idim));
                        }
                    };
                    ParallelFor(bx, func);
                } 
                else 
                {
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        for (int idim = 0; idim < SpaceDim; ++idim) 
                        {
                            vel(i,j,k,idim) = vel_o(i,j,k,idim) + l_dt * (
                                 0.5*(  dvdt_o(i,j,k,idim)+dvdt(i,j,k,idim))
                                +        vel_f(i,j,k,idim) );
                        }
                    };
                    ParallelFor(bx, func);
                }
            } 
        }
    }

    // **********************************************************************************************
    // Solve diffusion equation for u* at t^{n+1} but using eta at predicted new time
    // **********************************************************************************************

    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev)  
        {
            fillphysbc_velocity(lev, new_time, m_leveldata[lev]->velocity, ng_diffusion);
            fillphysbc_density (lev, new_time, m_leveldata[lev]->density , ng_diffusion);
        }

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : 0.5*m_dt;
        DiffuseVelocity(get_velocity_new(), get_density_new(), GetVecOfConstPtrs(vel_eta), dt_diff);
    }

    // **********************************************************************************************
    // 
    // Project velocity field, update pressure
    bool incremental = false;
    ApplyProjection(GetVecOfConstPtrs(density_nph),new_time, m_dt, incremental);

    #if (AMREX_USE_EB)
    // **********************************************************************************************
    // Over-write velocity in cells with vfrac < 1e-4
    // **********************************************************************************************
    CorrectSmallCells(get_velocity_new(),
        AMREX_D_DECL(GetVecOfConstPtrs(u_mac), GetVecOfConstPtrs(v_mac),GetVecOfConstPtrs(w_mac)));
    #endif

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
