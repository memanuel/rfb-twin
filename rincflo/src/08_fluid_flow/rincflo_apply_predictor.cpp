#include <rincflo.H>

//
// Apply predictor:
//
//  1. Use u = vel_old to compute
//
//      conv_u  = - u grad u
//      conv_r  = - div( u rho  )
//      conv_t  = - div( u trac )
//      eta_old     = visosity at m_cur_time
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau _old = div( eta ( (grad u) + (grad u)^T ) ) / rho^n
//         rhs = u + dt * ( conv + divtau_old )
//      else
//         divtau_old  = 0.0
//         rhs = u + dt * conv
//
//      eta     = eta at new_time
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho^nph )
//
//      Note that in order to add the pressure gradient terms divided by rho,
//      we convert the velocity to momentum before adding and then convert them back.
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho^nph * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho^nph )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho^nph * div ( eta_old grad ) ) u* = u^n + 
//            dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//          + dt * ( g - grad(p + p0) / rho^nph )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      u** = u* + dt * grad p / rho^nph
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho^nph ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho^nph
//
// It is assumed that the ghost cels of the old data have been filled and
// the old and new data are the same in valid region.
//
void Rincflo::ApplyPredictor (bool incremental_projection)
{
    #define FUNC_NAME "Rincflo::ApplyPredictor"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // We use the new time value for things computed on the "*" state
    Real new_time = m_cur_time + m_dt;

    // *************************************************************************************
    // Allocate space for the MAC velocities
    // *************************************************************************************
    Vector<MultiFab> u_mac(finest_level+1), v_mac(finest_level+1), w_mac(finest_level+1);
    int ngmac = nghost_mac();

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
    // DEBUG_PRINT("Initialized u_mac, v_mac, w_mac.\n");

    // *************************************************************************************
    // Allocate space for half-time density
    // *************************************************************************************
    Vector<MultiFab> density_nph;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {density_nph.emplace_back(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));}

    // Forcing terms
    Vector<MultiFab> vel_forces, tra_forces;
    Vector<MultiFab> vel_eta, tra_eta;

    // *************************************************************************************
    // Allocate space for the forcing terms
    // *************************************************************************************
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        vel_forces.emplace_back(grids[lev], dmap[lev], SpaceDim, nghost_force(),
                                MFInfo(), Factory(lev));

        if (do_AdvectConc()) 
        {
            tra_forces.emplace_back(grids[lev], dmap[lev], m_nspec, nghost_force(),
                                    MFInfo(), Factory(lev));
        }
        vel_eta.emplace_back(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));
        if (do_AdvectConc()) 
        {
            tra_eta.emplace_back(grids[lev], dmap[lev], m_nspec, 1, MFInfo(), Factory(lev));
        }
    }
    // DEBUG_PRINT("Allocated space for forcing terms: tra_forces, tra_eta.\n");

    // *************************************************************************************
    // We now define the forcing terms to use in the Godunov prediction inside the predictor
    // *************************************************************************************

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    ComputeViscosity(GetVecOfPtrs(vel_eta), get_density_old(), get_velocity_old(),  m_cur_time, 1);
    ComputeSpeciesDiffCoeff(GetVecOfPtrs(tra_eta),1);

    // *************************************************************************************
    // Compute explicit viscous term 
    // *************************************************************************************
    if (need_DivTau() || use_tensor_correction) 
    {
        ComputeDivTau(get_divtau_old(),get_velocity_old_const(), get_density_old_const(),GetVecOfConstPtrs(vel_eta));
    }

    // *************************************************************************************
    // Compute explicit diffusive terms
    // *************************************************************************************
    if (do_AdvectConc() && need_DivTau()) 
    {
        ComputeLaplacian(get_laps_old(), get_conc_old_const(), get_density_old_const(), GetVecOfConstPtrs(tra_eta));
    }

    // *************************************************************************************
    // if ( advection_type == "Godunov" ) Compute the explicit advective terms R_u^(n+1/2), R_s^(n+1/2) and R_t^(n+1/2)
    // if ( advection_type == "MOL" ) Compute the explicit advective terms R_u^n      , R_s^n       and R_t^n
    // Note that "get_conv_dconc_dt_old" returns div(rho u conc) 
    // *************************************************************************************
    ComputeConvectiveTermFull(
        get_conv_dv_dt_old(), get_conv_drho_dt_old(), get_conv_dconc_dt_old(),
        get_velocity_old_const(), get_density_old_const(), get_conc_old_const(),
        AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac), GetVecOfPtrs(w_mac)), 
        GetVecOfPtrs(vel_forces), GetVecOfPtrs(tra_forces), m_cur_time);
    
    // *************************************************************************************
    // Define local variables for lambda to capture.
    // *************************************************************************************
    Real l_dt = m_dt;
    bool l_constant_density = m_constant_density;

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (l_constant_density) 
    {
        for (int lev = 0; lev <= finest_level; lev++)
            {MultiFab::Copy(density_nph[lev], m_leveldata[lev]->density_o, 0, 0, 1, 1);}
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
                Array4<Real  const> const& rho_o  = ld.density_o.const_array(mfi);
                Array4<Real> const& rho_new       = ld.density.array(mfi);
                Array4<Real> const& rho_nph       = density_nph[lev].array(mfi);
                Array4<Real const> const& drdt    = ld.conv_drho_dt_o.const_array(mfi);

                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const Real rho_old = rho_o(i,j,k);

                    Real rho = rho_old + l_dt * drdt(i,j,k);
                    rho_nph(i,j,k) = 0.5 * (rho_old + rho);

                    rho_new(i,j,k) = rho;
                };
                ParallelFor(bx, func);
            } // mfi
        } // lev
    } // not constant density

    // *************************************************************************************
    // Compute (or if advecion_type != "MOL", re-compute) the conc forcing terms (forcing for (rho s), not for s)
    // *************************************************************************************
    if (do_AdvectConc())
        {ComputeTraForces(GetVecOfPtrs(tra_forces), GetVecOfConstPtrs(density_nph));}

    // *************************************************************************************
    // Update the conc next
    // *************************************************************************************
    int l_nspec = (do_AdvectConc()) ? m_nspec : 0;

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
                Array4<Real> const& conc          = ld.conc.array(mfi);
                Array4<Real const> const& rho     = ld.density.const_array(mfi);
                Array4<Real const> const& dtdt_o  = ld.conv_dconc_dt_o.const_array(mfi);
                Array4<Real const> const& tra_f   = (l_nspec > 0) ? tra_forces[lev].const_array(mfi)
                                                                  : Array4<Real const>{};

                if (m_diff_type == DiffusionType::Explicit) 
                {
                    Array4<Real const> const& laps_o = 
                        (l_nspec > 0) ? ld.laps_o.const_array(mfi) : Array4<Real const>{};
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        // (rho trac)^new = (rho trac)^old + dt * (
                        //                   div(rho trac u) + div (mu grad trac) + rho * f_t 
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt *
                                ( dtdt_o(i,j,k,n) + tra_f(i,j,k,n) + laps_o(i,j,k,n) );

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
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt *
                                ( dtdt_o(i,j,k,n) + tra_f(i,j,k,n) + 0.5 * laps_o(i,j,k,n) );

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
                            Real conc_new = rho_o(i,j,k)*conc_o(i,j,k,n) + l_dt *
                                ( dtdt_o(i,j,k,n) + tra_f(i,j,k,n) );

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

    } // if (do_AdvectConc)

    // *************************************************************************************
    // Define (or if advection_type != "MOL", re-define) the forcing terms, without the viscous terms 
    //    and using the half-time density
    // *************************************************************************************
    ComputeVelForces(GetVecOfPtrs(vel_forces), get_velocity_old_const(), 
                       GetVecOfConstPtrs(density_nph),
                       get_conc_old_const(), get_conc_new_const());


    // *************************************************************************************
    // Update the velocity
    // *************************************************************************************
    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& vel = ld.velocity.array(mfi);
            Array4<Real const> const& dvdt = ld.conv_dv_dt_o.const_array(mfi);
            Array4<Real const> const& vel_f = vel_forces[lev].const_array(mfi);

            if (m_diff_type == DiffusionType::Implicit) 
            {
                if (use_tensor_correction)
                {
                    Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        // Here divtau_o is the difference of tensor and scalar divtau_o!
                        AMREX_D_TERM(
                        vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0)+divtau_o(i,j,k,0));,
                        vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1)+divtau_o(i,j,k,1));,
                        vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)+divtau_o(i,j,k,2));)
                    };
                    ParallelFor(bx, func);
                } 
                else 
                {
                    auto func = 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(
                        vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0));,
                        vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1));,
                        vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2));)
                    };
                    ParallelFor(bx, func);
                }
            } 
            else if (m_diff_type == DiffusionType::Crank_Nicolson) 
            {

                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(
                    vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0)+0.5*divtau_o(i,j,k,0));,
                    vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1)+0.5*divtau_o(i,j,k,1));,
                    vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)+0.5*divtau_o(i,j,k,2));)
                };
                ParallelFor(bx, func);
            }
            else if (m_diff_type == DiffusionType::Explicit) 
            {
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(
                    vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0));,
                    vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1));,
                    vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2));)
                };
                ParallelFor(bx, func);
            }
        } // mfi
    } // lev

    // *************************************************************************************
    // Solve diffusion equation for u* but using eta_old at old time
    // *************************************************************************************
    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillphysbc_velocity(lev, new_time, m_leveldata[lev]->velocity, ng_diffusion);
            fillphysbc_density (lev, new_time, m_leveldata[lev]->density , ng_diffusion);
        }

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : 0.5*m_dt;
        DiffuseVelocity(get_velocity_new(), get_density_new(), GetVecOfConstPtrs(vel_eta), dt_diff);
    }

    // **********************************************************************************************
    // Project velocity field, update pressure
    // **********************************************************************************************
    ApplyProjection(GetVecOfConstPtrs(density_nph),new_time, m_dt, incremental_projection);

    #if (AMREX_USE_EB)
    // **********************************************************************************************
    // Over-write velocity in cells with vfrac < 1e-4
    // **********************************************************************************************
    if(m_advection_type == "MOL")
    {
        CorrectSmallCells(
            get_velocity_new(),
            AMREX_D_DECL(GetVecOfConstPtrs(u_mac), GetVecOfConstPtrs(v_mac), GetVecOfConstPtrs(w_mac)));
    }
    #endif
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
