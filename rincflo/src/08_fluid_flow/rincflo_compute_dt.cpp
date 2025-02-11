#include <rincflo.H>

//
// Compute new dt by using
//
//  dt= C_CFL / ( C+V + sqrt( (C+V)**2 + 4Fx/dx + 4Fy/dy + 4Fz/dz )
//
// where
//
// C = max( |U|/dx + |V|/dy + |W|/dz)    --> Convection
//
// V = 2 * max(eta/rho) * (1/dx^2 + 1/dy^2 +1/dz^2) --> Diffusion
//
// Fx, Fy, Fz = net acceleration due to external forces
//
// slightly modified with respect to
// "A Boundary Condition Capturing Method for Multiphase Incompressible Flow"
// by Kang et al. (JCP).

void Rincflo::ComputeDt (int initialization, bool explicit_diffusion)
{
    #define FUNC_NAME "Rincflo::ComputeDt"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Store the past two dt
    m_prev_prev_dt = m_prev_dt;
    m_prev_dt = m_dt;

    Real conv_cfl = 0.0;
    Real diff_cfl = 0.0;
    Real forc_cfl = 0.0;
    
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto const dxinv = geom[lev].InvCellSizeArray();
        MultiFab const& vel    = m_leveldata[lev]->velocity;
        MultiFab const& rho    = m_leveldata[lev]->density;
        MultiFab const& conc   = m_leveldata[lev]->conc;
        MultiFab const& conc_o = m_leveldata[lev]->conc_o;

        Real conv_lev = 0.0;
        Real diff_lev = 0.0;
        Real forc_lev = 0.0;

        // Make a temporary here to hold vel_forces
        MultiFab vel_forces(grids[lev], dmap[lev], SpaceDim, 0);
        ComputeVelForcesOnLevel (lev, vel_forces, vel, rho, conc_o, conc);

        #if (AMREX_USE_EB)
        if (!vel.isAllRegular()) 
        {
            auto const& flag = EBFactory(lev).getMultiEBCellFlagFab();
            auto func_cfl = 
            [=] AMREX_GPU_HOST_DEVICE 
            (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
            {
                Real mx = -1.0;
                auto func_b =
                [=, &mx] (int i, int j, int k) noexcept
                {
                    if (!f(i,j,k).isCovered()) 
                    {
                        #if (AMREX_IS_2D)
                        mx = max(std::abs(v(i,j,k,0)) * dxinv[0] + 
                                 std::abs(v(i,j,k,1)) * dxinv[1], 
                                 mx);
                        #else
                        mx = max(std::abs(v(i,j,k,0)) * dxinv[0] + 
                                 std::abs(v(i,j,k,1)) * dxinv[1] + 
                                 std::abs(v(i,j,k,2)) * dxinv[2], 
                                 mx);
                        #endif
                    }
                };
                Loop(b, func_b);
                return mx;
            };            
            conv_lev = ReduceMax(vel, flag, 0, func_cfl);
            
            if (explicit_diffusion) 
            {
                auto func = 
                [=] AMREX_GPU_HOST_DEVICE 
                (Box const& b, Array4<Real const> const& r,  Array4<EBCellFlag const> const& f)  -> Real
                {
                    Real mx = -1.0;
                    auto func_b = 
                    [=, &mx] (int i, int j, int k) noexcept
                    {
                        if (!f(i,j,k).isCovered()) 
                        {
                            Real rho_inv = 1.0/r(i,j,k);
                            mx = max(rho_inv, mx);
                        }
                    };
                    Loop(b, func_b);
                    return mx;
                };
                diff_lev = ReduceMax(rho, flag, 0, func);
                diff_lev *= m_mu;
            }
            
            // Forcing term -- new way of computing using "actual" forcing term
            auto func = 
            [=] AMREX_GPU_HOST_DEVICE 
            (Box const& b, Array4<Real const> const& vf,  Array4<EBCellFlag const> const& f) -> Real
            {
                Real mx = -1.0;
                auto func_b = 
                [=, &mx] (int i, int j, int k) noexcept
                {
                    if (!f(i,j,k).isCovered()) 
                    {
                        mx = max(
                            AMREX_D_DECL(
                            std::abs(vf(i,j,k,0))*dxinv[0],
                            std::abs(vf(i,j,k,1))*dxinv[1], 
                            std::abs(vf(i,j,k,2))*dxinv[2]), 
                            mx);
                    }
                };
                Loop(b, func_b);
                return mx;
            };
            forc_lev = ReduceMax(vel_forces, flag, 0, func);
        } 
        else
        #endif
        {
            auto func_cfl = 
            [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v) -> Real
            {
                Real mx = -1.0;
                auto func_b = 
                [=, &mx] (int i, int j, int k) noexcept
                {
                    #if (AMREX_IS_2D)
                    mx = max(
                        std::abs(v(i,j,k,0)) * dxinv[0] + 
                        std::abs(v(i,j,k,1)) * dxinv[1], 
                        mx);
                    #else
                    mx = max(
                        std::abs(v(i,j,k,0)) * dxinv[0] + 
                        std::abs(v(i,j,k,1)) * dxinv[1] + 
                        std::abs(v(i,j,k,2)) * dxinv[2], 
                        mx);
                    #endif
                };
                Loop(b, func_b);
                return mx;
            };
            conv_lev = ReduceMax(vel, 0, func_cfl);
            if (explicit_diffusion) 
            {
                auto func = 
                [=] AMREX_GPU_HOST_DEVICE 
                (Box const& b, Array4<Real const> const& r) -> Real
                {
                    Real mx = -1.0;
                    auto func_b = 
                    [=, &mx] (int i, int j, int k) noexcept
                    {
                        Real rho_inv = 1.0/r(i,j,k);
                        mx = max(rho_inv, mx);
                    };
                    Loop(b, func_b);
                    return mx;
                };
                diff_lev = ReduceMax(rho, 0, func);
                diff_lev *= m_mu;
            }

            // Forcing term -- new way of computing using "actual" forcing term
            auto func = 
            [=] AMREX_GPU_HOST_DEVICE 
            (Box const& b, Array4<Real const> const& vf) -> Real
            {
                Real mx = -1.0;
                auto func_b = 
                [=, &mx] (int i, int j, int k) noexcept
                {
                    mx = max(
                        AMREX_D_DECL(
                        std::abs(vf(i,j,k,0))*dxinv[0],
                        std::abs(vf(i,j,k,1))*dxinv[1], 
                        std::abs(vf(i,j,k,2))*dxinv[2]), 
                        mx);
                };
                Loop(b, func_b);
                return mx;
            };
            forc_lev = ReduceMax(vel_forces, 0, func);
        }

        forc_cfl = max(forc_cfl, forc_lev);
        conv_cfl = max(conv_cfl, conv_lev);
        
        #if (AMREX_IS_2D)
        diff_cfl = max(diff_cfl, diff_lev*2.*(dxinv[0]*dxinv[0] + dxinv[1]*dxinv[1]));
        #else
        diff_cfl = max(diff_cfl, diff_lev*2.*(dxinv[0]*dxinv[0] + dxinv[1]*dxinv[1] + dxinv[2]*dxinv[2]));
        #endif
    }

    Real cd_cfl;
    if (explicit_diffusion) 
    {
        ParallelAllReduce::Max<Real>({conv_cfl,diff_cfl},ParallelContext::CommunicatorSub());
        cd_cfl = conv_cfl + diff_cfl;
    } 
    else 
    {
        ParallelAllReduce::Max<Real>(conv_cfl,ParallelContext::CommunicatorSub());
        cd_cfl = conv_cfl;
    }

    ParallelAllReduce::Max<Real>(forc_cfl, ParallelContext::CommunicatorSub());

    // Combined CFL conditioner
    Real comb_cfl = cd_cfl + std::sqrt(cd_cfl*cd_cfl + 4.0 * forc_cfl);

    // Update dt
    Real dt_new;
    if (comb_cfl > 0.)
        {dt_new = m_cfl / comb_cfl;} 
    else 
    {
        // This is totally random but just a way to set a timestep 
        // when the initial velocity is zero and the forcing term is not a body force
        auto const dx = geom[finest_level].CellSizeArray();
        #if (AMREX_IS_2D)
        dt_new = min(dx[0], dx[1]); 
        #else
        dt_new = min(dx[0], dx[1], dx[2]);
        #endif
    }

    // Optionally reduce CFL for initial step
    if(initialization)
        {dt_new *= m_init_shrink;}

    // Protect against very small comb_cfl
    // This may happen, for example, when the initial velocity field
    // is zero for an inviscid flow with no external forcing
    Real eps = std::numeric_limits<Real>::epsilon();
    if(comb_cfl <= eps)
        {dt_new = 0.5 * m_dt;}

    // Don't let the timestep grow by more than 10% per step
    // unless the previous time step was unduly shrunk to match m_plot_per_exact
    // unless we are evolving flow during reaction part
    Real allowed_change_factor = 1.1;
    if( (m_dt > 0.0) && !(m_plot_per_exact > 0 && m_last_plt == m_nstep && m_nstep > 0) && m_rstep<0 )
        {dt_new = min(dt_new, allowed_change_factor * m_prev_dt);} 
    else if ( (m_dt > 0.0) && (m_plot_per_exact > 0 && m_last_plt == m_nstep && m_nstep > 0) && m_rstep<0 )
        {dt_new = min( dt_new, allowed_change_factor * max(m_prev_dt, m_prev_prev_dt) );}
    
    // Don't overshoot specified plot times
    if(m_plot_per_exact > 0.0 && 
            (std::trunc((m_cur_time + dt_new + eps) / m_plot_per_exact) > std::trunc((m_cur_time + eps) / m_plot_per_exact)))
        {dt_new = std::trunc((m_cur_time + dt_new) / m_plot_per_exact) * m_plot_per_exact - m_cur_time;}

    // Don't overshoot the final time if not running to steady state
    if(!m_steady_state && m_stop_time > 0.0)
    {
        if(m_cur_time + dt_new > m_stop_time)
            {dt_new = m_stop_time - m_cur_time;}
    }

    // Make sure the timestep is not set to zero after a m_plot_per_exact stop
    if(dt_new < eps) 
        {dt_new = 0.5 * m_dt;}

    // If using fixed time step, check CFL condition and give warning if not satisfied
    if(m_using_fixed_dt_single)
    {
        if(dt_new < m_fixed_dt)
        {
            Print() << format(
                "WARNING: fixed_dt does not satisfy CFL condition:\n"
                "max_dt by CFL      : {:12.9f}\n"
                "fixed dt specified : {:12.9f}\n",
                dt_new, m_fixed_dt);
        }
        m_dt = m_fixed_dt;
    }
    else
        {m_dt = dt_new;}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
