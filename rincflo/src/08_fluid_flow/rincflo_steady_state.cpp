#include <rincflo.H>

/******************************
steady state if
(flag) (condition)
rate_comp     :  max(|U-Uold|+|V-Vold|+|W-Wold|) / dt  < tol
rate_mag      :  sum||V-Vold|| / dt < tol
relative      :  sum||V-Vold|| / sum||Vold|| < tol
relative_rate :  sum||V-Vold|| / (sum||Vold||*dt) < tol
******************************/

bool Rincflo::SteadyStateReached()
{
    #define FUNC_NAME "Rincflo::SteadyStateReached"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    if(m_nstep < 2) 
        {return false;} 

    // Is the steady state criterion for velocity  met on each level?
    bool is_steady_state[finest_level + 1];
    // Is this a stagnant step for the flow simulation?
    bool is_stagnant_flow_step = true;

    for(int lev = 0; lev <= finest_level; lev++)
    {
        MultiFab const& vel = m_leveldata[lev]->velocity;
        MultiFab const& vel_old = m_leveldata[lev]->velocity_o;     

        auto const& flag = EBFactory(lev).getMultiEBCellFlagFab();
        Real change       = -1.0;
        if(m_tol_flow_rate_comp)    
        {
            auto func = 
            [=] AMREX_GPU_HOST_DEVICE 
            (Box const& b, Array4<Real const> const& v, Array4<Real const> const& vold,  
            Array4<EBCellFlag const> const& f) -> Real
            {
                Real mx = -1.0;
                auto func_b = 
                [=, &mx] (int i, int j, int k) noexcept
                {
                    if (!f(i,j,k).isCovered()) 
                    {
                        #if (AMREX_IS_2D)
                        mx = max(std::abs( v(i,j,k,0)-vold(i,j,k,0) ) +
                                 std::abs( v(i,j,k,1)-vold(i,j,k,1) ) ,  
                                 mx);
                        #else
                        mx = max(std::abs( v(i,j,k,0)-vold(i,j,k,0) ) +
                                 std::abs( v(i,j,k,1)-vold(i,j,k,1) ) +
                                 std::abs( v(i,j,k,2)-vold(i,j,k,2) ) ,  
                                 mx);
                        #endif
                    }
                };
                Loop(b, func_b);
                return mx;
            };
            change = ReduceMax(vel, vel_old, flag, 0, func);

            ReduceRealMax(change);
            change /= m_dt;
            }   // if m_tol_flow_rate_comp
      
        else if (m_tol_flow_rate_mag) 
        {
            MultiFab diff_vel;
            diff_vel.define(vel.boxArray(), vel.DistributionMap(), SpaceDim, 0);
            MultiFab::Copy(diff_vel,vel,0,0,SpaceDim,0);
            MultiFab::Subtract(diff_vel,vel_old,0,0,SpaceDim,0);
            Real sqchange=-1.0;
            sqchange=MultiFab::Dot(diff_vel,0,SpaceDim,0);
            change=sqrt(sqchange);
            change/=m_dt;
        }
        else if (m_tol_flow_relative) 
        {
            MultiFab diff_vel;
            diff_vel.define(vel.boxArray(), vel.DistributionMap(), SpaceDim, 0);
            MultiFab::Copy(diff_vel,vel,0,0,SpaceDim,0);
            MultiFab::Subtract(diff_vel,vel_old,0,0,SpaceDim,0);
            Real sqchange=-1.0;
            sqchange=MultiFab::Dot(diff_vel,0,SpaceDim,0);
            Real sqold=-1.0;
            sqold=MultiFab::Dot(vel_old,0,SpaceDim,0);
            change=sqrt(sqchange/sqold);
        }
        else if (m_tol_flow_relativerate) 
        {
            MultiFab diff_vel;
            diff_vel.define(vel.boxArray(), vel.DistributionMap(), SpaceDim, 0);
            MultiFab::Copy(diff_vel,vel,0,0,SpaceDim,0);
            MultiFab::Subtract(diff_vel,vel_old,0,0,SpaceDim,0);
            Real sqchange=-1.0;
            sqchange=MultiFab::Dot(diff_vel,0,SpaceDim,0);
            Real sqold=-1.0;
            sqold=MultiFab::Dot(vel_old,0,SpaceDim,0);
            change=sqrt(sqchange/sqold)/m_dt;
        }
        // Save the change in this level and the test whether it's below the tolerance
        m_change_vel[lev] = change;
        is_steady_state[lev] = (change < m_steady_state_tol);
        // Update the minimum change in velocity seen so far on this level and the stagnant step flag
        if (change < m_change_min_vel[lev])
        {
            m_change_min_vel[lev] = change;
            is_stagnant_flow_step = false;
        }
            
        if (m_verbose > 0)
        {
            Print() << "\nSteady state (tol=" << m_steady_state_tol << ") check level " << lev << std::endl; 

            if(m_tol_flow_rate_comp)
            #if (AMREX_IS_2D)
            Print() << "max (|u-uo|+|v-vo|)/dt = " ;
            #else
            Print() << "max (|u-uo|+|v-vo|+|w-wo|)/dt = " ;
            #endif
            else if(m_tol_flow_rate_mag)
                {Print() << "sum||V-Vo||/dt = " ;}
            else if(m_tol_flow_relative)
                {Print() << "sum||V-Vo||/sum||Vo|| = " ;}
            else if(m_tol_flow_relativerate)
                {Print() << "sum||V-Vo||/(sum||Vo||*dt) = " ;}
            Print() << change << std::endl;    
        }
      
    }//lev
    
    // Steady state is reached overall when the change < tolerance for every level
    bool reached = true;
    for(int lev = 0; lev <= finest_level; lev++)
        {reached = reached && is_steady_state[lev];}
    // Update counter for stagnant flow steps
    if (is_stagnant_flow_step)
        {++m_stagnant_flow_steps;}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return reached;     
}
