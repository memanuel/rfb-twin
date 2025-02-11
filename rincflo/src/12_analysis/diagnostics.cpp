#include <rincflo.H>

// *********************************************************************************************************************
// Print maximum values (useful for tracking evolution)
void Rincflo::PrintMaxValues(Real time_in)
{
    for(int lev = 0; lev <= finest_level; lev++)
    {
        Print() << "Level " << lev << std::endl;
        PrintMaxVal(lev);
    }
    Print() << std::endl;
}

// *********************************************************************************************************************
// Print the maximum values of the velocity components and velocity divergence
void Rincflo::PrintMaxVal(int lev)
{
    MultiFab const& vel = m_leveldata[lev]->velocity;
    AMREX_D_TERM(
    Real max_vx = -1.0;,
    Real max_vy = -1.0;,
    Real max_vz = -1.0;)
    
    #if (AMREX_USE_EB)
    if (!vel.isAllRegular()) 
    {
        auto const& flag = EBFactory(lev).getMultiEBCellFlagFab();
        auto func_vx = 
        [=] AMREX_GPU_HOST_DEVICE 
        (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mx = max(std::abs( v(i,j,k,0) ), mx);}
            };
            Loop(b, func_b);
            return mx;
        };
        max_vx = ReduceMax(vel, flag, 0, func_vx);

        auto func_vy = 
        [=] AMREX_GPU_HOST_DEVICE 
        (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mx = max(std::abs( v(i,j,k,1) ), mx);}
            };
            Loop(b, func_b);
            return mx;
        };
        max_vy = ReduceMax(vel, flag, 0, func_vy);
        
        #if (AMREX_IS_3D)
        auto func_vz = 
        [=] AMREX_GPU_HOST_DEVICE (
            Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=,&mx] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mx = max(std::abs( v(i,j,k,2) ), mx);}
            };
            Loop(b, func_b);
            return mx;
        };
        max_vz = ReduceMax(vel, flag, 0, func_vz);
        #endif
    } // vel.isAllRegular()
    else
    #endif
    {
        auto func_vx = 
        [=] AMREX_GPU_HOST_DEVICE (
            Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                mx = max(std::abs( v(i,j,k,0) ), mx);
            };
            Loop(b, func_b);
            return mx;
        };
        max_vx = ReduceMax(vel, 0, func_vx);

        auto func_vy = 
        [=] AMREX_GPU_HOST_DEVICE (
            Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                mx = max(std::abs( v(i,j,k,1) ),mx);
            };
            Loop(b, func_b);
            return mx;
        };
        max_vy = ReduceMax(vel, 0, func_vy);

        #if (AMREX_IS_3D)
        auto func_vz = 
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mx=-1.0;
            auto func_b = 
            [=, &mx] (int i, int j, int k) noexcept
            {
                mx = max(std::abs( v(i,j,k,2) ),mx);
            };
            Loop(b, func_b);
            return mx;
        };
        max_vz = ReduceMax(vel, 0, func_vz);
        #endif
    } // else / vel.isAllRegular()

    AMREX_D_TERM(
    ReduceRealMax(max_vx);,
    ReduceRealMax(max_vy);,
    ReduceRealMax(max_vz);)
    #if (AMREX_IS_2D)
    Print() << format("max(abs(u/v))= {:f}, {:f}.\n", max_vx, max_vy);
    #else
    Print() << format("max(abs(u/v))= {:f}, {:f}, {:f}.\n", max_vx, max_vy, max_vz);
    #endif

    if (m_nspec > 0)
    {
        MultiFab const& conc = m_leveldata[lev]->conc;
        for (int l = 0; l < m_nspec; l++)
        {
            Real max_t=0.0;
            auto func = 
            [=] AMREX_GPU_HOST_DEVICE (Box const& b,Array4<Real const> const& t) -> Real
            {
                Real mx=0.0;
                auto func_b = 
                [=, &mx] (int i, int j, int k) noexcept
                {
                    mx = max(t(i,j,k,l), mx);
                };
                Loop(b, func_b);
                return mx;
            };
            max_t = ReduceMax(conc, 0, func);
            ReduceRealMax(max_t);
        }
    }

    MultiFab const& gradp = m_leveldata[lev]->gradp;
    AMREX_D_TERM(
    Real max_gpx=-1.0;,
    Real max_gpy=-1.0;,
    Real max_gpz=-1.0;)
    auto func_gpx = 
    [=] AMREX_GPU_HOST_DEVICE (Box const& b,Array4<Real const> const& gradp) -> Real
    {
        Real mx=-1.0;
        auto func_b = 
        [=, &mx] (int i, int j, int k) noexcept
        {
            mx = max(std::abs(gradp(i,j,k,0)), mx);
        }
        ;
        Loop(b, func_b);
        return mx;
    };
    max_gpx = ReduceMax(gradp, 0, func_gpx);
    
    auto func_gpy = 
    [=] AMREX_GPU_HOST_DEVICE (Box const& b,Array4<Real const> const& gradp) -> Real
    {
        Real mx=-1.0;
        auto func_b = 
        [=, &mx] (int i, int j, int k) noexcept
        {
            mx = max(std::abs(gradp(i,j,k,1)), mx);
        };
        Loop(b, func_b);
        return mx;
    };
    max_gpy = ReduceMax(gradp, 0, func_gpy);
    #if (AMREX_IS_3D)
    max_gpz = ReduceMax(gradp, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& b,Array4<Real const> const& gradp) -> Real
    {
        Real mx=-1.0;
        auto func_b = 
        [=, &mx] (int i, int j, int k) noexcept
        {
            mx = max(std::abs( gradp(i,j,k,2) ),mx);
        };
        Loop(b, func_b);
        return mx;
    });
    #endif
  
    AMREX_D_TERM(
    ReduceRealMax(max_gpx);,
    ReduceRealMax(max_gpy);,
    ReduceRealMax(max_gpz);)
  
    MultiFab const& pressure = m_leveldata[lev]->pressure;
    Real max_p=-1.0;
    auto func_p = 
    [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& p) -> Real
    {
        Real mx=-1.0;
        auto func_b =
        [=, &mx] (int i, int j, int k) noexcept
        {
            mx = max(p(i,j,k), mx);
        };
        Loop(b, func_b);
        return mx;
    };
    max_p = ReduceMax(pressure, 0, func_p);    
    ReduceRealMax(max_p);

    #if (AMREX_IS_2D)
    Print() << format("max(abs(gpx/gpy/p))= {:f}, {:f}, {:f}\n", max_gpx, max_gpy, max_p);
    #else
    Print() << format("max(abs(gpx/gpy/gpz/p))= {:f}, {:f}, {:f}, {:f}\n", max_gpx, max_gpy, max_gpz, max_p);
    #endif
}

// *********************************************************************************************************************
// Print minimum values for all the levels (useful for tracking evolution)
void Rincflo::PrintMinValues(Real time_in)
{
    for(int lev = 0; lev <= finest_level; lev++)
    {
        Print() << "Level " << lev << std::endl;
        PrintMinVal(lev);
    }
    Print() << std::endl;
}

// *********************************************************************************************************************
// Print the minimum values for a level
void Rincflo::PrintMinVal(int lev)
{
    MultiFab const& vel = m_leveldata[lev]->velocity;
    AMREX_D_TERM(
    Real min_vx=1.0e10;,
    Real min_vy=1.0e10;,
    Real min_vz=1.0e10;)
    #if (AMREX_USE_EB)
    if (!vel.isAllRegular()) 
    {
        auto const& flag = EBFactory(lev).getMultiEBCellFlagFab();

        min_vx = ReduceMin(vel, flag, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mn = min( v(i,j,k,0), mn );}
            };
            Loop(b, func_b);
            return mn;
        });
        
        min_vy = ReduceMin(vel, flag, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mn = min( v(i,j,k,1), mn );}
            };
            Loop(b, func_b);
            return mn;
        });

        #if (AMREX_IS_3D)
        min_vz = ReduceMin(vel, flag, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v, Array4<EBCellFlag const> const& f) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                if (!f(i,j,k).isCovered()) 
                    {mn = min( v(i,j,k,2), mn );}
            };
            Loop(b, func_b);
            return mn;
        });
        #endif
    } 
    else
    #endif
    {
        min_vx = ReduceMax(vel, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                mn = min(v(i,j,k,0), mn);
            };
            Loop(b, func_b);
            return mn;
        });
      
        min_vy = ReduceMax(vel, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                mn = min(v(i,j,k,1), mn);
            };
            Loop(b, func_b);
            return mn;
        });

        #if (AMREX_IS_3D)
        min_vz = ReduceMax(vel, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& v) -> Real
        {
            Real mn=1.0e10;
            auto func_b = 
            [=, &mn] (int i, int j, int k) noexcept
            {
                mn = min(v(i,j,k,2), mn);
            };
            Loop(b, func_b);
            return mn;
        });
        #endif
    }

    AMREX_D_TERM(
    ReduceRealMin(min_vx);,
    ReduceRealMin(min_vy);,
    ReduceRealMin(min_vz);)

    #if (AMREX_IS_2D)
    Print() << format("min(u/v)= {:f}, {:f} \n", min_vx, min_vy);
    #else
    Print() << format("min(u/v)= {:f}, {:f}, {:f} \n", min_vx, min_vy, min_vz);
    #endif
  
    if (m_nspec > 0)
    {
        MultiFab const& conc = m_leveldata[lev]->conc;
        for (int l = 0; l < m_nspec; l++)
        {
            Real min_t=1.0e10;
            min_t = ReduceMin(conc, 0,
            [=] AMREX_GPU_HOST_DEVICE (Box const& b,Array4<Real const> const& t) -> Real
            {
                Real mn=1.0e10;
                auto func_b = 
                [=,&mn] (int i, int j, int k) noexcept
                {
                    mn = min(t(i,j,k,l),mn);
                };
                Loop(b, func_b);
                return mn;
            });
            ReduceRealMin(min_t);
        }
    }
}

// *********************************************************************************************************************
bool Rincflo::CheckForNans(int lev)
{
    bool nans=false;

    if(do_reaction() && m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        bool epL_has_nans = m_leveldata[lev]->epotL.contains_nan();
        nans = (nans || epL_has_nans);
        if(epL_has_nans)
            {Print() << "WARNING: epL contains NaNs!!! lev: " << lev << "\n";}

        bool epS_has_nans = m_leveldata[lev]->epotS.contains_nan();
        nans = (nans || epS_has_nans);
        if(epS_has_nans)
            {Print() << "WARNING: epS contains NaNs!!! lev: " << lev << "\n";}

        bool epLO_has_nans = m_leveldata[lev]->epotL_o.contains_nan();
        nans = (nans || epLO_has_nans);
        if(epLO_has_nans)
            {Print() << "WARNING: epLO contains NaNs!!! lev: " << lev << "\n";}

        bool epSO_has_nans = m_leveldata[lev]->epotS_o.contains_nan();
        nans = (nans || epSO_has_nans);
        if(epSO_has_nans)
            {Print() << "WARNING: epSO contains NaNs!!! lev: " << lev << "\n";}
  }

    for (int it=0; it<m_nspec; it++) 
    {
        bool conc_has_nans = m_leveldata[lev]->conc.contains_nan(it);
        nans = (nans || conc_has_nans);
        if(conc_has_nans)
            {Print() << "WARNING: conc(" << it << ") contains NaNs!!! lev: " << lev << "\n";}
    }
    
    bool ro_has_nans = m_leveldata[lev]->density.contains_nan(0);
    nans = (nans || ro_has_nans);
    if (ro_has_nans)
        {Print() << "WARNING: ro contains NaNs!!! lev: " << lev << "\n";}

    bool ug_has_nans = m_leveldata[lev]->velocity.contains_nan(0);
    nans = (nans || ug_has_nans);
    if (ug_has_nans)
        {Print() << "WARNING: u contains NaNs!!! lev: " << lev << "\n";}

    bool vg_has_nans = m_leveldata[lev]->velocity.contains_nan(1);
    nans = (nans || vg_has_nans);
    if(vg_has_nans)
        {Print() << "WARNING: v contains NaNs!!! lev: " << lev << "\n";}

    #if (AMREX_IS_3D)
    bool wg_has_nans = m_leveldata[lev]->velocity.contains_nan(2);
    nans = (nans || wg_has_nans);
    if(wg_has_nans)
        {Print() << "WARNING: w contains NaNs!!! lev: " << lev << "\n";}
    #endif
  
    if(m_do_flow == true) 
    {
        bool pg_has_nans = m_leveldata[lev]->pressure.contains_nan(0);
        nans = (nans || pg_has_nans);
        if(pg_has_nans)
            {Print() << "WARNING: p contains NaNs!!! lev: " << lev << "\n";}

        bool convu_has_nans = m_leveldata[lev]->conv_dv_dt.contains_nan(0);
        nans = (nans || convu_has_nans);
        if(convu_has_nans)
            {Print() << "WARNING: convu contains NaNs!!! lev: " << lev << "\n";}

        bool convv_has_nans = m_leveldata[lev]->conv_dv_dt.contains_nan(1);
        nans = (nans || convv_has_nans);
        if(convv_has_nans)
            {Print() << "WARNING: convv contains NaNs!!! lev: " << lev << "\n";}

        #if (AMREX_IS_3D)
        bool convw_has_nans = m_leveldata[lev]->conv_dv_dt.contains_nan(2);
        nans = (nans || convw_has_nans);
        if(convw_has_nans)
            {Print() << "WARNING: convw contains NaNs!!! lev: " << lev << "\n";}
        #endif
    
        bool convuO_has_nans = m_leveldata[lev]->conv_dv_dt_o.contains_nan(0);
        nans = (nans || convuO_has_nans);
        if(convuO_has_nans)
            {Print() << "WARNING: convu_old contains NaNs!!! lev: " << lev << "\n";}

        bool convvO_has_nans = m_leveldata[lev]->conv_dv_dt_o.contains_nan(1);
        nans = (nans || convvO_has_nans);
        if(convvO_has_nans)
            {Print() << "WARNING: convv_old contains NaNs!!! lev: " << lev << "\n";}

        #if (AMREX_IS_3D)
        bool convwO_has_nans = m_leveldata[lev]->conv_dv_dt_o.contains_nan(2);
        nans = (nans || convwO_has_nans);
        if(convwO_has_nans)
            {Print() << "WARNING: convw_old contains NaNs!!! lev: " << lev << "\n";}
        #endif
    }

    for (int it=0; it<m_nspec; it++) 
    {
        bool conv_conc_has_nans = m_leveldata[lev]->conv_dconc_dt.contains_nan(it);
        nans = (nans || conv_conc_has_nans);
        if(conv_conc_has_nans)
            {Print() << "WARNING: convtra(" << it << ") contains NaNs!!! lev: " << lev << "\n";}
    }
  
    return nans;
}

// *********************************************************************************************************************
bool Rincflo::CheckForNegativeConc(int lev)
{
    bool negconc=false;

    MultiFab const& conc = m_leveldata[lev]->conc;
    Real minval;
    for (int i=0; i<m_nspec; i++) 
    {
        Real minval=conc.min(i);
        if(minval < 0.0) 
        {
            Print() << format(
                "WARNING: found NEGATIVE CONC for species index= {:d}. minvalue= {:f}, lev: {:d}. \n", i, minval, lev);
            negconc=true;
        }
    } 
    return negconc;
}

// *********************************************************************************************************************
bool Rincflo::CheckForInfs(int lev)
{
    bool infs = false;

    bool ro_has_infs = m_leveldata[lev]->density.contains_inf(0);
    infs = (infs || ro_has_infs);

    AMREX_D_TERM(
    bool ug_has_infs = m_leveldata[lev]->velocity.contains_inf(0);,
    bool vg_has_infs = m_leveldata[lev]->velocity.contains_inf(1);,
    bool wg_has_infs = m_leveldata[lev]->velocity.contains_inf(2);)
    infs = infs || AMREX_D_TERM(ug_has_infs, || vg_has_infs, || wg_has_infs);

    if(ro_has_infs)
        {Print() << format("WARNING: ro contains Infs!!! lev: {:d}\n", lev);}
    if(ug_has_infs)
        {Print() << format("WARNING: u contains Infs!!! lev: {:d}\n", lev);}
    if(vg_has_infs)
        {Print() << format("WARNING: v contains Infs!!! lev: {:d}\n", lev);}
    #if (AMREX_IS_3D)
    if(wg_has_infs)
        {Print() << format("WARNING: w contains Infs!!! lev: {:d}\n", lev);}
    #endif
    for (int it=0; it<m_nspec; it++) 
    {
        bool conc_has_infs = m_leveldata[lev]->conc.contains_inf(it);
        infs = (infs || conc_has_infs);
        if(conc_has_infs)
            {Print() << "WARNING: conc(" << it << ") contains Infs!!! lev: " << lev << "\n";}
    }

    for (int it=0; it<m_nspec; it++) 
    {
        bool conv_conc_has_infs = m_leveldata[lev]->conv_dconc_dt.contains_inf(it);
        infs = (infs || conv_conc_has_infs);
        if(conv_conc_has_infs)
            {Print() << "WARNING: conv_dconc_dt(" << it << ") contains Infs!!! lev: " << lev << "\n";}
    }

  
    if(m_do_flow == true) 
    {
        bool pg_has_infs = m_leveldata[lev]->pressure.contains_inf(0);
        infs = (infs || pg_has_infs);
        if(pg_has_infs)   Print() << "WARNING: p contains Infs!!! lev: " << lev << " \n";
        bool convu_has_infs = m_leveldata[lev]->conv_dv_dt.contains_inf(0);
        infs = (infs || convu_has_infs);
        if(convu_has_infs)   Print() << "WARNING: convu contains Infs!!! lev: " << lev << " \n";
        bool convv_has_infs = m_leveldata[lev]->conv_dv_dt.contains_inf(1);
        infs = (infs || convv_has_infs);
        if(convv_has_infs)   Print() << "WARNING: convv contains Infs!!! lev: " << lev << " \n";
        #if (AMREX_IS_3D)
        bool convw_has_infs = m_leveldata[lev]->conv_dv_dt.contains_inf(2);
        infs = (infs || convw_has_infs);
        if(convw_has_infs)
            {Print() << "WARNING: convw contains Infs!!! lev: " << lev << " \n";}
        #endif
        
        bool convuO_has_infs = m_leveldata[lev]->conv_dv_dt_o.contains_inf(0);
        infs = (infs || convuO_has_infs);
        if(convuO_has_infs)
            {Print() << "WARNING: convu_old contains Infs!!! lev: " << lev << " \n";}
        bool convvO_has_infs = m_leveldata[lev]->conv_dv_dt_o.contains_inf(1);
            {infs = (infs || convvO_has_infs);}
        if(convvO_has_infs)   Print() << "WARNING: convv_old contains Infs!!! lev: " << lev << " \n";
        #if (AMREX_IS_3D)
        bool convwO_has_infs = m_leveldata[lev]->conv_dv_dt_o.contains_inf(2);
        infs = (infs || convwO_has_infs);
        if(convwO_has_infs)   Print() << "WARNING: convw_old contains Infs!!! lev: " << lev << " \n";
        #endif
    }

    if(do_reaction() && m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        bool epL_has_infs = m_leveldata[lev]->epotL.contains_inf();
            {infs = (infs || epL_has_infs);}
        bool epS_has_infs = m_leveldata[lev]->epotS.contains_inf();
            {infs = (infs || epS_has_infs);}
        bool epLO_has_infs = m_leveldata[lev]->epotL_o.contains_inf();
            {infs = (infs || epLO_has_infs);}
        bool epSO_has_infs = m_leveldata[lev]->epotS_o.contains_inf();
            {infs = (infs || epSO_has_infs);}

        if(epL_has_infs)
            {Print() << "WARNING: epL contains Infs!!! lev: " << lev << " \n";}
        if(epS_has_infs)
            {Print() << "WARNING: epS contains Infs!!! lev: " << lev << " \n";}
        if(epLO_has_infs)
            {Print() << "WARNING: epLO contains Infs!!! lev: " << lev << " \n";}
        if(epSO_has_infs)
            {Print() << "WARNING: epSO contains Infs!!! lev: " << lev << " \n";}
    }
  
    return infs;
}
