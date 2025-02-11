#include <rincflo.H>

// *********************************************************************************************************************
// advection- diffusion step in explicit time-stepping             //
void Rincflo::AdvDiffStep_exp (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt)
{
    Vector<MultiFab*> const& dconc_dt = get_conv_dconc_dt_new();
    Vector<MultiFab*> const& conc = get_conc_new();

    // Calculate convective term
    ComputeConvectiveTermConc(dconc_dt, GetVecOfConstPtrs(conc), AMREX_D_DECL(u_mac,v_mac,w_mac));

    // Calculate diffusive term div^2 conc
    get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());

    // Apply advection and diffusion
    for (int lev = 0; lev <= finest_level; lev++)
    {
        // Perform the concentration update on this level
        auto& ld = *m_leveldata[lev];
        auto const& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            // New concentration; will be updated
            Array4<Real> const& conc = ld.conc.array(mfi);                       
            // Laplacian operator of concentration; multiply this by diffusion coefficient to get diffusive term
            Array4<Real const> const& lap_conc = ld.laps.const_array(mfi);
            // Convective term
            Array4<Real const> const& conv_dconc_dt = ld.conv_dconc_dt.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(!flag(i,j,k).isCovered()) 
                {
                    for (int n = 0; n < m_nspec; ++n) 
                        {conc(i,j,k,n) += dt*(conv_dconc_dt(i,j,k,n) + m_D_s[n]*lap_conc(i,j,k,n));}
                }
            };
            ParallelFor(bx, func);
        }
    }

    return;
}

// *********************************************************************************************************************
// advection - diffusion - reaction step in explicit time-stepping
void Rincflo::AdvDiffReactStep_exp (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt,
    bool op_split1,
    bool op_split2)
{
    Vector<MultiFab*> const& dconc_dt = get_conv_dconc_dt_new();
    Vector<MultiFab*> const& conc = get_conc_new();

    // Calculate convective term
    ComputeConvectiveTermConc(dconc_dt,GetVecOfConstPtrs(conc),AMREX_D_DECL(u_mac,v_mac,w_mac));

    // Update conc with op_split1; advection is done separately in this step
    if(op_split1) 
    {
        //partial update of conc
        int l_nspec = m_nspec;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            auto const& fact = EBFactory(lev);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                // new conc
                Array4<Real> const& conc = ld.conc.array(mfi);                        
                // conc_o to update
                Array4<Real const> const& conc_o = ld.conc_o.const_array(mfi);          
                // convective term
                Array4<Real const> const& conv_dconc_dt  = ld.conv_dconc_dt.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if(!flag(i,j,k).isCovered()) 
                    {
                        for (int n = 0; n < l_nspec; ++n)
                            {conc(i,j,k,n) = conc_o(i,j,k,n) + dt * conv_dconc_dt(i,j,k,n);}
                    }
                };
                ParallelFor(bx, func);
            }    
        }
    }

    // Get div^2 conc
    get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());

    // Update conc with op_split2; advection and diffusion are done here
    if(op_split2) 
    {
        //partial update of conc
        int l_nspec = m_nspec;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            auto const& fact = EBFactory(lev);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                // new conc
                Array4<Real> const& conc   = ld.conc.array(mfi);
                // conc_o to update
                Array4<Real const> const& conc_o = ld.conc_o.const_array(mfi);
                // laplacian (~diffusion) term
                Array4<Real const> const& lpt  = ld.laps.const_array(mfi);              
                // convective term
                Array4<Real const> const& conv_dconc_dt  = ld.conv_dconc_dt.const_array(mfi);        
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if(!flag(i,j,k).isCovered()) 
                    {
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            if(op_split1)
                                {conc(i,j,k,n) = conc(i,j,k,n) + dt * m_D_s[n] * lpt(i,j,k,n);}
                            else
                                {conc(i,j,k,n) = conc_o(i,j,k,n) + dt * (conv_dconc_dt(i,j,k,n)+m_D_s[n]*lpt(i,j,k,n));}
                        }
                    }
                };
                ParallelFor(bx, func);
            }    
        }
    }

    // Calculate source term
    CalcSourceTermAll();

    // Allocate space for conductivity "kappaS" (if used need to be defined here to be visible)
    Vector<MultiFab> kappaS;
    const int nghost_kappa=1;
    const int nghost_flxrhs=0;
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
    {
            for (int lev = 0; lev <= finest_level; ++lev) 
                {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }
    
    // Calculate conductivity
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_kappa_l < 0.0 )
        CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);
        
    // Calculate conc flux
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL) 
    {
        if(op_split2) //conc has been updated in the meantime
            {get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());}
        CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
    }

    // Update concentration 
    int l_nspec = m_nspec;

    // Update conc main loop (regardless of op_split1 and op_split2)
    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        auto const& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            // new conc
            Array4<Real> const& conc = ld.conc.array(mfi);
            // conc_o to update
            Array4<Real const> const& conc_o = ld.conc_o.const_array(mfi);
            // Laplacian (~diffusion) term
            Array4<Real const> const& lpt  = ld.laps.const_array(mfi);
            // convective term
            Array4<Real const> const& conv_dconc_dt = ld.conv_dconc_dt.const_array(mfi);
            // source (reaction) term
            Array4<Real const> const& st = ld.source.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(!flag(i,j,k).isCovered()) 
                {
                    for (int n = 0; n < l_nspec; ++n) 
                    {
                        // we've already done advection and diffusion; add source term now
                        if(op_split2)
                            {conc(i,j,k,n)=conc(i,j,k,n) + dt* m_s_pref_conc[n]*st(i,j,k);}
                        // we've done advection already; add diffusion and source term now
                        else if(op_split1)
                            {conc(i,j,k,n)=conc(i,j,k,n) + dt*(m_D_s[n]*lpt(i,j,k,n) + m_s_pref_conc[n]*st(i,j,k));}
                        // Default operator splitting applies convection, diffusion and source in one step
                        else
                            {conc(i,j,k,n) = conc_o(i,j,k,n) + 
                                dt*(conv_dconc_dt(i,j,k,n) + m_D_s[n]*lpt(i,j,k,n) + m_s_pref_conc[n]*st(i,j,k));}
                    }
                }
            };
            ParallelFor(bx, func);
        }
    }

    // Update potentials (construct RHS and solve diff-react equation)
    //( -> for epot liquid )
    if(m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        if(m_calc_flux_epotL)
            {Update_DiffReactEpotL(get_kappa_const(),get_flxrhs_const(),get_source_const());}
        else
            {Update_DiffReactEpotL(get_kappa_const(),get_source_const());}
    }

    //( -> for epot solid  )
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
        {UpdateDiffReactEpotS(GetVecOfConstPtrs(kappaS),get_source_const());}
    
    return;
}

// *********************************************************************************************************************
// solve reaction part
void Rincflo::ReactStep_exp (Real dt)
{
    // Calculate source term
    CalcSourceTermAll();

    //allocate space for conductivity "kappaS" (if used need to be defined here to be visible)
    Vector<MultiFab> kappaS;
    const int nghost_kappa=1;
    const int nghost_flxrhs=0;
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)      
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }
  
    //(Calculate conductivity)
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_kappa_l < 0.0)
        {CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);}
        
    //(Calculate conc flux)
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL) 
    {
        get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());
        CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
    }
    
    // Update concentration 
    int l_nspec = m_nspec;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        auto const& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            // new conc
            Array4<Real> const& conc = ld.conc.array(mfi);
            // source (reaction) term
            Array4<Real const> const& st = ld.source.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(!flag(i,j,k).isCovered()) 
                {
                    for (int n = 0; n < l_nspec; ++n) 
                        {conc(i,j,k,n) = conc(i,j,k,n) + dt*m_s_pref_conc[n]*st(i,j,k);}
                }
            };
            ParallelFor(bx, func);
        }    
    }

    // Update potentials (construct RHS and solve diff-react equation)
    //( -> for epot liquid )
    if(m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        if(m_calc_flux_epotL)
            {Update_DiffReactEpotL(get_kappa_const(),get_flxrhs_const(),get_source_const());}
        else
            {Update_DiffReactEpotL(get_kappa_const(),get_source_const());}
    }

    //( -> for epot solid  )
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
        UpdateDiffReactEpotS(GetVecOfConstPtrs(kappaS),get_source_const());

    return;
}

// *********************************************************************************************************************
// advection-reaction step explicit                                         //
void Rincflo::AdvReactStep_exp( 
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt,
    bool op_split,
    bool update_old)
{
    Vector<MultiFab*> const& dconc_dt = get_conv_dconc_dt_new();
    Vector<MultiFab*> const& conc = get_conc_new();

    // Calculate convective term
    ComputeConvectiveTermConc(dconc_dt,GetVecOfConstPtrs(conc),AMREX_D_DECL(u_mac,v_mac,w_mac));

    if(op_split) 
    {
        //partial update of conc
        int l_nspec = m_nspec;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            auto const& fact = EBFactory(lev);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                // new conc
                Array4<Real> const& conc = ld.conc.array(mfi);
                // conc_o to update
                Array4<Real const> const& conc_o = ld.conc_o.const_array(mfi);
                // convective term 
                Array4<Real const> const& conv_dconc_dt  = ld.conv_dconc_dt.const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if(!flag(i,j,k).isCovered()) 
                    {
                        for (int n = 0; n < l_nspec; ++n) 
                        {
                            if(update_old)
                                {conc(i,j,k,n) = conc_o(i,j,k,n) + dt * conv_dconc_dt(i,j,k,n);}
                            else
                                {conc(i,j,k,n) = conc(i,j,k,n) + dt * conv_dconc_dt(i,j,k,n);}
                        }
                    }
                });
            }    
        }
    }
    
    // Calculate source term
    CalcSourceTermAll();

    //allocate space for conductivity "kappaS" (if used need to be defined here to be visible)
    Vector<MultiFab> kappaS;
    const int nghost_kappa=1;
    const int nghost_flxrhs=0;
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)      
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {kappaS.emplace_back(grids[lev], dmap[lev], 1, nghost_kappa, MFInfo(), Factory(lev));}
    }
    
    // Calculate conductivity
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_kappa_l < 0.0)
            {CalcKappa(GetVecOfPtrs(kappaS),get_kappa(),nghost_kappa);}
        
    // Calculate conc flux
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL) 
    {
        get_reaction_scalar_op()->ComputeLaplacian(get_laps_new(), get_conc_new_const());
        CalcFluxRhs(get_flxrhs(),nghost_flxrhs,get_laps_new_const());
    }

    // Update concentration 
    int l_nspec = m_nspec;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        auto const& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            // new conc
            Array4<Real> const& conc = ld.conc.array(mfi);
            // conc_o to update
            Array4<Real const> const& conc_o = ld.conc_o.const_array(mfi);
            // convective term
            Array4<Real const> const& dconc_dt  = ld.conv_dconc_dt.const_array(mfi);
            // source (reaction) term
            Array4<Real const> const& st = ld.source.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(!flag(i,j,k).isCovered()) 
                {
                    for (int n = 0; n < l_nspec; ++n) 
                    {
                        if(op_split)
                            {conc(i,j,k,n) = conc(i,j,k,n) + dt * ( m_s_pref_conc[n]*st(i,j,k) );}
                        else 
                        {
                            if(update_old)
                            {
                                conc(i,j,k,n) = conc_o(i,j,k,n) + dt * (dconc_dt(i,j,k,n) + m_s_pref_conc[n]*st(i,j,k));
                            }
                            else
                            {
                                conc(i,j,k,n) =   conc(i,j,k,n) + dt * (dconc_dt(i,j,k,n) + m_s_pref_conc[n]*st(i,j,k));
                            }
                        }
                    }
                }
            };
            ParallelFor(bx, func);
        }    
    }
  
    // Update potentials (construct RHS and solve diff-react equation)
    //( -> for epot liquid )
    if(m_reaction_model == ReactionModel::ButlerVolmer) 
    {
        if(m_calc_flux_epotL)
            {Update_DiffReactEpotL(get_kappa_const(),get_flxrhs_const(),get_source_const());}
        else
            {Update_DiffReactEpotL(get_kappa_const(),get_source_const());}
    }

    //( -> for epot solid  )
    if(m_reaction_model == ReactionModel::ButlerVolmer && m_solve_epotS)
        {UpdateDiffReactEpotS(GetVecOfConstPtrs(kappaS),get_source_const());}
  
  return;
}
