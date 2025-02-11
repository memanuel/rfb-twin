#include <rincflo.H>

// *********************************************************************************************************************
// init epot                                                              //
// *********************************************************************************************************************
void Rincflo::InitEpot ()
{
    // Value used to initialize epotS, potential on the wires (solid)
    const Real initvalue_epotS = m_V_electrode;

    // Value used to initialize epotL, potential in the liquid
    const Real initvalue_epotL = 0.0;

    // init values
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
        for (MFIter mfi(ld.epotL,TilingIfNotGPU()); mfi.isValid(); ++mfi)
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

            // non-const array of epot
            Array4<Real> const& epS = ld.epotS.array(mfi);
            Array4<Real> const& epL = ld.epotL.array(mfi);
            Array4<Real> const& pot_diff = ld.pot_diff.array(mfi);

            // The box contains all liquid cells
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // On liquid cells, initialize epotL if requested
                    if (m_react_reinit_epotL) 
                        {epL(i,j,k) = initvalue_epotL;}
                    // On liquid cells, epotS is not defined
                    epS(i,j,k) = NaN;
                    // DEBUG
                    // epS(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
            // The box contains cells of multiple types including some boundary cells
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // On liquid cells (covered), initialize epotL if requested and set epotS to NaN
                    if (flag(i,j,k).isRegular()) 
                    {
                        if (m_react_reinit_epotL)
                            {epL(i,j,k) = initvalue_epotL;}
                        epS(i,j,k) = NaN;
                    }
                    // On cut cells, initialize both epotS and epotL if requested, then set pot_diff
                    else if (flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                    {
                        if (m_react_reinit_epotL) 
                            {epL(i,j,k) = initvalue_epotL;}
                        if (m_react_reinit_epotS)
                            {epS(i,j,k) = initvalue_epotS;}
                        // DEBUG
                        // if (isnan(epL(i,j,k)))
                        // {
                        //     epL(i,j,k) = initvalue_epotL;
                        //     Print() << format("***** epL({:d},{:d},{:d}) was NaN in checkpoint.\n", i, j, k);
                        // }
                        // Set pot_diff using the values just set above
                        pot_diff(i,j,k) = epL(i,j,k) - epS(i,j,k);
                    }
                    // On covered cells, initialize epotS if requested and set epotL to zero
                    // else if (flag(i,j,k).isCovered()) 
                    else
                    {
                        if (m_react_reinit_epotS)
                            {epS(i,j,k) = initvalue_epotS;}
                        epL(i,j,k) = 0.0;
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
                    // Initialize epotS if requested and set epotL to zero
                    if (m_react_reinit_epotS) 
                        {epS(i,j,k) = initvalue_epotS;}
                    epL(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
        }
    }
 
    // We also want to intialize epotS_o and epotL_o so the first step is correct
    CopyNewToOld_epotL();
    CopyNewToOld_epotS();

    // Patch epotS and epotL
    const int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        fillpatch_epotS(lev, m_t_old[lev], m_leveldata[lev]->epotS_o, ng);
        fillpatch_epotL(lev, m_t_old[lev], m_leveldata[lev]->epotL_o, ng);
        fillpatch_epotS(lev, m_t_new[lev], m_leveldata[lev]->epotS  , ng);
        fillpatch_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL  , ng);
    }
}

// *********************************************************************************************************************
// update value of imposed epotS                                          //
// *********************************************************************************************************************
void Rincflo::UpdateInitEpotS (Real value_epotS)
{
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
        for (MFIter mfi(ld.epotS,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            #if (AMREX_USE_EB)
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_solid = fabIsSolid(typ);
            bool is_boundary = fabIsBoundary(typ);
            #endif

            // non-const array of epot
            Array4<Real> const& epS = ld.epotS.array(mfi);

            // solid
            if (is_solid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    epS(i,j,k) = value_epotS;
                };
                ParallelFor(bx, func);
            }
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (flag(i,j,k).isCovered() || flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                        {epS(i,j,k) = value_epotS;}
                };
                ParallelFor(bx, func);
            }
        }
    } 
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// calculate conductivity for electric potentials                         //
// -----------------------------------------------------------------------//
void Rincflo::CalcKappa (Vector<MultiFab*> const& kappaS, Vector<MultiFab*> const& kappaL, int nghost)
{
    // solid:
    if(m_solve_epotS) 
    {
        for (int lev = 0; lev <= finest_level; ++lev)    
            {kappaS[lev]->setVal(m_kappa_s,0,1,nghost);}
    }

    // liquid:
    // might be input constant
    if(m_kappa_l > 0.0 ) 
    {
        for (int lev = 0; lev <= finest_level; ++lev)    
            {kappaL[lev]->setVal(m_kappa_l,0,1,nghost);}
    }
    // or calculated based on local concentrations
    else 
    {
        Vector<MultiFab const*> const& conc = get_conc_new_const();
        for (int lev = 0; lev <= finest_level; ++lev)    
        {
            #if (AMREX_USE_EB)
            auto const& fact = EBFactory(lev);
            auto const& flags = fact.getMultiEBCellFlagFab();
            #endif
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(*kappaL[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                #if (AMREX_USE_EB)
                EBCellFlagFab const& flag_fab = flags[mfi];
                Array4<EBCellFlag const> const& flag = flag_fab.const_array();
                auto typ = flag_fab.getType(bx);
                bool is_liquid = fabIsLiquid(typ);
                bool is_boundary = fabIsBoundary(typ);
                #endif

                // Arrays used in calculating the conductivity
                Array4<Real> const& kL = kappaL[lev]->array(mfi);
                Array4<Real const> const& con = conc[lev]->const_array(mfi);

                // The array of diffusion coefficients x charge number squared to use
                const amrex::Vector<amrex::Real>& Dzsq = m_extra_species ? m_Dzsq_ex : m_Dzsq;

                // all liquid
                if (is_liquid)
                {
                    auto func = 
                    [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        kL(i,j,k)= 0.0;
                        for(int n=0; n < m_nspec ; n++) 
                        {
                            kL(i,j,k) += Dzsq[n]*con(i,j,k,n);
                        }
                        kL(i,j,k)*=m_F2_over_RT_const;
                    };
                    ParallelFor(bx, func);
                }
                // some boundary cells
                else if (is_boundary)
                {
                    auto func = 
                    [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        // Set RHS as above for both liquid and boundary cells
                        if(flag(i,j,k).isRegular() || flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                        {
                            kL(i,j,k)= 0.0;
                            for(int n=0; n < m_nspec ; n++) 
                            {
                                kL(i,j,k) += Dzsq[n]*con(i,j,k,n);
                            }
                            kL(i,j,k)*=m_F2_over_RT_const;
                        }
                        else 
                            {kL(i,j,k) = 0.0;}
                    };
                    ParallelFor(bx, func);
                }
                // solid
                else
                {
                    auto func = 
                    [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        kL(i,j,k)= 0.0;
                    };
                    ParallelFor(bx, func);
                }

                // if (flagfab.getType(bx) != FabType::covered)
                // {
                //     Array4<EBCellFlag const> const& flag = flagfab.const_array();          
                //     Array4<Real> const& kL = kappaL[lev]->array(mfi);
                //     Array4<Real const> const& con = conc[lev]->const_array(mfi);
                //     auto func = 
                //     [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                //     {
                //         // TODO - update treatment of cut cells
                //         //same treatment in regular and cut-cells...
                //         if ( !flag(i,j,k).isCovered() ) 
                //         {
                //             kL(i,j,k)= 0.0;
                //             for(int n=0; n < m_nspec ; n++) 
                //             {
                //                 if(m_extra_species==true) 
                //                     {kL(i,j,k) += m_Dzsq_ex[n]*con(i,j,k,n);}
                //                 else
                //                     {kL(i,j,k) += m_Dzsq[n]*con(i,j,k,n);}
                //             }
                //             kL(i,j,k)*=m_F2_over_RT_const;
                //         }
                //         else 
                //             {kL(i,j,k)=0.0;}
                //     };
                //     ParallelFor(bx, func);
                // }
                // else 
                // { 
                //     //everything covered
                //     Array4<Real> const& kL=kappaL[lev]->array(mfi);
                //     auto func = 
                //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                //         {kL(i,j,k)=0.0;};
                //     ParallelFor(bx, func);
                // }
            }
        }
    }
}

// *********************************************************************************************************************
// ------------------------------------------------------------------------------------------------------  //
// calculate current due to species concentrations fluxes, it will be rhs in the diff-react eqs for epotL  //
// ------------------------------------------------------------------------------------------------------  //
// input is laplacian of concentration, already calculated
void Rincflo::CalcFluxRhs (Vector<MultiFab*> const& flxrhs, int nghost, Vector<MultiFab const*> const& laps)
{
    const int ncomp = m_nspec;
    for (int lev = 0; lev <= finest_level; lev++)
    {
        // Initialize the flux to zero
        flxrhs[lev]->setVal(0.0,0,1,nghost); 
        // Embedded boundaries
        #if (AMREX_USE_EB)
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
        #endif
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*flxrhs[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_liquid = fabIsLiquid(typ);
            bool is_boundary = fabIsBoundary(typ);

            // The flux we are calculating here
            Array4<Real> const& flx = flxrhs[lev]->array(mfi);
            // The laplacian of the concentration as a constant array
            Array4<Real const> const& div2c = laps[lev]->const_array(mfi);
            // The array of diffusion coefficients x charge number to use
            const amrex::Vector<amrex::Real>& Dz = m_extra_species ? m_Dz_ex : m_Dz;

            // all liquid
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    for(int n=0; n<ncomp; n++) 
                    {
                        flx(i,j,k) += Dz[n] * div2c(i,j,k,n);
                    }
                    flx(i,j,k) *= m_F_const;
                };
                ParallelFor(bx, func);
            }
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (flag(i,j,k).isRegular() || flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                    {
                        for(int n=0; n<ncomp; n++) 
                        {
                            flx(i,j,k) += Dz[n] * div2c(i,j,k,n);
                        }
                        flx(i,j,k) *= m_F_const;
                    }
                };
                ParallelFor(bx, func);
            }

            // EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];      
            // if (flagfab.getType(bx) != FabType::covered)
            // {
            //     Array4<EBCellFlag const> const& flag = flagfab.const_array();
            //     Array4<Real> const& flx=flxrhs[lev]->array(mfi);
            //     Array4<Real const> const& div2c=laps[lev]->const_array(mfi);
            //     int ncomp = m_nspec;
            //     auto func = 
            //     [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            //     {
            //         if(!flag(i,j,k).isCovered()) 
            //         {
            //             for(int n=0;n<ncomp;n++) 
            //             {
            //                 if(m_extra_species==true)
            //                     {flx(i,j,k) += m_Dz_ex[n]*div2c(i,j,k,n);}
            //                 else
            //                     {flx(i,j,k) += m_Dz[n]*div2c(i,j,k,n);}
            //             }
            //             flx(i,j,k) *= m_F_const;
            //         }
            //     };
            //     Loop(bx, func);
            // }
        }
    }     
}

// *********************************************************************************************************************
//$ more testing needed, and can be improved (e.g. read from input file the parameters)
Real Rincflo::UpdateBCMem( ) 
{
    Real bc_value=0.0;

    Orientation ori;
    if(m_mem_loc == MembraneLocation::ylo) {ori=Orientation(Direction::y,Orientation::low);}
  
    // calculate value and set bc
    if( m_membc_type == MembraneBCType::potential) 
    {  

        //given area-specific membrane resistance
        Real avcurrd= CalcAvgCurrentDensity();
        // area-specific membrane resistance
        // typically range 0.07-0.1 Ohm*cm^2 (in Gerhardt/Wong/Aziz measured 0.625 Ohm*cm^2)
        m_mem_Res=0.1e-4; // Ohm*m^2
        // pot: phi_m = r_m * i_m
        bc_value = m_mem_Res * avcurrd;

        if(m_verbose > 3) 
        { 
            //mainly for debug
            Print()  << " bc_value= " << bc_value << " avcurrd= " << avcurrd << std::endl ;
        }
        m_bc_epotL[ori]=bc_value;   
    }
    else if( m_membc_type == MembraneBCType::current) 
    {

        //compute average current density
        Real avcurrd= CalcAvgCurrentDensity();

        // ent= -kappa dphi/dn -> bcvalue= current/kappa
        
        Real kappa_m=33.4; //conductivity at or of membrane?
        bc_value = avcurrd/kappa_m;
        
        m_bc_epotL[ori]=bc_value;
        
        if(m_verbose > 3) 
        { 
            //mainly for debug
            Print() << "avcurrd= " << avcurrd << " bc_value= " << bc_value << std::endl ;
        }
    
    }
    else 
        {Abort("UpdateBCMem: why are we here?");}

    return bc_value;
}

// *********************************************************************************************************************
// construct RHS and solve diff-react equation for epot in the solid              //
// *********************************************************************************************************************
void Rincflo::UpdateDiffReactEpotS (Vector<MultiFab const*> const& kappa, Vector<MultiFab const*> const& source)
{
    Vector<MultiFab> rhs_arr;
    for (int lev = 0; lev <= finest_level; ++lev) 
        //$ nghost=1?
        {rhs_arr.emplace_back(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));}
  
    // construct RHS =  source
    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        auto const& fact = EBFactory(lev);
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(rhs_arr[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            const bool cover_multiple_cuts = m_cover_multiple_cuts;
            Array4<Real const> vfrac = fact.getVolFrac().const_array(mfi);
            Array4<Real> const& rhs   = rhs_arr[lev].array(mfi);
            Array4<Real const> const& st = source[lev]->const_array(mfi);  //source (reaction) term
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(flag(i,j,k).isBoundary(cover_multiple_cuts) && (vfrac(i,j,k) < vfrac_almost_one) ) 
                    {rhs(i,j,k) = m_s_pref_epotS * st(i,j,k);}
            };
            ParallelFor(bx, func);
        }
    }

    //adjust number ghost cells (and fill them with inhomog. b.c. if any)
    const int ng_diffreact = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillphysbc_epotS(lev, m_t_new[lev], m_leveldata[lev]->epotS, ng_diffreact);}

    //solve equation (in: rhs, epotS as initial guess ;  out: updated epotS)
    get_reaction_esolid_op()->react_epot_solid(get_epotS_new(),GetVecOfPtrs(rhs_arr),GetVecOfConstPtrs(kappa));

    //re-update number ghost cells
    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillpatch_epotS(lev, m_t_new[lev], m_leveldata[lev]->epotS, ng);}
  
}

// *********************************************************************************************************************
// construct RHS and solve diff-react equation for epot in the liquid             //
// *********************************************************************************************************************
void Rincflo::Update_DiffReactEpotL (Vector<MultiFab const*> const& kappa,
                      Vector<MultiFab const*> const& flxrhs,
                      Vector<MultiFab const*> const& source)
{
    Vector<MultiFab> rhs_arr;
    for (int lev = 0; lev <= finest_level; ++lev)
        {rhs_arr.emplace_back(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));}
  
    // construct RHS =  source + flxrhs
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
        for (MFIter mfi(rhs_arr[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            #if (AMREX_USE_EB)
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_liquid = fabIsLiquid(typ);
            bool is_boundary = fabIsBoundary(typ);
            #endif

            // Constant arrays used in solving linear system
            Array4<Real> const& rhs = rhs_arr[lev].array(mfi);
            Array4<Real const> const& flx = flxrhs[lev]->const_array(mfi);      // conc flux term
            Array4<Real const> const& st = source[lev]->const_array(mfi);       // source (reaction) term

            // box is all liquid
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // DEBUG
                    // rhs(i,j,k) = m_s_pref_epotL * st(i,j,k) + flx(i,j,k);
                    // All liquid cells, so there is no source term
                    rhs(i,j,k) = flx(i,j,k);
                };
                ParallelFor(bx, func);
            }
            // box has some boundary cells
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // DEBUG
                    // Set RHS as above for both liquid and boundary cells
                    // if(flag(i,j,k).isRegular() || flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                    //     {rhs(i,j,k) = m_s_pref_epotL * st(i,j,k) + flx(i,j,k);}

                    // Boundary cells have two terms in the RHS
                    if(flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                        {rhs(i,j,k) = m_s_pref_epotL * st(i,j,k) + flx(i,j,k);}
                    // Liquid cells have just the flux terms
                    else if(flag(i,j,k).isRegular()) 
                        {rhs(i,j,k) = flx(i,j,k);}
                    // Solid cells are zero
                    else 
                        {rhs(i,j,k) = 0.0;}
                };
                ParallelFor(bx, func);
            }
            // box is all solid
            else
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    rhs(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }

            // auto func = 
            // [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            // {
            //     if(!flag(i,j,k).isCovered()) 
            //         {rhs(i,j,k) = m_s_pref_epotL * st(i,j,k) + flx(i,j,k);}
            //     else 
            //         {rhs(i,j,k) = 0.0;}
            // };
            // ParallelFor(bx, func);
        }
    }
  
    // adjust number of ghost cells (and fill them with inhomog. b.c. if any)
    const int ng_diffreact = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillphysbc_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL, ng_diffreact);}

    // solve equation (in: rhs, epotL as initial guess ;  out: updated epotL)
    get_reaction_eliquid_op()->react_epot_liquid(get_epotL_new(), GetVecOfPtrs(rhs_arr), GetVecOfConstPtrs(kappa));
  
    // re-update number ghost cells
    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillpatch_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL, ng);}
}

// *********************************************************************************************************************
// construct RHS and solve diff-react equation for epot in the liquid  (no flx)
// *********************************************************************************************************************
void Rincflo::Update_DiffReactEpotL (Vector<MultiFab const*> const& kappa, Vector<MultiFab const*> const& source)
{
    Vector<MultiFab> rhs_arr;
    for (int lev = 0; lev <= finest_level; ++lev) 
        //$ nghost=1 ??
        {rhs_arr.emplace_back(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));}

    //construct RHS =  source   (no flx)
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
        for (MFIter mfi(rhs_arr[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            #if (AMREX_USE_EB)
            EBCellFlagFab const& flag_fab = flags[mfi];
            Array4<EBCellFlag const> const& flag = flag_fab.const_array();
            auto typ = flag_fab.getType(bx);
            bool is_liquid = fabIsLiquid(typ);
            bool is_boundary = fabIsBoundary(typ);
            #endif

            // Constant arrays used in solving linear system
            Array4<Real> const& rhs   = rhs_arr[lev].array(mfi);
            Array4<Real const> const& st = source[lev]->const_array(mfi);      // source (reaction) term

            // all liquid
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    rhs(i,j,k) = m_s_pref_epotL * st(i,j,k);
                };
                ParallelFor(bx, func);
            }
            // some boundary cells
            else if (is_boundary)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Set RHS as above for both liquid and boundary cells
                    if(flag(i,j,k).isRegular() || flag(i,j,k).isBoundary(m_cover_multiple_cuts)) 
                        {rhs(i,j,k) = m_s_pref_epotL * st(i,j,k);}
                    else 
                        {rhs(i,j,k) = 0.0;}
                };
                ParallelFor(bx, func);
            }
            // solid
            else
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    rhs(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }

            // auto func = 
            // [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            // {
            //     if(!flag(i,j,k).isCovered()) 
            //         {rhs(i,j,k) = m_s_pref_epotL * st(i,j,k);}
            //     else 
            //         {rhs(i,j,k) = 0.0;}
            // };
            // ParallelFor(bx, func);
        }
    }

    // adjust number ghost cells (and fill them with inhomog. b.c. if any)
    const int ng_diffreact = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillphysbc_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL, ng_diffreact);}

    // solve equation (in: rhs, epotL as initial guess ;  out: updated epotL)
    get_reaction_eliquid_op()->react_epot_liquid(get_epotL_new(), GetVecOfPtrs(rhs_arr), GetVecOfConstPtrs(kappa));
  
    // re-update number ghost cells
    int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
        {fillpatch_epotL(lev, m_t_new[lev], m_leveldata[lev]->epotL, ng);}  
}

// needed to set-up coefficients for epot equations
Array<MultiFab,SpaceDim>
Rincflo::AverageEpotScalToFaces (int lev, int comp, MultiFab const& cc_scal) const
{
    const auto& ba = cc_scal.boxArray();
    const auto& dm = cc_scal.DistributionMap();
    const auto& fact = cc_scal.Factory();
    MultiFab cc(cc_scal, amrex::make_alias, comp, 1);
    Array<MultiFab,SpaceDim> r{
        AMREX_D_DECL(
        MultiFab(convert(ba,IntVect::TheDimensionVector(0)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(1)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(2)), dm, 1, 0, MFInfo(), fact))};

    // kappa is anyway defined at cell centers (calc from conc), same as conc to be used for divcgpot (electromigration)
    amrex::average_cellcenter_to_face(GetArrOfPtrs(r), cc, Geom(lev)); 
    
    FixupScalOnDomainFaces(lev, r, cc);
    return r;
}

// *********************************************************************************************************************
void
Rincflo::FixupScalOnDomainFaces (int lev, Array<MultiFab,SpaceDim>& fc, MultiFab const& cc) const
{
    const Geometry& gm = Geom(lev);
    const Box& domain = gm.Domain();
    MFItInfo mfi_info{};
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(cc,mfi_info); mfi.isValid(); ++mfi) 
    {
        Box const& bx = mfi.validbox();
        Array4<Real const> const& cca = cc.const_array(mfi);

        int idim = 0;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                ParallelFor(amrex::bdryLo(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                });
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                ParallelFor(amrex::bdryHi(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i-1,j,k);
                });
            }
        }

        idim = 1;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                ParallelFor(amrex::bdryLo(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                });
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                ParallelFor(amrex::bdryHi(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j-1,k);
                });
            }
        }
        #if (AMREX_IS_3D)
        idim = 2;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                ParallelFor(amrex::bdryLo(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                });
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                ParallelFor(amrex::bdryHi(bx, idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k-1);
                });
            }
        }
        #endif
    }
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// get reaction operator for epot solid                                   //
// -----------------------------------------------------------------------//
ReactionEsolidOp* Rincflo::get_reaction_esolid_op ()
{
    if (!m_reaction_esolid_op) 
        {m_reaction_esolid_op.reset(new ReactionEsolidOp(this));}
    return m_reaction_esolid_op.get();
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// get reaction operator for epot liquid                                  //
// -----------------------------------------------------------------------//
ReactionEliquidOp* Rincflo::get_reaction_eliquid_op ()
{
    if (!m_reaction_eliquid_op) 
        {m_reaction_eliquid_op.reset(new ReactionEliquidOp(this));}
    return m_reaction_eliquid_op.get();
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// get b.c. for reaction epot liquid                                      //
// -----------------------------------------------------------------------//
Array<LinOpBCType,SpaceDim>
Rincflo::get_react_eliquid_bc (Orientation::Side side) const noexcept
{
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
                case BC::pressure_inflow:
                case BC::pressure_outflow:
                case BC::slip_wall:
                case BC::no_slip_wall:
                case BC::mass_inflow:
                case BC::mix_wall_potS:
                {
                    r[dir] = LinOpBCType::Neumann;
                    break;
                }
                case BC::charging_wall_pot:
                case BC::mix_wall_potL:
                case BC::mix_wall_potL_currS:
                {
                    r[dir] = LinOpBCType::Dirichlet;
                    break;
                }
                case BC::charging_wall_cur:
                case BC::mix_wall_currL_potS:
                {
                    r[dir] = LinOpBCType::inhomogNeumann;
                    break;
                }
                default:
                    Abort("get_react_eliquid_bc: undefined BC type");
            };
        }
    }
    return r;
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// get b.c. for reaction epot solid                                       //
// -----------------------------------------------------------------------//
Array<LinOpBCType,SpaceDim>
Rincflo::get_react_esolid_bc (Orientation::Side side) const noexcept
{
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
                case BC::pressure_inflow:
                case BC::pressure_outflow:
                case BC::slip_wall:
                case BC::no_slip_wall:
                case BC::mass_inflow:
                case BC::mix_wall_potL:
                {
                    r[dir] = LinOpBCType::Neumann;
                    break;
                }
                case BC::charging_wall_pot:
                case BC::mix_wall_potS:
                case BC::mix_wall_currL_potS:
                {
                    r[dir] = LinOpBCType::Dirichlet;
                    break;
                }
                case BC::charging_wall_cur:
                case BC::mix_wall_potL_currS:
                {
                    r[dir] = LinOpBCType::inhomogNeumann;
                    break;
                }
                default:
                    {Abort("get_react_esolid_bc: undefined BC type");}
            };
        }
    }
    return r;
}
