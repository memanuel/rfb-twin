#include <rincflo.H>
#include <rincflo_derive_K.H>

// *********************************************************************************************************************
void Rincflo::CalcDivU(Real time_in)
{
    #if 0
    rincflo_set_velocity_bcs(time_in, vel);

    int bc_lo[3], bc_hi[3];
    Box domain(geom[0].Domain());

    set_ppe_bcs(bc_lo, bc_hi,
                domain.loVect(), domain.hiVect(),
                &nghost,
                bc_ilo[0]->dataPtr(), bc_ihi[0]->dataPtr(),
                bc_jlo[0]->dataPtr(), bc_jhi[0]->dataPtr(),
                bc_klo[0]->dataPtr(), bc_khi[0]->dataPtr());

    ppe_lobc = {(LinOpBCType)bc_lo[0], (LinOpBCType)bc_lo[1], (LinOpBCType)bc_lo[2]};
    ppe_hibc = {(LinOpBCType)bc_hi[0], (LinOpBCType)bc_hi[1], (LinOpBCType)bc_hi[2]};

    LPInfo lpinfo;

    //
    // This rebuilds integrals each time linop is created -- must find a better way
    //

    #if (AMREX_USE_EB)
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo, GetVecOfConstPtrs(ebfactory));
    #else
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo);
    #endif
    linop.setDomainBC(ppe_lobc,ppe_hibc);
    linop.compDivergence(GetVecOfPtrs(divu),GetVecOfPtrs(vel));
    #endif
}

// *********************************************************************************************************************
void Rincflo::ComputeStrainRateAtLevel (
    int lev, MultiFab* strainrate, MultiFab* vel, Geometry& lev_geom, Real time, int nghost)
{
    #if (AMREX_USE_EB)
    auto const& fact = EBFactory(lev);
    auto const& flags = fact.getMultiEBCellFlagFab();
    #endif
    AMREX_D_TERM(
    Real idx = 1.0 / lev_geom.CellSize(0);,
    Real idy = 1.0 / lev_geom.CellSize(1);,
    Real idz = 1.0 / lev_geom.CellSize(2);)

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(*strainrate,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(nghost);
        Array4<Real> const& sr_arr = strainrate->array(mfi);
        Array4<Real const> const& vel_arr = vel->const_array(mfi);
        const bool cover_multiple_cuts = m_cover_multiple_cuts;
        #if (AMREX_USE_EB)
        auto const& flag_fab = flags[mfi];
        auto typ = flag_fab.getType(bx);
        if (typ == FabType::covered)
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {sr_arr(i,j,k) = 0.0;};
            ParallelFor(bx, func);
        }
        if (fabIsBoundary(typ))
        {
            auto const& flag = flag_fab.const_array();
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                sr_arr(i,j,k) = rincflo_strainrate_eb(
                    i,j,k,AMREX_D_DECL(idx,idy,idz), vel_arr, cover_multiple_cuts, flag(i,j,k));
            };
            ParallelFor(bx, func);
        }
        else
        #endif
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {sr_arr(i,j,k) = rincflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);};
            ParallelFor(bx, func);
        }
    }
}

// *********************************************************************************************************************
Real Rincflo::CalcKineticEnergy () const
{
    #if 0
    #define FUNC_NAME "Rincflo::CalcKineticEnergy"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // integrated total Kinetic energy
    Real KE = 0.0;

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Real cell_vol = geom[lev].CellSize()[0]*geom[lev].CellSize()[1]*geom[lev].CellSize()[2];
        auto func = 
        [=] AMREX_GPU_HOST_DEVICE 
        (Box const& bx, Array4<Real const> const& den_arr, Array4<Real const> const& vel_arr,
         Array4<int const>  const& mask_arr) -> Real
        {
            Real KE_Fab = 0.0;
            auto func = 
            [=, &KE_Fab] (int i, int j, int k) noexcept
            {
                KE_Fab += cell_vol*mask_arr(i,j,k)*den_arr(i,j,k)*
                    ( vel_arr(i,j,k,0)*vel_arr(i,j,k,0)
                     +vel_arr(i,j,k,1)*vel_arr(i,j,k,1)
                     +vel_arr(i,j,k,2)*vel_arr(i,j,k,2));
            };
            Loop(bx, func);
            return KE_Fab;
        };
        KE += ReduceSum(*density[lev],*vel[lev],*level_mask[lev],0, func);
    }

    // total volume of grid on level 0
    Real total_vol = geom[0].ProbDomain().volume();

    KE *= 0.5/total_vol/ro_0;

    ReduceRealSum(KE);

    return KE;
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    #endif
    return 0;
}

// *********************************************************************************************************************
#if (AMREX_IS_2D)
void Rincflo::CalcVorticity (int lev, Real t, MultiFab& vort, MultiFab const& vel)
{
    #define FUNC_NAME "Rincflo::CalcVorticity"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);

    #if (AMREX_USE_EB)
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
    #endif

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

        #if (AMREX_USE_EB)
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        bool is_boundary = fabIsBoundary(typ);
        if (typ == FabType::covered)
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {vort_fab(i,j,k) = 0.0;};
            ParallelFor(bx, func);
        }
        else if (is_boundary)
        {
            const auto& flag_fab = flags.const_array();
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = -1.5;
                constexpr Real c1 = 2.0;
                constexpr Real c2 = -0.5;

                if (flag_fab(i,j,k).isCovered())
                    {vort_fab(i,j,k) = 0.0;}
                else
                {
                    Real vx, uy;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = 0.5 * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                    }
                    else
                    {
                        uy = 0.5 * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                    }
                    vort_fab(i,j,k) = vx-uy;
                }
            };
            ParallelFor(bx, func);
        }
        else
        #endif
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = 0.5 * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real uy = 0.5 * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                vort_fab(i,j,k) = vx-uy;
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
#endif
// *********************************************************************************************************************
#if (AMREX_IS_3D)
void Rincflo::CalcVorticity (int lev, Real t, MultiFab& vort, MultiFab const& vel)
{
    #define FUNC_NAME "Rincflo::CalcVorticity"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);
    const Real idz = Geom(lev).InvCellSize(2);

    #if (AMREX_USE_EB)
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
    #endif

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

        #if (AMREX_USE_EB)
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        bool is_boundary = fabIsBoundary(typ);
        if (typ == FabType::covered)
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {vort_fab(i,j,k) = 0.0;};
            ParallelFor(bx, func);
        }
        else if (is_boundary)
        {
            const auto& flag_fab = flags.const_array();
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = -1.5;
                constexpr Real c1 = 2.0;
                constexpr Real c2 = -0.5;

                if (flag_fab(i,j,k).isCovered())
                {
                    vort_fab(i,j,k) = 0.0;
                }
                else
                {
                    Real vx, wx, uy, wy, uz, vz;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                        wx = - (c0 * ccvel_fab(i  ,j,k,2)
                              + c1 * ccvel_fab(i-1,j,k,2)
                              + c2 * ccvel_fab(i-2,j,k,2)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                        wx = (c0 * ccvel_fab(i  ,j,k,2)
                            + c1 * ccvel_fab(i+1,j,k,2)
                            + c2 * ccvel_fab(i+2,j,k,2)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = 0.5 * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                        wx = 0.5 * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                        wy = - (c0 * ccvel_fab(i,j  ,k,2)
                              + c1 * ccvel_fab(i,j-1,k,2)
                              + c2 * ccvel_fab(i,j-2,k,2)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                        wy = (c0 * ccvel_fab(i,j  ,k,2)
                            + c1 * ccvel_fab(i,j+1,k,2)
                            + c2 * ccvel_fab(i,j+2,k,2)) * idy;
                    }
                    else
                    {
                        uy = 0.5 * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                        wy = 0.5 * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;
                    }
                    // Do the same in z-direction
                    if (!flag_fab(i,j,k).isConnected(0,0, 1))
                    {
                        uz = - (c0 * ccvel_fab(i,j,k  ,0)
                              + c1 * ccvel_fab(i,j,k-1,0)
                              + c2 * ccvel_fab(i,j,k-2,0)) * idz;
                        vz = - (c0 * ccvel_fab(i,j,k  ,1)
                              + c1 * ccvel_fab(i,j,k-1,1)
                              + c2 * ccvel_fab(i,j,k-2,1)) * idz;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,0,-1))
                    {
                        uz = (c0 * ccvel_fab(i,j,k  ,0)
                            + c1 * ccvel_fab(i,j,k+1,0)
                            + c2 * ccvel_fab(i,j,k+2,0)) * idz;
                        vz = (c0 * ccvel_fab(i,j,k  ,1)
                            + c1 * ccvel_fab(i,j,k+1,1)
                            + c2 * ccvel_fab(i,j,k+2,1)) * idz;
                    }
                    else
                    {
                        uz = 0.5 * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                        vz = 0.5 * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;
                    }
                    vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
                }
            };
            ParallelFor(bx, func);
        }
        else
        #endif
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = 0.5 * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real wx = 0.5 * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;

                Real uy = 0.5 * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                Real wy = 0.5 * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;

                Real uz = 0.5 * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                Real vz = 0.5 * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;

                vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
#endif

// *********************************************************************************************************************
void Rincflo::CalcVelMag (int lev, MultiFab& vmag, MultiFab const& vel)
{
    #define FUNC_NAME "Rincflo::CalcVelMag"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    #if (AMREX_USE_EB)
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
    #endif

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& vel_fab = vel.const_array(mfi);
        Array4<Real> const& vmag_fab = vmag.array(mfi);

        #if (AMREX_USE_EB)
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        bool is_boundary = fabIsBoundary(typ);
        if (typ == FabType::covered)
        {
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {vmag_fab(i,j,k) = NaN;};
            ParallelFor(bx, func);
        }
        else if (is_boundary)
        {
            const auto& flag_fab = flags.const_array();
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (!flag_fab(i,j,k).isCovered() ) 
                {
                    vmag_fab(i,j,k) = sqrt(AMREX_D_TERM(
                         vel_fab(i,j,k,0)*vel_fab(i,j,k,0),
                        +vel_fab(i,j,k,1)*vel_fab(i,j,k,1),
                        +vel_fab(i,j,k,2)*vel_fab(i,j,k,2)));
                }
                else 
                    {vmag_fab(i,j,k) = NaN;}
            };
            ParallelFor(bx, func);
        }
        else
        #endif
        {
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                vmag_fab(i,j,k) = sqrt(AMREX_D_TERM(
                    vel_fab(i,j,k,0)*vel_fab(i,j,k,0),
                    +vel_fab(i,j,k,1)*vel_fab(i,j,k,1),
                    +vel_fab(i,j,k,2)*vel_fab(i,j,k,2)));
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::CalcAdjustedPressure(int lev, MultiFab& pressure_adj, MultiFab const& pressure, bool write_nan) const
{
    #define FUNC_NAME "Rincflo::CalcAdjustedPressure"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    #if (AMREX_USE_EB)
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
    #endif

    // The geometry object for this level
    const auto& geom {Geom(lev)};
    
    // Low end of the domain in x, y, z axes
    AMREX_D_TERM(
    const Real x_lo {geom.ProbLo(0)};,
    const Real y_lo {geom.ProbLo(1)};,
    const Real z_lo {geom.ProbLo(2)};)

    // High end of the domain in x, y, z axes
    AMREX_D_TERM(
    const Real x_hi {geom.ProbHi(0)};,
    const Real y_hi {geom.ProbHi(1)};,
    const Real z_hi {geom.ProbHi(2)};)

    // Grid spacing in x, y, z axes
    AMREX_D_TERM(
    const Real dx {geom.CellSize(0)};,
    const Real dy {geom.CellSize(1)};,
    const Real dz {geom.CellSize(2)};)

    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    if (write_nan)
    {
        for(MFIter mfi(pressure, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();
            Array4<Real const> const& p_fab = pressure.const_array(mfi);
            Array4<Real> const& pa_fab = pressure_adj.array(mfi);

            #if (AMREX_USE_EB)
            const EBCellFlagFab& flags = flags_mf[mfi];
            auto typ = flags.getType(bx);
            bool is_boundary = fabIsBoundary(typ);
            if (typ == FabType::covered)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {pa_fab(i,j,k) = NaN;};
                ParallelFor(bx, func);
            }
            else if (is_boundary)
            {
                const auto& flag_fab = flags.const_array();
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (!flag_fab(i,j,k).isCovered() ) 
                    {
                        // Coordinates at this nodal point
                        AMREX_D_TERM(
                        Real x = x_lo + i * dx;,
                        Real y = y_lo + j * dy;,
                        Real z = z_lo + k * dz;)
                        // Adjustment due to pressure gradient
                        Real adj_node = AMREX_D_TERM(
                            (x - x_hi) * m_gp0[0], + (y - y_hi) * m_gp0[1], + (z - z_hi) * m_gp0[2]);
                        // Write out the adjusted pressure
                        pa_fab(i,j,k) = p_fab(i,j,k) + adj_node;
                    }
                    else 
                        {pa_fab(i,j,k) = NaN;}
                };
                ParallelFor(bx, func);
            }
            else
            #endif
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Coordinates at this nodal point
                    AMREX_D_TERM(
                    Real x = x_lo + i * dx;,
                    Real y = y_lo + j * dy;,
                    Real z = z_lo + k * dz;)
                    // Adjustment due to pressure gradient
                    Real adj_node = AMREX_D_TERM(
                        (x - x_hi) * m_gp0[0], + (y - y_hi) * m_gp0[1], + (z - z_hi) * m_gp0[2]);
                    // Write out the adjusted pressure
                    pa_fab(i,j,k) = p_fab(i,j,k) + adj_node;
                };
                ParallelFor(bx, func);
            }
        }
    }
    
    // If we're not writing out NaNs for plotting, don't need to check for covered cells
    else
    {
        for(MFIter mfi(pressure, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();
            Array4<Real const> const& p_fab = pressure.const_array(mfi);
            Array4<Real> const& pa_fab = pressure_adj.array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Coordinates at this nodal point
                AMREX_D_TERM(
                Real x = x_lo + i * dx;,
                Real y = y_lo + j * dy;,
                Real z = z_lo + k * dz;)
                // Adjustment due to pressure gradient
                Real adj_node = AMREX_D_TERM(
                    (x - x_hi) * m_gp0[0], + (y - y_hi) * m_gp0[1], + (z - z_hi) * m_gp0[2]);
                // Write out the adjusted pressure
                pa_fab(i,j,k) = p_fab(i,j,k) + adj_node;
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::CalcAdjustedGradp(int lev, MultiFab& gradp_adj, MultiFab const& gradp) const
{
    #define FUNC_NAME "Rincflo::CalcAdjustedGradp"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    #if (AMREX_USE_EB)
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
    #endif

    // The geometry object for this level
    const auto& geom {Geom(lev)};
    
    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for(MFIter mfi(gradp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& gp_fab = gradp.const_array(mfi);
        Array4<Real> const& gpa_fab = gradp_adj.array(mfi);

        #if (AMREX_USE_EB)
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        bool is_boundary = fabIsBoundary(typ);
        if (typ == FabType::covered)
        {
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                AMREX_D_TERM(
                gpa_fab(i,j,k,0) = NaN;,
                gpa_fab(i,j,k,1) = NaN;,
                gpa_fab(i,j,k,2) = NaN;);
            };
            ParallelFor(bx, func);
        }
        else if (is_boundary)
        {
            const auto& flag_fab = flags.const_array();
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (!flag_fab(i,j,k).isCovered() ) 
                {
                    // Write out the adjusted pressure
                    AMREX_D_TERM(
                    gpa_fab(i,j,k,0) = gp_fab(i,j,k,0) + m_gp0[0];,
                    gpa_fab(i,j,k,1) = gp_fab(i,j,k,1) + m_gp0[1];,
                    gpa_fab(i,j,k,2) = gp_fab(i,j,k,2) + m_gp0[2];);
                }
                else 
                {
                    AMREX_D_TERM(
                    gpa_fab(i,j,k,0) = NaN;,
                    gpa_fab(i,j,k,1) = NaN;,
                    gpa_fab(i,j,k,2) = NaN;);
                }
            };
            ParallelFor(bx, func);
        }
        else
        #endif
        {
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Write out the adjusted pressure
                AMREX_D_TERM(
                gpa_fab(i,j,k,0) = gp_fab(i,j,k,0) + m_gp0[0];,
                gpa_fab(i,j,k,1) = gp_fab(i,j,k,1) + m_gp0[1];,
                gpa_fab(i,j,k,2) = gp_fab(i,j,k,2) + m_gp0[2];);
            };
            ParallelFor(bx, func);
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
