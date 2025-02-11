#include <Convection.H>

using namespace amrex;

// *********************************************************************************************************************
void convection::compute_convective_term_full_impl (
    Box const& bx, MFIter const& mfi,
    // velocity
    Array4<Real> const& dv_dt, 
    // density
    Array4<Real> const& drho_dt, 
    // conc
    Array4<Real> const& dconc_dt, 
    AMREX_D_DECL(
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac),
    AMREX_D_DECL(
    Array4<Real const> const& fx,
    Array4<Real const> const& fy,
    Array4<Real const> const& fz),
    int const* l_conserv_velocity_d,
    int const* l_conserv_density_d,
    int const* l_conserv_conc_d,
    bool l_constant_density, 
    bool l_advect_conc, int l_nspec,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom)
{
    #define FUNC_NAME "convection::compute_convective_term"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME))

    #if (AMREX_USE_EB)
    EBCellFlagFab const& flagfab = ebfact->getMultiEBCellFlagFab()[mfi];
    Array4<EBCellFlag const> const& flag = flagfab.const_array();
    if (flagfab.getType(bx) == FabType::covered)
    {
        auto func = 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            AMREX_D_TERM(
            dv_dt(i,j,k,0) = 0.0;,
            dv_dt(i,j,k,1) = 0.0;,
            dv_dt(i,j,k,2) = 0.0;)
            drho_dt(i,j,k) = 0.0;
        };
        ParallelFor(bx, func);
        if (l_advect_conc) 
        {
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                dconc_dt(i,j,k,n) = 0.0;
            };
            ParallelFor(bx, l_nspec, func);
        }
        DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
        return;
    }

    auto typ = flagfab.getType(amrex::grow(bx,2));
    // bool is_liquid = (typ == FabType::regular);
    bool is_boundary = (typ == FabType::singlevalued) || (cover_multiple_cuts && typ == FabType::multivalued);

    Array4<Real const> AMREX_D_DECL(apx, apy, apz);
    Array4<Real const> vfrac;

    if (is_boundary) 
    {
        AMREX_D_TERM(
        apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
        apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
        apz = ebfact->getAreaFrac()[2]->const_array(mfi);)
        vfrac = ebfact->getVolFrac().const_array(mfi);

        // velocity
        int flux_comp = 0;
        convection::compute_convective_update_eb(
            bx, flux_comp, SpaceDim, dv_dt, 
            AMREX_D_DECL(fx, fy, fz), 
            flag, vfrac, 
            AMREX_D_DECL(apx, apy, apz), 
            geom, l_conserv_velocity_d);
        flux_comp += SpaceDim;

        // density
        if (!l_constant_density) 
        {
            convection::compute_convective_update_eb(
                bx, flux_comp, 1, drho_dt, 
                AMREX_D_DECL(fx, fy, fz), 
                flag, vfrac, 
                AMREX_D_DECL(apx, apy, apz),  
                geom, l_conserv_density_d);
            flux_comp += 1;
        }
        if (l_advect_conc) 
        {
            convection::compute_convective_update_eb(
                bx, flux_comp, l_nspec, dconc_dt, 
                AMREX_D_DECL(fx, fy, fz), 
                flag, vfrac, 
                AMREX_D_DECL(apx, apy, apz), 
                geom, l_conserv_conc_d);
        }
    }
    else
    #endif
    // liquid + solid cells; or all cells when not using EB
    {
        // velocity
        int flux_comp = 0;
        convection::compute_convective_update(
            bx, flux_comp, SpaceDim, dv_dt, 
            AMREX_D_DECL(fx, fy, fz), 
            AMREX_D_DECL(umac, vmac, wmac), 
            geom, l_conserv_velocity_d);
        flux_comp += SpaceDim;

        // density
        if (!l_constant_density)
        {
            convection::compute_convective_update(
                bx, flux_comp, 1, drho_dt, 
                AMREX_D_DECL(fx, fy, fz), 
                AMREX_D_DECL(umac, vmac, wmac),  
                geom, l_conserv_density_d);
            flux_comp += 1;
        }

        // conc
        if (l_advect_conc) 
        {
            convection::compute_convective_update(
                bx, flux_comp, l_nspec, dconc_dt, 
                AMREX_D_DECL(fx, fy, fz), 
                AMREX_D_DECL(umac, vmac, wmac),  
                geom, l_conserv_conc_d);
        }
    }

    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void convection::compute_convective_update (
    Box const& bx, int flux_comp, int ncomp, Array4<Real> const& dqdt,
    AMREX_D_DECL(
    Array4<Real const> const& fx,
    Array4<Real const> const& fy,
    Array4<Real const> const& fz),
    AMREX_D_DECL(
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac),
    Geometry& geom, int const* iconserv)
  {
    #define FUNC_NAME "convection::compute_convective_update"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME));

    const auto dxinv = geom.InvCellSizeArray();
    auto func = 
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        #if (AMREX_IS_3D)
        if (iconserv[n])
        {
            dqdt(i,j,k,n) = dxinv[0] * ( fx(i,j,k,flux_comp+n) - fx(i+1,j,k,flux_comp+n) )
                +           dxinv[1] * ( fy(i,j,k,flux_comp+n) - fy(i,j+1,k,flux_comp+n) )
                +           dxinv[2] * ( fz(i,j,k,flux_comp+n) - fz(i,j,k+1,flux_comp+n) );
        } 
        else 
        {
            // NOTE: in this case fx == qx, fy == qy, fz == qz
            dqdt(i,j,k,n) = 0.5*dxinv[0]*(umac(i,j,k  ) + umac(i+1,j  ,k  ))
                *                        (fx  (i,j,k,flux_comp+n) - fx  (i+1,j  ,k  ,flux_comp+n))
                +           0.5*dxinv[1]*(vmac(i,j,k  )           + vmac(i  ,j+1,k  ))
                *                        (fy  (i,j,k,flux_comp+n) - fy  (i  ,j+1,k  ,flux_comp+n)) 
                +           0.5*dxinv[2]*(wmac(i,j,k  )           + wmac(i  ,j  ,k+1))
                *                        (fz  (i,j,k,flux_comp+n) - fz  (i  ,j  ,k+1,flux_comp+n));
        }
        #else
        if (iconserv[n])
        {
            dqdt(i,j,k,n) = dxinv[0]*( fx(i  ,j,k,flux_comp+n) - fx(i+1,j,k,flux_comp+n) )
                +           dxinv[1]*( fy(i,j  ,k,flux_comp+n) - fy(i,j+1,k,flux_comp+n));
        } 
        else 
        {
            // NOTE: in this case fx == qx, fy == qy
            dqdt(i,j,k,n) = 0.5*dxinv[0]*(umac(i,j,k  )           + umac(i+1,j  ,k  ))
                *                        (fx  (i,j,k,flux_comp+n) - fx  (i+1,j  ,k  ,flux_comp+n))
                +           0.5*dxinv[1]*(vmac(i,j,k  )           + vmac(i  ,j+1,k  ))
                *                        (fy  (i,j,k,flux_comp+n) - fy  (i  ,j+1,k  ,flux_comp+n));
        }
        #endif
    };
    ParallelFor(bx, ncomp, func);

    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}

// *********************************************************************************************************************
#if (AMREX_USE_EB)
void convection::compute_convective_update_eb (
    Box const& bx, int flux_comp, int ncomp,
    Array4<Real> const& dUdt,
    AMREX_D_DECL(
    Array4<Real const> const& fx,
    Array4<Real const> const& fy,
    Array4<Real const> const& fz),
    Array4<EBCellFlag const> const& flag,
    Array4<Real const> const& vfrac,
    AMREX_D_DECL(
    Array4<Real const> const& apx,
    Array4<Real const> const& apy,
    Array4<Real const> const& apz),
    Geometry& geom, int const* iconserv)
{
    #define FUNC_NAME "convection::compute_convective_update_eb"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME));

    const auto dxinv = geom.InvCellSizeArray();
    const Box dbox   = geom.growPeriodicDomain(2);

    // We always use conservative discretization with EB
    for (int n = 0; n < ncomp; n++)
       AMREX_ALWAYS_ASSERT(iconserv[n]);
    
    auto func = 
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        #if (AMREX_IS_3D)
        if (!dbox.contains(IntVect(AMREX_D_DECL(i,j,k))) || flag(i,j,k).isCovered()) 
            {dUdt(i,j,k,n) = 0.0;} 
        else if (flag(i,j,k).isRegular()) 
        {
            dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,flux_comp+n) - fx(i+1,j,k,flux_comp+n))
                +           dxinv[1] * (fy(i,j,k,flux_comp+n) - fy(i,j+1,k,flux_comp+n))
                +           dxinv[2] * (fz(i,j,k,flux_comp+n) - fz(i,j,k+1,flux_comp+n));
        } 
        else 
        {
            dUdt(i,j,k,n) = (1.0/vfrac(i,j,k)) *
                ( dxinv[0] * (apx(i,j,k)*fx(i,j,k,flux_comp+n) - apx(i+1,j,k)*fx(i+1,j,k,flux_comp+n))
                + dxinv[1] * (apy(i,j,k)*fy(i,j,k,flux_comp+n) - apy(i,j+1,k)*fy(i,j+1,k,flux_comp+n))
                + dxinv[2] * (apz(i,j,k)*fz(i,j,k,flux_comp+n) - apz(i,j,k+1)*fz(i,j,k+1,flux_comp+n)) );
        }
        #else
        if (!dbox.contains(IntVect(AMREX_D_DECL(i,j,k))) || flag(i,j,k).isCovered()) 
            {dUdt(i,j,k,n) = 0.0;}
        else if (flag(i,j,k).isRegular()) 
        {
            dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,flux_comp+n) - fx(i+1,j,k,flux_comp+n))
                +           dxinv[1] * (fy(i,j,k,flux_comp+n) - fy(i,j+1,k,flux_comp+n));
        } 
        else 
        {
            dUdt(i,j,k,n) = (1.0/vfrac(i,j,k)) *
                ( dxinv[0] * (apx(i,j,k)*fx(i,j,k,flux_comp+n) - apx(i+1,j,k)*fx(i+1,j,k,flux_comp+n))
                + dxinv[1] * (apy(i,j,k)*fy(i,j,k,flux_comp+n) - apy(i,j+1,k)*fy(i,j+1,k,flux_comp+n)) );
        }
        #endif
    };
    ParallelFor(bx, ncomp, func);

    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}
#endif
