#include <Convection.H>
#include <Convection_conc_only.H>
#include <Godunov.H>
#include <MOL.H>
#include <Redistribution.H>

#if (AMREX_USE_EB)
#include <EBGodunov.H>
#endif

namespace Gpu = amrex::Gpu;
using amrex::Box, amrex::Array4, amrex::Real, amrex::Vector, amrex::BCRec, amrex::EBFArrayBoxFactory, amrex::Geometry,
amrex::EBCellFlagFab, amrex::EBCellFlag, amrex::FabType, amrex::MFIter, amrex::FArrayBox, amrex::Elixir;

// *********************************************************************************************************************
void
convection::compute_fluxes_conc (
    Box const& bx, MFIter const& mfi,
    Array4<Real const> const& conc,
    Array4<Real const> const& divu,
    AMREX_D_DECL(
    Array4<Real const> const& u_mac,
    Array4<Real const> const& v_mac,
    Array4<Real const> const& w_mac),
    AMREX_D_DECL(
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz),
    Array4<Real const> const& ftra,
    Vector<BCRec> const& l_bcrec_conc,
    BCRec const* l_bcrec_conc_d,
    int const* l_iconserv_conc_d,
    std::string l_advection_type,
    int l_nspec,
    bool l_godunov_ppm, bool l_godunov_use_forces_in_trans,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom,
    Real l_dt)
{
    #define FUNC_NAME "convection::compute_fluxes_conc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME))
    
    #if (AMREX_USE_EB)
    EBCellFlagFab const& flagfab = ebfact->getMultiEBCellFlagFab()[mfi];
    Array4<EBCellFlag const> const& flag = flagfab.const_array();

    // Original SD version of code did special handling for boxes that were not regular
    // MSE - corrected a crash by switching this test to boxes that *are* single valued
    auto typ = flagfab.getType(amrex::grow(bx,2));
    bool is_boundary = (typ == FabType::singlevalued) || (cover_multiple_cuts && typ == FabType::multivalued);

    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), ccc, vfrac, AMREX_D_DECL(apx, apy, apz);
    if (is_boundary) 
    {
        // Get the centers of each face
        auto fc = ebfact->getFaceCent();
        AMREX_D_TERM(
        fcx = fc[0]->const_array(mfi);,
        fcy = fc[1]->const_array(mfi);,
        fcz = fc[2]->const_array(mfi);)
        // Get the centroid
        ccc = ebfact->getCentroid().const_array(mfi);
        // Get the area fraction of each face
        auto afrac = ebfact->getAreaFrac();
        AMREX_D_TERM(
        apx = afrac[0]->const_array(mfi);,
        apy = afrac[1]->const_array(mfi);,
        apz = afrac[2]->const_array(mfi);)
        // Get the volume fraction
        vfrac = ebfact->getVolFrac().const_array(mfi);
    }
    #endif
  
    int nmaxcomp = l_nspec;

    int n_tmp_fac;
    #if (AMREX_IS_2D)
    n_tmp_fac = 10;
    #else
    n_tmp_fac = 14;
    #endif

    // n_tmp_grow is used to create tmpfab which is passed in to compute_godunov_fluxes
    // (both regular and EB) as pointer "p" and is used to hold Imx/Ipx etc ...
    int n_tmp_grow {4};

    FArrayBox tmpfab(amrex::grow(bx,n_tmp_grow), nmaxcomp*n_tmp_fac+1);
    Elixir eli = tmpfab.elixir();

    #if (AMREX_USE_EB)
    if (is_boundary)
    {
        int flux_comp = 0;
        if (l_advection_type == "Godunov")
        {
            ebgodunov::compute_godunov_fluxes(
                bx, flux_comp, l_nspec,
                AMREX_D_DECL(fx, fy, fz), 
                conc,
                AMREX_D_DECL(u_mac, v_mac, w_mac), 
                ftra, divu, l_dt, 
                l_bcrec_conc,
                l_bcrec_conc_d,
                l_iconserv_conc_d,
                tmpfab.dataPtr(), flag,
                AMREX_D_DECL(apx, apy, apz), 
                vfrac,
                AMREX_D_DECL(fcx, fcy, fcz), 
                ccc,  geom);
        }
        else
        {
            mol::compute_convective_fluxes_eb(
                bx, flux_comp, l_nspec,
                AMREX_D_DECL(fx, fy, fz), 
                conc, 
                AMREX_D_DECL(u_mac, v_mac, w_mac),
                l_bcrec_conc.data(),
                l_bcrec_conc_d,
                l_iconserv_conc_d,
                flag, 
                AMREX_D_DECL(fcx, fcy, fcz), 
                ccc, geom);
        }
        Gpu::streamSynchronize();
    }
    else
    #endif
    {
        int flux_comp = 0;
        if (l_advection_type == "Godunov")
        {
            godunov::compute_godunov_fluxes(
                bx, flux_comp, l_nspec,
                AMREX_D_DECL(fx, fy, fz), 
                conc, 
                AMREX_D_DECL(u_mac, v_mac, w_mac),
                ftra, divu, l_dt, 
                l_bcrec_conc_d,
                l_iconserv_conc_d,
                tmpfab.dataPtr(),l_godunov_ppm,
                l_godunov_use_forces_in_trans,
                geom);
        }
        else
        {
            mol::compute_convective_fluxes(
                bx, flux_comp, l_nspec, 
                AMREX_D_DECL(fx, fy, fz), 
                conc,
                AMREX_D_DECL(u_mac, v_mac, w_mac),
                l_bcrec_conc.data(),
                l_bcrec_conc_d, 
                l_iconserv_conc_d, geom);
        }
        Gpu::streamSynchronize();
    }
    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void convection::compute_convective_term_conc_impl (
    Box const& bx, 
    MFIter const& mfi,
    Array4<Real> const& dconc_dt,
    AMREX_D_DECL(
    Array4<Real const> const& u_mac,
    Array4<Real const> const& v_mac,
    Array4<Real const> const& w_mac),
    AMREX_D_DECL(
    Array4<Real const> const& fx,
    Array4<Real const> const& fy,
    Array4<Real const> const& fz),
    int const* l_conserv_conc_d,
    std::string l_advection_type,
    int l_nspec,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom)
{
    #define FUNC_NAME "convection::compute_convective_term_conc_impl"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME))

    #if (AMREX_USE_EB)
    EBCellFlagFab const& flagfab = ebfact->getMultiEBCellFlagFab()[mfi];
    Array4<EBCellFlag const> const& flag = flagfab.const_array();
    if (flagfab.getType(bx) == FabType::covered)
    {
        auto func = 
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            dconc_dt(i,j,k,n) = 0.0;
        };
        ParallelFor(bx, l_nspec, func);
        return;
    }
    
    auto typ = flagfab.getType(amrex::grow(bx,2));
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

        int flux_comp = 0;
        convection::compute_convective_update_eb(
            bx, flux_comp, l_nspec, dconc_dt, 
            AMREX_D_DECL(fx, fy, fz),
            flag, vfrac, 
            AMREX_D_DECL(apx, apy, apz), 
            geom, l_conserv_conc_d);
    }
    else
    #endif
    {
        int flux_comp = 0;
        convection::compute_convective_update(
            bx, flux_comp, l_nspec, dconc_dt, 
            AMREX_D_DECL(fx, fy, fz), 
            AMREX_D_DECL(u_mac, v_mac, w_mac), 
            geom, l_conserv_conc_d);
    }
    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}

// *********************************************************************************************************************
#if (AMREX_USE_EB) 
void
convection::redistribute_convective_term_conc (
    Box const& bx, MFIter const& mfi,
    // conc
    Array4<Real const > const& trac, 
    //initial conc udpdate
    Array4<Real> const& dtdt_tmp, 
    //final conc update
    Array4<Real> const& dtdt, 
    std::string l_redistribution_type,
    int l_nspec,
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    Geometry& geom,
    Real l_dt)
{
    #define FUNC_NAME "convection::compute_fluxes_conc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_PRINT(format("Entering function {:s}.\n", FUNC_NAME))
    
    EBCellFlagFab const& flagfab = ebfact->getMultiEBCellFlagFab()[mfi];
    Array4<EBCellFlag const> const& flag = flagfab.const_array();

    // bool regular = (flagfab.getType(amrex::grow(bx,2)) == FabType::regular);
    auto typ = flagfab.getType(amrex::grow(bx,2));
    bool is_boundary = (typ == FabType::singlevalued) || (cover_multiple_cuts && typ == FabType::multivalued);

    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), AMREX_D_DECL(apx, apy, apz);
    Array4<Real const> ccc, vfrac;

    if (is_boundary) 
    {
        AMREX_D_TERM(
        fcx = ebfact->getFaceCent()[0]->const_array(mfi);,
        fcy = ebfact->getFaceCent()[1]->const_array(mfi);,
        fcz = ebfact->getFaceCent()[2]->const_array(mfi);)
        ccc   = ebfact->getCentroid().const_array(mfi);
        AMREX_D_TERM(
        apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
        apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
        apz = ebfact->getAreaFrac()[2]->const_array(mfi);)
        vfrac = ebfact->getVolFrac().const_array(mfi);

        Box gbx = bx;
        
        if (l_redistribution_type == "StateRedist") 
            {gbx.grow(3);}
        else if (l_redistribution_type == "FluxRedist") 
            {gbx.grow(2);}

        int nmaxcomp = l_nspec;

        FArrayBox scratch_fab(gbx,nmaxcomp);
        Array4<Real> scratch = scratch_fab.array();
        Elixir eli_scratch = scratch_fab.elixir();

        redistribution::redistribute_eb(bx, l_nspec, dtdt, dtdt_tmp, trac, scratch, flag,
                        AMREX_D_DECL(apx, apy, apz), vfrac,
                        AMREX_D_DECL(fcx, fcy, fcz), ccc, geom, l_dt, l_redistribution_type);
    } 
    else 
    { 
        auto func = 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int n = 0; n < l_nspec; n++)
                {dtdt(i,j,k,n) = dtdt_tmp(i,j,k,n);}
        };
        ParallelFor(bx, func);
    }
    DEBUG_PRINT(format("Exiting function {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}
#endif
