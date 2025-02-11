#include <Convection.H>
#include <Godunov.H>
#include <MOL.H>

#if (AMREX_USE_EB)
#include <EBGodunov.H>
#endif

namespace Gpu = amrex::Gpu;

using amrex::Box, amrex::MFIter, amrex::Array4, amrex::Real, amrex::Vector, amrex::SpaceDim, amrex::Geometry, amrex::BCRec,
amrex::EBFArrayBoxFactory, amrex::EBCellFlagFab, amrex::EBCellFlag, amrex::FabType, amrex::BoxArray,
amrex::FArrayBox, amrex::Elixir;

// *********************************************************************************************************************
void convection::compute_fluxes (
    Box const& bx, MFIter const& mfi,
    Array4<Real const> const& vel,
    Array4<Real const> const& rho,
    Array4<Real const> const& rhoconc,
    Array4<Real const> const& divu,
    AMREX_D_DECL(
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac),
    AMREX_D_DECL(
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz),
    Array4<Real const> const& fvel,
    Array4<Real const> const& ftra,
    Vector<BCRec> const& l_bcrec_velocity,
    BCRec const* l_bcrec_velocity_d,
    int const* l_iconserv_velocity_d,
    Vector<BCRec> const& l_bcrec_density,
    BCRec const* l_bcrec_density_d,
    int const* l_iconserv_density_d,
    Vector<BCRec> const& l_bcrec_conc,
    BCRec const* l_bcrec_conc_d,
    int const* l_iconserv_conc_d,
    std::string l_advection_type, bool l_constant_density,
    bool l_advect_conc, int l_nspec,
    bool l_godunov_ppm, bool l_godunov_use_forces_in_trans,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom, Real l_dt)
{
    #define FUNC_NAME "convection::compute_fluxes"
    DEBUG_PRINT(format("Entering {:s}.\n", FUNC_NAME));

    #if (AMREX_USE_EB)
    EBCellFlagFab const& flagfab = ebfact->getMultiEBCellFlagFab()[mfi];
    Array4<EBCellFlag const> const& flag = flagfab.const_array();
    // Original SD version of code did special handling for boxes that were not regular
    // MSE - corrected a crash by switching this test to boxes that *are* single valued
    // bool regular = (flagfab.getType(amrex::grow(bx,2)) == FabType::regular);

    auto typ = flagfab.getType(amrex::grow(bx,2));
    // bool is_liquid = (typ == FabType::regular);
    bool is_boundary = (typ == FabType::singlevalued) || (cover_multiple_cuts && typ == FabType::multivalued);
    // DEBUG_PRINT(format("Built flagfab. is_boundary={}.\n", is_boundary));

    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), ccc, vfrac, AMREX_D_DECL(apx, apy, apz);
    if (is_boundary)
    {
        // Get the centers of each face
        auto fc = ebfact->getFaceCent();
        // DEBUG_PRINT("compute_fluxes - got face centers.\n");
        AMREX_D_TERM(
        fcx = fc[0]->const_array(mfi);,
        fcy = fc[1]->const_array(mfi);,
        fcz = fc[2]->const_array(mfi);)
        // DEBUG_PRINT("compute_fluxes - converted face centers to const array.\n");
        // Get the centroid
        ccc = ebfact->getCentroid().const_array(mfi);
        // DEBUG_PRINT("compute_fluxes - got centroid.\n");
        // Get the area fraction of each face
        AMREX_D_TERM(
        apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
        apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
        apz = ebfact->getAreaFrac()[2]->const_array(mfi);)
        // DEBUG_PRINT("compute_fluxes - got area fraction.\n");
        // Get the volume fraction
        vfrac = ebfact->getVolFrac().const_array(mfi);
        // DEBUG_PRINT("compute_fluxes - got volume fraction.\n");
    }
    // DEBUG_PRINT("Got face centers, centroid, area fractions, and vfrac.\n");
    #endif
  
    int nmaxcomp = SpaceDim;
    if (l_advect_conc) 
        {nmaxcomp = amrex::max(nmaxcomp,l_nspec);}

    int n_tmp_fac;
    #if (AMREX_SPACEDIM == 3)
    n_tmp_fac = 14;
    #else
    n_tmp_fac = 10;
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
                bx, flux_comp, SpaceDim,
                AMREX_D_DECL(fx, fy, fz), 
                vel, 
                AMREX_D_DECL(umac, vmac, wmac), 
                fvel, divu, l_dt, 
                l_bcrec_velocity,
                l_bcrec_velocity_d,
                l_iconserv_velocity_d,
                tmpfab.dataPtr(), flag, 
                AMREX_D_DECL(apx, apy, apz), 
                vfrac,
                AMREX_D_DECL(fcx, fcy, fcz), 
                ccc, geom, true); 
                // last bool argument = is_velocity
                // DEBUG_PRINT("compute_fluxes() - is_boundary; delegated to compute_godunov_fluxes.\n")
        }
        else
        {
            mol::compute_convective_fluxes_eb(
                bx, flux_comp, SpaceDim,
                AMREX_D_DECL(fx, fy, fz), 
                vel, 
                AMREX_D_DECL(umac, vmac, wmac),
                l_bcrec_velocity.data(),
                l_bcrec_velocity_d,
                l_iconserv_velocity_d,
                flag, 
                AMREX_D_DECL(fcx, fcy, fcz), 
                ccc, geom);
                // DEBUG_PRINT("compute_fluxes() - is_boundary; delegated to mol::compute_convective_fluxes_eb.\n")
        }
        flux_comp += SpaceDim;

        if (!l_constant_density) 
        {
            if (l_advection_type == "Godunov")
            {
                ebgodunov::compute_godunov_fluxes(
                    bx, flux_comp, 1,
                    AMREX_D_DECL(fx, fy, fz), 
                    rho,
                    AMREX_D_DECL(umac, vmac, wmac), 
                    {}, divu, l_dt, 
                    l_bcrec_density,
                    l_bcrec_density_d,
                    l_iconserv_density_d,
                    tmpfab.dataPtr(), flag,
                    AMREX_D_DECL(apx, apy, apz), 
                    vfrac,
                    AMREX_D_DECL(fcx, fcy, fcz), 
                    ccc, geom);
            }
            else
            {
                mol::compute_convective_fluxes_eb(
                    bx, flux_comp, 1,
                    AMREX_D_DECL(fx, fy, fz), 
                    rho, 
                    AMREX_D_DECL(umac, vmac, wmac),
                    l_bcrec_density.data(),
                    l_bcrec_density_d,
                    l_iconserv_density_d,
                    flag, 
                    AMREX_D_DECL(fcx, fcy, fcz), 
                    ccc, geom);
            }
            flux_comp += 1;
        }
        // DEBUG_PRINT("compute_fluxes() - is_boundary; computed fluxes when density not constant.\n");

        if (l_advect_conc) 
        {
            if (l_advection_type == "Godunov")
            {
                ebgodunov::compute_godunov_fluxes(
                    bx, flux_comp, l_nspec,
                    AMREX_D_DECL(fx, fy, fz), 
                    rhoconc,
                    AMREX_D_DECL(umac, vmac, wmac), 
                    ftra, divu, l_dt, 
                    l_bcrec_conc,
                    l_bcrec_conc_d,
                    l_iconserv_conc_d,
                    tmpfab.dataPtr(), flag,
                    AMREX_D_DECL(apx, apy, apz), 
                    vfrac,
                    AMREX_D_DECL(fcx, fcy, fcz), 
                    ccc, 
                    geom);
            }
            else
            {
                mol::compute_convective_fluxes_eb(
                    bx, flux_comp, l_nspec,
                    AMREX_D_DECL(fx, fy, fz), 
                    rhoconc, 
                    AMREX_D_DECL(umac, vmac, wmac),
                    l_bcrec_conc.data(),
                    l_bcrec_conc_d,
                    l_iconserv_conc_d,
                    flag, 
                    AMREX_D_DECL(fcx, fcy, fcz), 
                    ccc, geom);
            }
        }
        Gpu::streamSynchronize();
        // DEBUG_PRINT("compute_fluxes() - is_boundary; advected concentrations.\n");

    } // if is_boundary
    else
    #endif
    {
        int flux_comp = 0;
        if (l_advection_type == "Godunov")
        {
            godunov::compute_godunov_fluxes(
                bx, flux_comp, SpaceDim,
                AMREX_D_DECL(fx, fy, fz), vel, 
                AMREX_D_DECL(umac, vmac, wmac), 
                fvel, divu, l_dt, 
                l_bcrec_velocity_d,
                l_iconserv_velocity_d,
                tmpfab.dataPtr(),l_godunov_ppm, 
                l_godunov_use_forces_in_trans, 
                geom, true);
            // DEBUG_PRINT("compute_fluxes() - is_boundary; delegated to compute_godunov_fluxes.\n");
        }
        else
        {
            mol::compute_convective_fluxes(
                bx, flux_comp, SpaceDim, 
                AMREX_D_DECL(fx, fy, fz), 
                vel,
                AMREX_D_DECL(umac, vmac, wmac),
                l_bcrec_velocity.data(),
                l_bcrec_velocity_d,
                l_iconserv_velocity_d, geom);
            // DEBUG_PRINT("compute_fluxes() - is_boundary; delegated to mol::compute_convective_fluxes.\n");

        }
        flux_comp += SpaceDim;

        if (!l_constant_density) 
        {
            if (l_advection_type == "Godunov")
            {
                godunov::compute_godunov_fluxes(
                    bx, flux_comp, 1,
                    AMREX_D_DECL(fx, fy, fz), 
                    rho, 
                    AMREX_D_DECL(umac, vmac, wmac),
                    {}, divu, l_dt, 
                    l_bcrec_density_d,
                    l_iconserv_density_d,
                    tmpfab.dataPtr(),l_godunov_ppm,
                    l_godunov_use_forces_in_trans,
                    geom);
            }            
            else
            {
                mol::compute_convective_fluxes(
                    bx, flux_comp, 1, 
                    AMREX_D_DECL(fx, fy, fz), 
                    rho,
                    AMREX_D_DECL(umac, vmac, wmac),
                    l_bcrec_density.data(),
                    l_bcrec_density_d, 
                    l_iconserv_density_d, geom);
            }
            flux_comp += 1;
        }
        if (l_advect_conc) 
        {
            if (l_advection_type == "Godunov")
            {
                godunov::compute_godunov_fluxes(
                    bx, flux_comp, l_nspec,
                    AMREX_D_DECL(fx, fy, fz), 
                    rhoconc, 
                    AMREX_D_DECL(umac, vmac, wmac),
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
                    rhoconc,
                    AMREX_D_DECL(umac, vmac, wmac),
                    l_bcrec_conc.data(),
                    l_bcrec_conc_d, 
                    l_iconserv_conc_d, geom);
            }
        }
        Gpu::streamSynchronize();
    }

    DEBUG_PRINT(format("Exiting {:s}.\n", FUNC_NAME));
    #undef FUNC_NAME
}
