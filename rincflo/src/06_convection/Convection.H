//  Convection.H

#pragma once
#include <rincflo_macros.H>

#include <string>
#include <format>

#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#if (AMREX_USE_EB)
#include <AMReX_MultiCutFab.H>
#endif

using std::string;
using std::format;
using amrex::Print;
using amrex::ParallelDescriptor::MyProc;

namespace convection 
{

// *********************************************************************************************************************
// Implementation to compute the fully general convective term for concentration in a box given velocity, fluxes, etc.
void compute_convective_term_full_impl (
    amrex::Box const& bx, 
    amrex::MFIter const& mfi,
    // velocity
    amrex::Array4<amrex::Real> const& dv_dt, 
    // density
    amrex::Array4<amrex::Real> const& drho_dt, 
    // conc
    amrex::Array4<amrex::Real> const& dconc_dt, 
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& u_mac,
    amrex::Array4<amrex::Real const> const& v_mac,
    amrex::Array4<amrex::Real const> const& w_mac),
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& flux_x,
    amrex::Array4<amrex::Real const> const& flux_y,
    amrex::Array4<amrex::Real const> const& flux_z),
    int const* d_vel_iconserv,
    int const* d_den_iconserv,
    int const* d_conc_iconserv,
    bool l_constant_density, 
    bool l_advect_conc, int l_nspec,
    #if (AMREX_USE_EB)
    amrex::EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    amrex::Geometry& geom);

// *********************************************************************************************************************
void compute_fluxes (
    amrex::Box const& bx, amrex::MFIter const& mfi,
    amrex::Array4<amrex::Real const> const& vel,
    amrex::Array4<amrex::Real const> const& rho,
    amrex::Array4<amrex::Real const> const& conc,
    amrex::Array4<amrex::Real const> const& divu,
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& u_mac,
    amrex::Array4<amrex::Real const> const& v_mac,
    amrex::Array4<amrex::Real const> const& w_mac),
    AMREX_D_DECL(
    amrex::Array4<amrex::Real> const& flux_x,
    amrex::Array4<amrex::Real> const& flux_y,
    amrex::Array4<amrex::Real> const& flux_z),
    amrex::Array4<amrex::Real const> const& fv,
    amrex::Array4<amrex::Real const> const& ft,
    amrex::Vector<amrex::BCRec> const& h_vel_bcrec,
    amrex::BCRec  const* d_vel_bcrec,
    int const* d_vel_iconserv,
    amrex::Vector<amrex::BCRec> const& h_den_bcrec,
    amrex::BCRec const* d_den_bcrec,
    int const* d_den_iconserv,
    amrex::Vector<amrex::BCRec> const& h_conc_bcrec,
    amrex::BCRec const* d_conc_bcrec,
    int const* d_conc_iconserv,
    std::string l_advection_type, bool l_constant_density, 
    bool l_advect_conc, int l_nspec,
    bool l_godunov_ppm, bool l_godunov_use_forces_in_trans,
    #if (AMREX_USE_EB)
    amrex::EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    amrex::Geometry& geom,
    amrex::Real l_dt);

void compute_convective_update (
    amrex::Box const& bx, int flux_comp, int ncomp,
    amrex::Array4<amrex::Real> const& dUdt,
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& fx,
    amrex::Array4<amrex::Real const> const& fy,
    amrex::Array4<amrex::Real const> const& fz),
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& umac,
    amrex::Array4<amrex::Real const> const& vmac,
    amrex::Array4<amrex::Real const> const& wmac),
    amrex::Geometry& geom, int const* iconserv);

#if (AMREX_USE_EB)

// *********************************************************************************************************************
void compute_convective_update_eb (
    amrex::Box const& bx, int flux_comp, int ncomp,
    amrex::Array4<amrex::Real> const& dUdt,
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& fx,
    amrex::Array4<amrex::Real const> const& fy,
    amrex::Array4<amrex::Real const> const& fz),
    amrex::Array4<amrex::EBCellFlag const> const& flag,
    amrex::Array4<amrex::Real const> const& vfrac,
    AMREX_D_DECL(
    amrex::Array4<amrex::Real const> const& apx,
    amrex::Array4<amrex::Real const> const& apy,
    amrex::Array4<amrex::Real const> const& apz),
    amrex::Geometry& geom, int const* iconserv);

// *********************************************************************************************************************
void redistribute_convective_term (
    amrex::Box const& bx, amrex::MFIter const& mfi,
    amrex::Array4<amrex::Real const> const& vel,        // velocity
    amrex::Array4<amrex::Real const> const& rho,        // density
    amrex::Array4<amrex::Real const> const& rhotrac,    // conc
    amrex::Array4<amrex::Real> const& dvdt_tmp,         // initial velocity update
    amrex::Array4<amrex::Real> const& drdt_tmp,         // initial density update
    amrex::Array4<amrex::Real> const& dtdt_tmp,         // initial conc update
    amrex::Array4<amrex::Real> const& dvdt,             // final velocity update
    amrex::Array4<amrex::Real> const& drdt,             // final density update
    amrex::Array4<amrex::Real> const& dtdt,             // final conc update
    std::string l_redistribution_type, 
    bool l_constant_density, bool l_advect_conc, int l_nspec,
    amrex::EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    amrex::Geometry& geom, amrex::Real l_dt);
#endif

// *********************************************************************************************************************
#if (DEBUG_RFB)
// Print a debug message to a debug log file corresponding to its MPI rank, e.g. debug_01.txt
void DebugPrint(const string &msg);

// Number of MPI processors with debug logging
static constexpr int debug_log_size {0};
#endif

#if (DEBUG_RFB)
#define DEBUG_PRINT(msg)                DebugPrint(msg);
#else
#define DEBUG_PRINT(msg)
#endif

} // namespace convection
