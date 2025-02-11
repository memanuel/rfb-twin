//  Convection_tracer_only.H

#pragma once
#include <rincflo_macros.H>

#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>

#if (AMREX_USE_EB)
#include <AMReX_MultiCutFab.H>
#endif

namespace convection 
{

// Names used from amrex
using amrex::Real, amrex::Vector, amrex::Array4, amrex::Box, amrex::Geometry, 
amrex::MFIter, amrex::EBFArrayBoxFactory, amrex::BCRec;

// *********************************************************************************************************************
// Implementation to compute the simple convective term for concentration in a box given velocity, fluxes, etc.
void compute_convective_term_conc_impl (
    Box const& bx, 
    MFIter const& mfi,
    // conc
    Array4<Real> const& dconc_dt, 
    AMREX_D_DECL(
    Array4<Real const> const& u_mac,
    Array4<Real const> const& v_mac,
    Array4<Real const> const& w_mac),
    AMREX_D_DECL(
    Array4<Real const> const& flux_x,
    Array4<Real const> const& flux_y,
    Array4<Real const> const& flux_z),
    int const* d_conc_iconserv,
    std::string l_advection_type,
    int l_nspec,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom);
    
  void compute_fluxes_conc (
    Box const& bx, MFIter const& mfi,
    Array4<Real const> const& conc,
    Array4<Real const> const& divu,
    AMREX_D_DECL(
    Array4<Real const> const& u_mac,
    Array4<Real const> const& v_mac,
    Array4<Real const> const& w_mac),
    AMREX_D_DECL(
    Array4<Real> const& flux_x,
    Array4<Real> const& flux_y,
    Array4<Real> const& flux_z),
    Array4<Real const> const& ft,
    Vector<BCRec> const& h_conc_bcrec,
    BCRec const* d_conc_bcrec,
    int const* d_conc_iconserv,
    std::string l_advection_type,
    int l_nspec,
    bool l_godunov_ppm, 
    bool l_godunov_use_forces_in_trans,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    #endif
    Geometry& geom,
    Real l_dt);

// *********************************************************************************************************************
#if (AMREX_USE_EB)
void redistribute_convective_term_conc (
    Box const& bx, MFIter const& mfi,
    // conc
    Array4<Real const> const& conc,     
    // initial update to dconc_dt due to convection
    Array4<Real> const& dconc_dt_tmp,       
    // final update to dcond_dt due to convection
    Array4<Real> const& dconc_dt,           
    std::string l_redistribution_type, 
    int l_nspec,
    EBFArrayBoxFactory const* ebfact,
    bool cover_multiple_cuts,
    Geometry& geom,
    Real l_dt);
#endif
} // namespace convection
