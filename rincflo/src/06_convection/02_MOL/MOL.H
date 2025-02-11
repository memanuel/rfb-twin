//  MOL.H

#pragma once
#include <rincflo_macros.H>

#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>

#if (AMREX_USE_EB)
#include <AMReX_MultiCutFab.H>
#endif


namespace mol {

// Names used from amrex
using amrex::Real, amrex::Vector, amrex::Array4, amrex::Geometry, amrex::Box, amrex::BCRec, 
    amrex::MultiFab, amrex::EBFArrayBoxFactory, amrex::EBCellFlag, amrex::min, amrex::max;

// *********************************************************************************************************************
void predict_vels_on_faces (
    int lev,  
    AMREX_D_DECL(
    MultiFab& u_mac,
    MultiFab& v_mac, 
    MultiFab& w_mac),
    MultiFab const& vel,
    Vector<BCRec> const& h_bcrec,
    BCRec  const* d_bcrec,
    #if (AMREX_USE_EB)
    EBFArrayBoxFactory const* ebfact,
    #endif
    Vector<Geometry> geom);

// *********************************************************************************************************************
void predict_vels_on_faces (
    int lev,
    AMREX_D_DECL(
    Box const& ubx,
    Box const& vbx,
    Box const& wbx),
    AMREX_D_DECL(
    Array4<Real> const& u,
    Array4<Real> const& v,
    Array4<Real> const& w),
    Array4<Real const> const& vcc,
    Vector<BCRec> const& h_bcrec,
    BCRec  const* d_bcrec,
    Vector<Geometry> geom);

// *********************************************************************************************************************
void compute_convective_fluxes (
    Box const& bx, int flux_comp, int ncomp,
    AMREX_D_DECL(
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz),
    Array4<Real const> const& q,
    AMREX_D_DECL(
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac),
    BCRec const* h_bcrec,
    BCRec const* d_bcrec,
    int const* iconserv,
    Geometry& geom);

// *********************************************************************************************************************
#if (AMREX_USE_EB)
void predict_vels_on_faces_eb (
    int lev,
    AMREX_D_DECL(
    Box const& ubx,
    Box const& vbx,
    Box const& wbx),
    AMREX_D_DECL(
    Array4<Real> const& u,
    Array4<Real> const& v,
    Array4<Real> const& w),
    Array4<Real const> const& vcc,
    Array4<EBCellFlag const> const& flag,
    AMREX_D_DECL(
    Array4<Real const> const& fcx,
    Array4<Real const> const& fcy,
    Array4<Real const> const& fcz), 
    Array4<Real const> const& ccc,
    Vector<BCRec> const& h_bcrec,
    BCRec  const* d_bcrec,
    Vector<Geometry> geom);

// *********************************************************************************************************************
void compute_convective_fluxes_eb (
    Box const& bx, int flux_comp, int ncomp,
    AMREX_D_DECL(
    Array4<Real> const& fx,
    Array4<Real> const& fy,
    Array4<Real> const& fz),
    Array4<Real const> const& q,
    AMREX_D_DECL(
    Array4<Real const> const& umac,
    Array4<Real const> const& vmac,
    Array4<Real const> const& wmac),
    BCRec const* h_bcrec,
    BCRec const* d_bcrec,
    int const* iconserv,
    Array4<EBCellFlag const> const& flag,
    AMREX_D_DECL(
    Array4<Real const> const& fcx,
    Array4<Real const> const& fcy,
    Array4<Real const> const& fcz),
    Array4<Real const> const& ccc,
    Geometry& geom);
    #endif
} // namespace mol
