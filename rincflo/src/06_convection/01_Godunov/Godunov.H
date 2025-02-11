//  Godunov.H

#pragma once
#include <rincflo_macros.H>

#include <AMReX_MultiFabUtil.H>
#include <AMReX_BCRec.H>

namespace godunov {

// Names used from amrex
using amrex::Real, amrex::Vector, amrex::Array4, amrex::Geometry, amrex::Box, amrex::BCRec, amrex::MultiFab;

constexpr Real small_vel = 1.e-8;

// *********************************************************************************************************************
void predict_godunov (
    Real time, 
    AMREX_D_DECL(
    MultiFab& u_mac,
    MultiFab& v_mac,
    MultiFab& w_mac),
    MultiFab const& vel, 
    MultiFab const& vel_forces,
    Vector<BCRec> const& h_bcrec,
    BCRec  const* d_bcrec,
    Geometry& geom,
    Real dt, 
    bool use_ppm, 
    bool use_forces_in_trans,
    AMREX_D_DECL(
    MultiFab const& gmacphi_x,
    MultiFab const& gmacphi_y,
    MultiFab const& gmacphi_z),
    bool use_mac_phi_in_godunov);

// *********************************************************************************************************************
void predict_plm_x (
    Box const& bx,
    Array4<Real> const& Imx, 
    Array4<Real> const& Ipx,
    Array4<Real const> const& q,
    Array4<Real const> const& vcc,
    Geometry& geom,
    Real dt,
    Vector<BCRec> const& h_bcrec,
    BCRec const* d_bcrec);

// *********************************************************************************************************************
void predict_plm_y (
    Box const& bx,
    Array4<Real> const& Imy, 
    Array4<Real> const& Ipy,
    Array4<Real const> const& q,
    Array4<Real const> const& vcc,
    Geometry& geom,
    Real dt,
    Vector<BCRec> const& h_bcrec,
    BCRec const* d_bcrec);

// *********************************************************************************************************************
#if (AMREX_SPACEDIM == 3)
void predict_plm_z (
    Box const& bx,
    Array4<Real> const& Imz, 
    Array4<Real> const& Ipz,
    Array4<Real const> const& q,
    Array4<Real const> const& vcc,
    Geometry& geom,
    Real dt,
    Vector<BCRec> const& h_bcrec,
    BCRec const* d_bcrec);
#endif

// *********************************************************************************************************************
void predict_ppm (
    Box const& bx,
    AMREX_D_DECL(
    Array4<Real> const& Imx,
    Array4<Real> const& Imy,
    Array4<Real> const& Imz),
    AMREX_D_DECL(
    Array4<Real> const& Ipx,
    Array4<Real> const& Ipy,
    Array4<Real> const& Ipz),
    Array4<Real const> const& q,
    Array4<Real const> const& vel,
    Geometry& geom,
    Real dt,
    BCRec const* d_bcrec);

// *********************************************************************************************************************
void make_trans_velocities (
    AMREX_D_DECL(
    Box const& xbx,
    Box const& ybx, 
    Box const& zbx),
    AMREX_D_DECL(
    Array4<Real> const& u_ad,
    Array4<Real> const& v_ad,
    Array4<Real> const& w_ad),
    AMREX_D_DECL(
    Array4<Real const> const& Imx,
    Array4<Real const> const& Imy,
    Array4<Real const> const& Imz),
    AMREX_D_DECL(
    Array4<Real const> const& Ipx,
    Array4<Real const> const& Ipy,
    Array4<Real const> const& Ipz),
    Array4<Real const> const& vel,
    Array4<Real const> const& f,
    const Box& domain,
    Real dt,
    BCRec const* d_bcrec,
    bool use_forces_in_trans);

// *********************************************************************************************************************
void predict_godunov_on_box (
    Box const& bx, int ncomp,
    AMREX_D_DECL(
    Box const& xbx, 
    Box const& ybx, 
    Box const& zbx),
    AMREX_D_DECL(
    Array4<Real> const& qx,
    Array4<Real> const& qy,
    Array4<Real> const& qz),
    Array4<Real const> const& q,
    AMREX_D_DECL(
    Array4<Real const> const& u_ad,
    Array4<Real const> const& v_ad,
    Array4<Real const> const& w_ad),
    AMREX_D_DECL(
    Array4<Real> const& Imx,
    Array4<Real> const& Imy,
    Array4<Real> const& Imz),
    AMREX_D_DECL(
    Array4<Real> const& Ipx,
    Array4<Real> const& Ipy,
    Array4<Real> const& Ipz),
    Array4<Real const> const& f,
    const Box& domain,
    const Real* dx,
    Real dt,
    BCRec const* d_bcrec,
    bool use_forces_in_trans,
    AMREX_D_DECL(
    Array4<Real const> const& gmacphi_x,
    Array4<Real const> const& gmacphi_y,
    Array4<Real const> const& gmacphi_z),
    bool use_mac_phi_in_godunov,
    Real* p);

// *********************************************************************************************************************
void compute_godunov_fluxes (
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
    Array4<Real const> const& fq,
    Array4<Real const> const& divu,
    Real dt,
    BCRec const* d_bcrec, 
    int const* iconserv,
    Real* p, 
    bool use_ppm, 
    bool use_forces_in_trans,
    Geometry& geom,
    bool is_velocity = false);

} // namespace godunov
