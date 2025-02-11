//  Redistribution.H

#pragma once
#include <rincflo_macros.H>

#if (AMREX_USE_EB)

#include <AMReX_MultiFabUtil.H>
#include <AMReX_MultiCutFab.H>

namespace redistribution {

// Names used from amrex
using amrex::Real, amrex::Vector, amrex::Array4, amrex::Geometry, amrex::Box, amrex::BCRec, 
    amrex::MultiFab, amrex::EBFArrayBoxFactory, amrex::EBCellFlag, amrex::min, amrex::max;


// *********************************************************************************************************************
void redistribute_eb (
    Box const& bx, int ncomp,
    Array4<Real> const& dUdt_out,
    Array4<Real> const& dUdt_in,
    Array4<Real const> const& U_in,
    Array4<Real> const& scratch,
    Array4<EBCellFlag const> const& flag,
    AMREX_D_DECL(
    Array4<Real const> const& apx,
    Array4<Real const> const& apy,
    Array4<Real const> const& apz),
    Array4<Real const> const& vfrac,
    AMREX_D_DECL(
    Array4<Real const> const& fcx,
    Array4<Real const> const& fcy,
    Array4<Real const> const& fcz),
    Array4<Real const> const& ccent,
    Geometry& geom, Real dt, std::string redistribution_type);

// *********************************************************************************************************************
void RedistributeInitialData (
    Box const& bx, int ncomp,
    Array4<Real> const& U_out,
    Array4<Real> const& U_in,
    Array4<EBCellFlag const> const& flag,
    AMREX_D_DECL(
    Array4<Real const> const& apx,
    Array4<Real const> const& apy,
    Array4<Real const> const& apz),
    Array4<Real const> const& vfrac,
    AMREX_D_DECL(
    Array4<Real const> const& fcx,
    Array4<Real const> const& fcy,
    Array4<Real const> const& fcz),
    Array4<Real const> const& ccent,
    Geometry& geom, std::string redistribution_type);

// *********************************************************************************************************************
void flux_redistribute_eb (
    Box const& bx, int ncomp,
    Array4<Real> const& dUdt_out,
    Array4<Real const> const& dUdt_in,
    Array4<Real> const& scratch,
    Array4<EBCellFlag const> const& flag,
    Array4<Real const> const& vfrac,
    Geometry& geom);

// *********************************************************************************************************************
void state_redistribute(
    Box const& bx, int ncomp,
    Array4<Real> const& dUdt_out,
    Array4<Real> const& dUdt_in,
    Array4<EBCellFlag const> const& flag,
    Array4<Real const> const& vfrac,
    AMREX_D_DECL(
    Array4<Real const> const& fcx,
    Array4<Real const> const& fcy,
    Array4<Real const> const& fcz),
    Array4<Real const> const& ccent,
    Array4<int> const& itracker,
    Geometry& geom);

// *********************************************************************************************************************
void make_itracker (
    Box const& bx,
    AMREX_D_DECL(
    Array4<Real const> const& apx,
    Array4<Real const> const& apy,
    Array4<Real const> const& apz),
    Array4<Real const> const& vfrac,
    Array4<int> const& itracker,
    Geometry& geom,
    std::string redist_type);

// *********************************************************************************************************************
void merge_redistribute(
    Box const& bx, 
    int ncomp,
    Array4<Real> const& dUdt_out,
    Array4<Real> const& dUdt_in,
    Array4<Real const> const& vfrac,
    Array4<int> const& itracker,
    Geometry& geom);

} // namespace redistribution

#endif
