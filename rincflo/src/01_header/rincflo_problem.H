// rincflo_problem.H
// Fragment of Rincflo.H
// Initialize flow to problems with known analytical solutions for testing purposes
#pragma once
#include <rincflo_macros.H>

private:

///////////////////////////////////////////////////////////////////////////
// problem setup - high level interface
///////////////////////////////////////////////////////////////////////////

void ProblemInitFluid (int lev);

void ProblemSetInflowVelocity (
    int grid_id, Orientation ori, Box const& bx, Array4<Real> const& v, int lev, Real time);

void RedistributeInitialData(MultiFab& data, std::string redistribution_type);

// Calculate difference between numerical solution and known analytical solution (i.e. the error)
void DiffFromExact (
    int lev, Geometry& lev_geom, Real time, Real dt, MultiFab& error, int icomp_src, int icomp_err);

///////////////////////////////////////////////////////////////////////////
// problem setup - specific problems
///////////////////////////////////////////////////////////////////////////

void ProblemInit_taylor_green (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_taylor_green3d (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_taylor_vortex (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_couette (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_channel_slant (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_rayleigh_taylor (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_tuscan (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_boussinesq_bubble (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_periodic_conc (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_double_shear_layer (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);

void ProblemInit_plane_poiseuille (
    Box const& vbx, Box const& gbx,
    Array4<Real> const& pressure,
    Array4<Real> const& vel,
    Array4<Real> const& density,
    Array4<Real> const& conc,
    Box const& domain,
    GpuArray<Real, SpaceDim> const& dx,
    GpuArray<Real, SpaceDim> const& problo,
    GpuArray<Real, SpaceDim> const& probhi);


void init_flow_from_file ();
