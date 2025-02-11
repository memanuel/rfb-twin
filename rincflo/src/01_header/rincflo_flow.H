// rincflo_flow.H
// Fragment of Rincflo.H
// Methods in Rincflo for simulating fluid flow
#pragma once
#include <rincflo_macros.H>

public:

// Evolve solution to final time through repeated calls to Advance()
void Evolve ();

private:

// Advance the fluid flow simulation one time step
void Advance ();

// Has the steady state been reached to within the specified tolerance on the flow simulation?
bool SteadyStateReached ();
    
// Project the fluid flow to be divergence free
void ApplyProjection(Vector<MultiFab const*> density, Real time, Real scaling_factor, bool incremental);

// Compute the time step for fluid flow using the CFL condition
void ComputeDt (int initialisation, bool explicit_diffusion);

// Print a summary of the fluid model parameters to the console
void PrintFluidModelSummary();

// Should we write out a plotfile now? Called during main flow simulation loop.
bool WriteNow ();

void InitialRedistribution ();
void InitialProjection ();
void InitialIterations ();

void SetBackgroundPressure ();

void ComputeTraForces (
    Vector<MultiFab*> const& tra_forces,
    Vector<MultiFab const*> const& density);

void ComputeVelForces (
    Vector<MultiFab*> const& vel_forces,
    Vector<MultiFab const*> const& velocity,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& conc_old,
    Vector<MultiFab const*> const& conc_new,
    bool include_pressure_gradient = true);

void ComputeVelForcesOnLevel (
    int lev,
    MultiFab& vel_forces,
    const MultiFab& velocity,
    const MultiFab& density,
    const MultiFab& conc_old,
    const MultiFab& conc_new,
    bool include_pressure_gradient = true);

// boundary_conditions
void SetInflowVelocity (int lev, Real time, MultiFab& vel, int nghost);

//**********************************************************************************************************************
// convection
//**********************************************************************************************************************

void ApplyPredictor(bool incremental_projection = false);
void ApplyCorrector();

// The most general form of the convection term including variable density and forces due to velocity and tracers.
// The convection term is saved into the conv_* MultiFabs.
void ComputeConvectiveTermFull (
    Vector<MultiFab*> const& conv_u,
    Vector<MultiFab*> const& conv_r,
    Vector<MultiFab*> const& conv_t,
    Vector<MultiFab const*> const& vel,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& conc,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Vector<MultiFab*> const& vel_forces,
    Vector<MultiFab*> const& tra_forces,
    Real time);

// Compute the convective term on all levels; simplified form that only requires the concentration and velocity.
void ComputeConvectiveTermConc (
    // Time derivative of the concentration due to convection; written out in this function
    Vector<MultiFab*> const& dconc_dt,
    // The concentration
    Vector<MultiFab const*> const& conc,
    // The MAC velocity components
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac));

void Compute_MacProjectedVelocities (
    Vector<MultiFab const*> const& vel,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    AMREX_D_DECL(
    Vector<MultiFab*> const& inv_rho_x,
    Vector<MultiFab*> const& inv_rho_y,
    Vector<MultiFab*> const& inv_rho_z),
    Vector<MultiFab*> const& vel_forces,
    Real time);

#if (AMREX_USE_EB)
void CorrectSmallCells (
    Vector<MultiFab*> const& cc_vel,
    AMREX_D_DECL(
    Vector<MultiFab const*> const& u_mac,
    Vector<MultiFab const*> const& v_mac,
    Vector<MultiFab const*> const& w_mac));
#endif

//**********************************************************************************************************************
// diffusion
//**********************************************************************************************************************

Array<MultiFab,SpaceDim>
average_velocity_eta_to_faces (int lev, MultiFab const& cc_eta) const;

void ComputeDivTau (
    Vector<MultiFab*> const& divtau,
    Vector<MultiFab const*> const& velocity,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& eta);

void DiffuseVelocity (
    Vector<MultiFab*> const& velocity,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt_diff); 

Array<MultiFab,SpaceDim>
AverageScalarEtaToFaces (int lev, int comp, MultiFab const& cc_eta) const;

void ComputeLaplacian (
    Vector<MultiFab *> const& laps,
    Vector<MultiFab const*> const& scalar,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& eta);

void DiffuseScalar (
    Vector<MultiFab*> const& scalar,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt_diff); 

void FixupEtaOnDomainFaces (
    int lev, Array<MultiFab,SpaceDim>& fc, MultiFab const& cc) const;

//**********************************************************************************************************************
// rheology
//**********************************************************************************************************************

void ReadRheologyParameters ();
