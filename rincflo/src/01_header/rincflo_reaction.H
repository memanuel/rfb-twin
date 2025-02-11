// rincflo_reaction.H
// Fragment of Rincflo.H
// Methods in Rincflo for simulating fluid flow
#pragma once
#include <rincflo_macros.H>

public:

// Read the reaction parameters
void ReadReactionParameters ();

// Simulate the chemical reaction
void React();

private:

// Initialize the reaction; once per simulation
void InitReact();

// Solve one time step in the reaction
void SolveReact(
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac, 
    Vector<MultiFab*> const& v_mac, 
    Vector<MultiFab*> const& w_mac));

// Solve one time step in the reaction when simulating the membrane explicitly
void SolveReactMem(
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac));

// Has the steady state been reached to within the specified tolerance on the reaction simulation?
bool ReactionSteadyStateReached ();

// Copy concentration from new to old; also copy epotL (and optionally epotS) for BV
void StoreQuantities();

// Copy concentration from old to new; also copy epot; used for reverting failed reaction steps
void StoreQuantities_undo();

// Compute the reaction time step in models with a source term: linear, special, ButlerVolmer
void ComputeDtReaction();

// Impose constraints on the concentrations of redox active species on all levels
void ImposeRedoxConcConstraint();

// Impose constraints on the concentrations of redox active species for one level
void ImposeRedoxConcConstraint(int lev);

// Calculate the current from the source term; used in convergence checking
void CalcCurrentFromSource();

// Reinitialize concentrations with an index n0 <= n < n1
void ReinitConcImpl(int n0, int n1);
void ReinitConcAll();
void ReinitConcSome();
void InitEpot();
void UpdateInitEpotS (Real value_epotS);

// Update m_curtime according to m_dt; also delegate to UpdateTimeArrays
void UpdateTime();

// Set new time on the arrays m_t_old and m_t_new to use in fillpatching
void UpdateTimeArrays();

// Perform one explicit advection and diffusion time step; for the Nernst model
void AdvDiffStep_exp (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt);

// Apply a single step in the Nernst model
void NernstStep(
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt,
    bool explicit_diffusion);

// Compute the time step in the Nernst model; only need to run this once
void ComputeDtNernst();

// Apply the concentration implied by the Nernst equation to all boundary cells
void NernstApplyBoundaryConc(bool save_old_nernst);

// Compute the source term and from there the overpotential in the Nenst model - implementation
void CalcOverpotNernstAll(Real omega, bool update_source);

// Compute the source term and from there the overpotential, current etc. in the Nernst model - on all levels
void CalcOverpotNernst (
    int lev, Box const& bx,
    Array4<Real> const& overpot,
    Array4<Real> const& source,
    Array4<Real const> const& conc,
    Array4<Real const> const& conc_on,
    Array4<Real const> const& area_over_volume,
    Array4<EBCellFlag const> const& flag,
    Real omega,
    bool update_source);

// Perform one explicit advection + diffusion + reaction time step
void AdvDiffReactStep_exp (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt,  bool op_split1, bool op_split2);

// Perform one implicit advection step; updates conc with explicit values and prepares for linear solver later
void AdvectionStep (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt, bool store_old);

// Apply the post-step corrector to an advection step
void AdvectionStepCorrector (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt);

// Perform one explicit advection + reaction time step
void AdvReactStep_exp (
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    Real dt, bool op_split, bool update_old);

// Perform one diffusion time step using Picard iteration
bool DiffStep_pic (Real dt, int* tot_it);

// Perform one diffusion + rection time step using Picard iteration
bool DiffReactStep_pic (Real dt, int* tot_it);

// Perform one diffusion + rection time step using Newton's method iteration
bool DiffReactStep_nwt (Real dt);

// Perform one implicit diffusion step
void DiffStep (Real dt, bool set_rhs);

// Perform one explicit reaction step
void ReactStep_exp (Real dt);

// Perform one complete reaction step using Picard iteration
bool ReactStep_pic (Real dt, int* tot_it);

// Perform one complete reaction step using Netwton's method iteration
bool ReactStep_nwt (Real dt);

// Calculate the marker and cell velocities on all levels
void CalcMacVel (
    Vector<MultiFab const*> const& vel,
    Vector<MultiFab const*> const& density,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac));

// Calculate the reaction source term on all levels of the simulation
void CalcSourceTermAll();

// Calculate the reaction source term for the given box and level of the simulation
void CalcSourceTerm (
    int lev, Box const& bx,
    Array4<Real> const& pot_diff, 
    Array4<Real> const& overpot,
    Array4<Real> const& source_mol,
    Array4<Real> const& source,
    Array4<Real> const& conc,
    Array4<Real const> const& epotL,
    Array4<Real const> const& epotS,        
    Array4<Real const> const& area_over_volume,
    Array4<EBCellFlag const> const& flag);

void RelaxationFixPoint (Real omega);
bool CheckConvergenceFixPoint (int verbose);

void CalcKappa ( Vector<MultiFab*> const& kappaS, Vector<MultiFab*> const& kappaL, int nghost);
void CalcFluxRhs (Vector<MultiFab*> const& flxrhs, int nghost, Vector<MultiFab const*> const& laps);

// update concentrations for diffusion and reaction in a reaction with a source term using Picard iteration
void UpdateDiffReactConc_pic (Real dt);

// update concentrations for reaction source term using Picard iteration
void UpdateReactConc_pic (Real dt);

// update concentrations for diffusion only with explicit
void UpdateDiffConc (Real dt, bool set_rhs);

void GetDiffCoeffSpecies (Vector<MultiFab*> const& species_coeff, int nghost);

void UpdateDiffReactEpotS (Vector<MultiFab const*> const& kappaS, Vector<MultiFab const*> const& source);

void Update_DiffReactEpotL (
    Vector<MultiFab const*> const& kappaL, 
    Vector<MultiFab const*> const& flxrhs,
    Vector<MultiFab const*> const& source);

void Update_DiffReactEpotL (Vector<MultiFab const*> const& kappaL, Vector<MultiFab const*> const& source);

Array<MultiFab,SpaceDim>
AverageEpotScalToFaces (int lev, int comp, MultiFab const& cc_scal) const;

void FixupScalOnDomainFaces (int lev, Array<MultiFab,SpaceDim>& fc, MultiFab const& cc) const;

Real UpdateBCMem();

// Check whether there are any NaNs in a level
bool CheckForNans (int lev);

// Check whether there are any infinities in a level
bool CheckForInfs (int lev);

// Check whether there are any negative concentrations in a level
bool CheckForNegativeConc (int lev);
