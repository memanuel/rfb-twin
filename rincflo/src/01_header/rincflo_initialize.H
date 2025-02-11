// rincflo_intialize.H
// Fragment of Rincflo.H
// Methods in Rincflo to initialize the simulation.
#pragma once
#include <rincflo_macros.H>

public:

// Initialize multilevel AMR data
void InitData ();

private:

// Read parameters for the flow simulation
void ReadParameters ();

// Read paraemters for I/O, e.g. which fields to include in plotfile
void ReadIOParameters ();

// Resize arrays to fit (up to) max_level + 1 AMR levels
void ResizeArrays (); 

// Scale the velocity and pressure up after reading it in by m_vel_scale_factor
void ScaleInputVelocity();

// Initialize boundary condititions
void InitBoundaryConditions ();

// Initialize advection
void InitAdvection ();

// Create the geometry objects to support embedded boundaries
#if (AMREX_USE_EB)
// Build the entire EB hierarchy
void MakeEBGeometry ();
// Build a "regular" EB with no embedded boundaries
void MakeEB_regular ();
// Build EB of type lattice - collection of cylinders
void MakeEB_lattice ();
// Build EB of type box
void MakeEB_box ();
// Build EB of type cylinder
void MakeEB_cylinder ();
// Build EB of type annulus
void MakeEB_annulus ();
// Build EB of type sphere
void MakeEB_sphere ();
// Build EB of type spherecube
void MakeEB_spherecube ();
// Build EB of type tuscan
void MakeEB_tuscan ();
// Build EB of type randomfibers
void MakeEB_random_fibers ();

void CheckCutCells();
#endif
