// rincflo_analysis.H
// Fragment of Rincflo.H
// Methods in Rincflo to analyze the simulation results.
#pragma once
#include <rincflo_macros.H>

public:

// Perform Data Analysis - high level

// Analyze the whole simulation and print results to console
void Analysis();

// Analyze the geometry and print results to console
void AnalyzeGeometry() const;

// Analyze the flow simlulation and print results to console; Not const because it modifies m_t_flow
void AnalyzeFlow();                         

// Analyze the current and print results to console
void AnalyzeCurrent() const;

// Analyze electric potential and print results to console
void AnalyzePotential() const;

// Compute the following geometric quantities: cell_type, area, volume
void CalcGeometry();

// The area of one full cell on a level in square meters
Real CalcCellArea (int lev) const;

// The volume of one full cell on a level in cubic meters
Real CalcCellVolume (int lev) const;

// The average volume fraction of liquid a.k.a. porosity at this level
Real CalcAverageVolumeFraction(int lev) const;

// The total reactive surface area at this level
Real CalcTotalSurfaceArea(int lev) const;

// The total reactive surface area for the whole domain
Real CalcTotalSurfaceArea() const;

// The total number of grid cells at this level
long CalcGridSize(int lev) const;

// The total number of grid cells at this level, cast to a Real for floating point
inline Real CalcGridSizeReal(int lev) const
    {return static_cast<Real> (CalcGridSize(lev));}

// Compute derived fields related to the fluid flow
void CalcVorticity (int lev, Real time, MultiFab& vort, MultiFab const& vel);
void CalcDivU (Real time);
// void CalcDrag ();
Real CalcKineticEnergy () const;

// Compute the magnitude of the velocity in a new MultiFab
void CalcVelMag (int lev, MultiFab& vmag, MultiFab const& vel);

// Compute an adjusted pressure that incorporates the background pressure gradient from inlet / outlet
void CalcAdjustedPressure (int lev, MultiFab& pressure_adj, MultiFab const& pressure, bool write_nan) const;

// Compute an adjusted pressure gradient that incorporates the background pressure gradient from inlet / outlet
void CalcAdjustedGradp (int lev, MultiFab& gradp_adj, MultiFab const& gradp) const;

// Compute the following derived reaction quantities: soc, pot_diff, overpot, current, curr_dens
void CalcDerivedReact();

// Compute the total current - by summing the current MultiFab at the finest level
Real CalcTotalCurrent() const;

// Compute average current density - by summing the current MultiFab at the finest level
Real CalcAvgCurrentDensity();

// Calculate the total current - by summing up the the BV current at the finest level.
// Use the values of the soc and overpot MultiFabs as inputs to the Butler-Volmer equation.
Real CalcTotalCurrent_BV() const;

// Calculate net flux of reduced species and convert to an equivalent current;
// at steady state the net flux will equal the current.
Real CalcNetFluxCurrent() const;

// Calculate the total concentration of redox species, which should be conserved, and print the results
Real CalcConservedRedoxConc(bool verbose = false) const;

private:

// Calculate the volume weighted sum of a MultiFab
Real VolumeWeightedSum (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Calculate the area weighted sum of a MultiFab
Real AreaWeightedSum (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Calculate the volume weighted average of a MultiFab
Real VolumeWeightedAvg (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Calculate the area weighted average of a MultiFab
Real AreaWeightedAvg (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Sum a MultiFab over boundary cells
Real SumOverBoundary (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Min of a MultiFab over boundary cells
Real MinOverBoundary (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Max of a MultiFab over boundary cells
Real MaxOverBoundary (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Sum a MultiFab over liquid cells
Real SumOverLiquid (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Min of a MultiFab over liquid cells
Real MinOverLiquid (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Max of a MultiFab over liquid cells
Real MaxOverLiquid (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Sum of Squares of a MultiFab over liquid cells
Real SumSquareOverLiquid (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Count of liquid cells in a level
Real CountLiquidCells (int lev, bool local=false) const;

// Sum a MultiFab over active cells
Real SumOverActive (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Min of a MultiFab over active cells in the simulation (liquid and boundary_
Real MinOverActive (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Max of a MultiFab over active cells in the simulation (liquid and boundary_
Real MaxOverActive (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Sum of Squares of a MultiFab over active cells
Real SumSquareOverActive (int lev, const MultiFab& mf, int comp, bool local=false) const;

// Count of active cells in a level
Real CountActiveCells (int lev, bool local=false) const;

// Sum a MultiFab over the specified slice (direction and index value)
Real SumOverSlice(int dir, int idx, const MultiFab& mf, int comp, bool local) const;

// Sum a MultiFab over the physical boundary cells (e.g. x_lo = inlet, x_hi = outlet)
Real SumOverBoundaryCells (int dir, bool lo, const MultiFab& mf, int comp, bool is_nodal, bool local) const;
