// rincflo_virtual.H
// Fragment of Rincflo.H
// Virtual functions in Rincflo.
#pragma once
#include <rincflo_macros.H>

private:

// ************************************************************************************************
// Virtual functions for fluid flow
virtual void ComputeSpeciesDiffCoeff (Vector<MultiFab*> const& tra_eta, int nghost);

virtual void ComputeStrainRateAtLevel (
    int lev, MultiFab* sr, MultiFab* vel, Geometry& lev_geom, Real time, int nghost);
    
virtual void ComputeViscosity (
    Vector<MultiFab*> const& eta,
    Vector<MultiFab*> const& rho,
    Vector<MultiFab*> const& vel,
    Real time, int nghost);

virtual void ComputeViscosityAtLevel (
    int lev, MultiFab* eta, MultiFab* rho, MultiFab* vel, Geometry& lev_geom, Real time, int nghost);

// ************************************************************************************************
// Virtual functions for dealing with AMReX levels

// Make a new level from scratch using provided BoxArray and DistributionMapping; only used during initialization
virtual void MakeNewLevelFromScratch (
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm) override;

// Make a new level using provided BoxArray and DistributionMapping, and fill with interpolated coarse level data
virtual void MakeNewLevelFromCoarse (
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm) override;

// Remake an existing level using provided BoxArray and DistributionMapping,,
// and fill with existing fine and coarse data
virtual void RemakeLevel (
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm) override;

// Delete level data
virtual void ClearLevel (int lev) override;

// Tag cells for refinement
virtual void ErrorEst (int lev, TagBoxArray& tags, Real time, int ngrow) override;
