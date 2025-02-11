#pragma once
#include <rincflo_macros.H>

// Include headers to all the libraries and classes used in Rincflo
#include <rincflo_includes.H>

// ************************************************************************************************
class Rincflo : public amrex::AmrCore
{

    // Class inheritance
    public:

    // Concentration is advected together with fluid density in the flow part and without in the reaction part
    friend DiffusionTensorOp;
    friend DiffusionScalarOp;
    friend ReactionScalarOp;
    friend ReactionEliquidOp;
    friend ReactionEsolidOp;

    // Default constructor and virtual destructor
    Rincflo ();
    virtual ~Rincflo ();

    // Definitions of levels - we want these in namespace of Rincflo
    #include <rincflo_levels.H>

    // All the enums used by Rincflo
    #include <rincflo_enums.H>

    // Member variables
    #include <rincflo_members.H>

    // Initialize the simulation - construct MultiFabs, read parameters, etc.
    #include <rincflo_initialize.H>

    // Functions for the fluid flow simulation
    #include <rincflo_flow.H>

    // Functions for the reaction simulation
    #include <rincflo_reaction.H>

    // Functions for AMReX interface
    #include <rincflo_amrex.H>

    // Virtual functions - mainly for AMReX interface
    #include <rincflo_virtual.H>

    // Get vectors of multifab, constant or non-constant
    #include <rincflo_get.H>

    // Copy multifab from new to old or old to new
    #include <rincflo_copy.H>

    // Get status indicators
    #include <rincflo_status.H>

    // Functions related to numpy data files
    #include <rincflo_numpy.H>

    // Analyze the results of a simulation and report them to console    
    #include <rincflo_analysis.H>

    // Handle checkpoints, plotfiles, debug printing, etc.
    #include <rincflo_utilities.H>

    // Benchmark problems with known analytical solutions
    #include <rincflo_problem.H>
};
