// rincflo_levels.H
// Fragment of Rincflo.H
// Structures used to define different data groupings applicable to each level.
// LevelData - quantities used in simulating the system
// DerivedData - derived quantities useful for plotting and analysis
// GeometryData - quantities that depend only on the domain and embedded boundary
#pragma once
#include <rincflo_macros.H>

// Structure to hold all the simulated data in one level of the simulation
struct LevelData 
{
    LevelData () = default;
    LevelData (
        BoxArray const& ba,
        DistributionMapping const& dm,
        FabFactory<FArrayBox> const& fact,
        int nspec, int ng_state,
        std::string advection_type, bool is_ImplicitDiffusion,
        bool use_tensor_correction, bool advect_conc,
        bool need_Laplacian, bool need_Kappa, bool need_FluxRhs,
        bool need_reaction, bool need_electromigration); 

    // The BoxArray shared by all the data arrays on this level is grids[lev]
    // The DistributionMapping shared by all the data arrays on this level is dmap[lev]

    // Geometry information
    MultiFab cell_type;         // type of cell; same values as enum class CellType
    MultiFab area;              // active surface area in a grid cell (m^2)
    MultiFab volume;            // volume (liquid) in a grid cell (m^3)
    MultiFab area_over_volume;  // area over volume in a grid cell (m^-1); useful in source calculation
    // MultiFab cell_centroid;     // centroid of cut cell in dimensionless coordinates relative to the center
    // MultiFab bdry_centroid;     // centroid of boundary cell in dimensionless coordinates relative to the center
    // MultiFab bdry_normal;       // normal vector of boundary face

    // Flow variables (cell-centered)
    MultiFab velocity;          // fluid velocity
    MultiFab velocity_o;        // fluid velocity (old)
    MultiFab density;           // fluid density in kg / m^3 = gram / cc
    MultiFab density_o;         // fluid density (old)
    MultiFab pressure;          // fluid pressure
    MultiFab gradp;             // pressure gradient
    MultiFab mac_phi;           // cell-centered pressure used in MAC projection

    // Flow variables (nodal multifabs)

    // Concentration rate of change due to convection
    MultiFab conv_dconc_dt;
    // Concentration rate of change due to convection - old
    MultiFab conv_dconc_dt_o;

    // Concentration after it is updated for advection only
    MultiFab conc_adv;
    // Concentration after it is updated for advection only - old
    MultiFab conc_adv_o;

    // Velocity rate of change due to convection
    MultiFab conv_dv_dt;     
    // Velocity rate of change due to convection - old
    MultiFab conv_dv_dt_o;
    
    // Density rate of change due to convection; in kg / m^3 = gram / cc; not used with constant density
    MultiFab conv_drho_dt;      
    // Density rate of change due to convection; in kg / m^3 = gram / cc - old; not used with constant density
    MultiFab conv_drho_dt_o;

    // Fancy flow variables
    MultiFab divtau;
    MultiFab divtau_o;
    MultiFab laps;
    MultiFab laps_o;

    // Species concentrations (cell-centered)
    MultiFab conc;              // Concentration of each chemical species (millimolar)
    MultiFab conc_o;            // Concentration of each chemical species (old)
    MultiFab conc_on;           // Concentration of each conc species (old, start of Nernst batch)

    // Reaction calculations (cell-centered)        
    MultiFab pot_diff;          // The interfacial potential difference (epotL - epotS) in volts
    MultiFab overpot;           // The overpotential (deltaPhi - deltaPhi_eq) in volts in a reducing direction
    MultiFab source;            // source term (reaction rate in millimolarity/sec)
    MultiFab epotL;             // electric potential in liquid in volts
    MultiFab epotL_o;           // electric potential in liquid (old)
    MultiFab epotS;             // electric potential in solid in volts
    MultiFab epotS_o;           // electric potential in solid (old) 

    // IMPORTANT: the sign in the definition of pot_diff is reversed from the usual convention.
    // This makes sense in the context of this model because we are simulating a reduction of AQDS to H2-AQDS.
    // A higher pot_diff leads to a higher overpot and a higher source term.

    // Reaction quantities in picard iteration for nonlinearity (cell-centered)
    MultiFab conc_op;
    MultiFab epotL_op; 
    MultiFab epotS_op;

    // ButlerVolmer reaction model
    MultiFab kappa;             // conductivity in the liquid
    MultiFab flxrhs;            // flux (?)
    MultiFab divcgpot;          // divergence of electric potential - only used for electromigration

    // Derived quantities for analysis and plotting; only updated intermittently before building plots
    MultiFab soc;               // state of charge (dimensionless)
    MultiFab soc_o;             // state of charge (old)
    MultiFab source_mol;        // source term (reaction rate in millimoles / sec)
    MultiFab current;           // current (amps)
    MultiFab curr_dens;         // current density (amps / m^2)
};

// ************************************************************************************************
// Structure with all single grid MultiFabs on one level used for reading numpy data
struct LevelDataNumpyInput
{
    // Default constructor
    LevelDataNumpyInput () = default;

    // Constructor
    LevelDataNumpyInput(
        BoxArray const& ba, DistributionMapping const& dm, 
        FabFactory<FArrayBox> const& fact, int dim, int nt, int ng);

    // Members

    // Geometry information
    MultiFab cell_type;

    // Flow variables
    MultiFab velocity;
    MultiFab pressure;
    MultiFab gradp;
    
    // Reaction state variables
    MultiFab conc;
    MultiFab epotL;
    MultiFab epotS;

    // Reaction cut cells
    MultiFab current;
    MultiFab overpot;
};

// ************************************************************************************************
// Structure with all single grid MultiFabs on one level used for writing numpy data
struct LevelDataNumpyOutput
{
    // Default constructor
    LevelDataNumpyOutput () = default;

    // Constructor
    LevelDataNumpyOutput(
        BoxArray const& ba, DistributionMapping const& dm, 
        FabFactory<FArrayBox> const& fact, int dim, int nt, int ng);

    // Members

    // Geometry information
    MultiFab cell_type;
    MultiFab area;
    MultiFab volume;
    // MultiFab cell_centroid;
    // MultiFab bdry_centroid;
    // MultiFab bdry_normal;

    // Flow variables
    MultiFab velocity;
    MultiFab pressure;
    MultiFab gradp;
    
    // Reaction state variables
    MultiFab conc;
    MultiFab soc;
    MultiFab epotL;
    MultiFab epotS;

    // Reaction cut cells    
    MultiFab current;
    MultiFab overpot;
};
