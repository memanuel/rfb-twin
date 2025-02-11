// rincflo_enums.H
// Fragment of Rincflo.H
// all the enumerations used by Rincflo
#pragma once
#include <rincflo_macros.H>

public:

// Support fluid models
enum class FluidModel {
    Newtonian, powerlaw, Bingham, HerschelBulkley, deSouzaMendesDutra
};

// Supported methods for calculating diffusion
enum class DiffusionType {
    Invalid, Explicit, Crank_Nicolson, Implicit
};

// Supported reaction models
enum class ReactionModel 
{
    // No reaction
    none = 0,
    // No reaction kinetics; concentration on boundary cells determined by Nernst equation
    Nernst = 2,
    // Ignore potential in liquid; interfacial potential = applied voltage drop with BV kinetics
    SpecialRedox = 3,
    // BV reaction kinetics; solve for potential in the liquid, imposed potential at boundary
    ButlerVolmer = 4,
};

// Type of checkpoints
enum class CheckPointType 
{
    FluidFlow = 0,
    Reaction = 1,
    Error = -1,
};

// Types of convergence criteria for solvers
enum class ToleranceType
{
    // The criterion is the root mean square of the absolute change in the quantity over one time step
    // Convergence when RMS(Q_new - Q_old) < tol
    abs_step = 0,

    // The criterion is the root mean square of the absolute change in the quantity per time to flow across the domain
    // Convergence when RMS((Q_new - Q_old) * (tau / dt)) < tol
    abs_rate = 1,

    // The criterion is the root mean square of the relative change in the quantity over one time step
    // Convergence when RMS((Q_new - Q_old) / Q_old)) < tol
    rel_step = 2,

    // Convergence when RMS((Q_new - Q_old) / Q_old) * (tau / dt)) < tol
    // The criterion is the root mean square of the relative change in the quantity per time to flow across the domain
    rel_rate = 3,
};

// option to (iteratively) update b.c. at membrane to compensate for current flowing out of cell
enum class MembraneBCType 
{
    none, potential, current
};

//$ should be read-in and check for consistency with other b.c.s... (also make it an Orientation variable...)
enum class MembraneLocation 
{
    xlo, xhi, ylo, yhi, zlo, zhi
};

// Different kinds of boundary conditions
enum class BC 
{
    undefined, periodic,
    no_slip_wall, slip_wall,
    pressure_inflow, pressure_outflow, 
    mass_inflow, 
    charging_wall_pot, charging_wall_cur, 
    mix_wall_potL, mix_wall_potS, 
    mix_wall_potL_currS, mix_wall_currL_potS
};

// Different types of cells (MSE classification)
enum class CellType : uint8_t
{
    unset = 0,
    liquid = 1,
    boundary = 2,
    solid = 4
};

// Shape of various objects used in writing and reading numpy arrays
struct NumpyShapes
{
    // Constructor
    NumpyShapes(int num_levels, int dim):
        grid_shape(num_levels * dim, 0),
        domain_lo(num_levels * 3, 0.0),
        domain_hi(num_levels * 3, 0.0),
        shape_of_grid_metadata {static_cast<size_t>(num_levels), static_cast<size_t>(dim)},
        shape_of_domain {static_cast<size_t>(num_levels), 3ul},
        shape_npy_scalar(num_levels), 
        shape_npy_vector(num_levels), 
        shape_npy_conc(num_levels), 
        shape_npy_pressure(num_levels)
    {}

    // Members

    // Vector with the grid shape at each level
    vector<int> grid_shape;
    // Vector with the domain coordinates (low end) at each level; in 2D sim, write out nominal height in z axis
    vector<Real> domain_lo;
    // Vector with the domain coordinates (high end) at each level; in 2D sim, write out nominal height in z axis
    vector<Real> domain_hi;

    // Shape of the numpy array storing the grid metadata
    vector<size_t> shape_of_grid_metadata;
    // Shape of the numpy array storing the domain boundaries; always 3D to include nominal height
    vector<size_t> shape_of_domain;

    // Vector with the shape of numpy arrays to store a scalar field; one entry for each level
    vector< vector<size_t> > shape_npy_scalar;
    // Vector with the shape of numpy arrays to store a vector field matching spatial dimension (e.g. velocity); one entry for each level
    vector< vector<size_t> > shape_npy_vector;
    // Vector with the grid shape for concentrations; one entry for each level
    vector< vector<size_t> > shape_npy_conc;
    // Vector with the grid shape for pressure; one entry for each level
    vector< vector<size_t> > shape_npy_pressure;
};
