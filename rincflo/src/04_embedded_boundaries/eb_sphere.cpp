#include <rincflo.H>

namespace EB2 = amrex::EB2;

/********************************************************************************
 *                                                                              *
 * Function to create a simple sphere EB.                                     *
 *                                                                              *
 ********************************************************************************/
void Rincflo::MakeEB_sphere()
{
    // Initialise sphere parameters
    bool inside = true;
    Real radius = 0.0002;
    Vector<Real> centervec(SpaceDim);

    // Get sphere information from inputs file.                               *
    ParmParse pp("sphere");

    pp.query("internal_flow", inside);
    pp.query("radius", radius);
    pp.getarr("center", centervec, 0, SpaceDim);
    Array<Real, SpaceDim> center = {AMREX_D_DECL(centervec[0], centervec[1], centervec[2])};

    // Print info about sphere
    Print() << " " << std::endl;
    Print() << " Internal Flow: " << inside << std::endl;
    Print() << " Radius:    " << radius << std::endl;
    #if (AMREX_IS_2D)
    Print() << format(" Center:    {:f}, {:f}.\n", center[0], center[1]);
    #else
    Print() << format(" Center:    {:f}, {:f}, {:f}.\n", center[0], center[1], center[2]);
    #endif

    // Build the sphere implicit function 
    EB2::SphereIF my_sphere(radius, center, inside);

    // Generate GeometryShop
    auto gshop = EB2::makeShop(my_sphere);

    // Build index space
    int max_level_here = 0;
    int max_coarsening_level = 100;
    EB2::Build(gshop, geom.back(), max_level_here, max_level_here + max_coarsening_level);
}
