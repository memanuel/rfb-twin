#include <rincflo.H>

/********************************************************************************
 *                                                                              *
 * Function to create a simple cylinder EB.                                     *
 *                                                                              *
 ********************************************************************************/
void Rincflo::MakeEB_cylinder()
{
    // Initialise cylinder parameters
    bool inside = true;
    Real radius = 0.0002;
    int direction = 0;
    Vector<Real> centervec(SpaceDim);

    // Get cylinder information from inputs file.                               *
    ParmParse pp("cylinder");

    pp.query("internal_flow", inside);
    pp.query("radius", radius);
    pp.query("direction", direction);
    pp.getarr("center", centervec, 0, SpaceDim);
    Array<Real, SpaceDim> center = {AMREX_D_DECL(centervec[0], centervec[1], centervec[2])};

    // Print info about cylinder
    Print() << " " << std::endl;
    Print() << " Internal Flow: " << inside << std::endl;
    Print() << " Radius:    " << radius << std::endl;
    Print() << " Direction: " << direction << std::endl;
    #if (AMREX_IS_2D)
    Print() << format(" Center:    {:f}, {:f} \n", center[0], center[1]);
    #else
    Print() << format(" Center:    {:f}, {:f}, {:f}\n", center[0], center[1], center[2]);
    #endif

    // Build the Cylinder implicit function representing the curved walls     
    EB2::CylinderIF my_cyl(radius, direction, center, inside);

    // Generate GeometryShop
    auto gshop = EB2::makeShop(my_cyl);

    // Build index space
    int max_level_here = 0;
    int max_coarsening_level = 100;
    EB2::Build(gshop, geom.back(), max_level_here, max_level_here + max_coarsening_level);
}
