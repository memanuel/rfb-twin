#include <rincflo.H>

/********************************************************************************
 *                                                                              *
 * Function to create a simple sphereCUBE EB.                                     *
 *                                                                              *
 ********************************************************************************/
void Rincflo::MakeEB_spherecube()
{
    Print() << " " << std::endl;
    Print() << " SPHERECUBE: " << std::endl;

    // Build the sphere implicit function 
    EB2::SphereIF sphere(0.5, {AMREX_D_DECL(1.8, 1.8, 2.8)}, false);
    EB2::BoxIF cube({AMREX_D_DECL(1.85, 1.85, 2.85)}, {AMREX_D_DECL(2.5, 2.5, 3.5)}, false);
    auto cubesphere = EB2::makeUnion(sphere, cube);

    // Generate GeometryShop
    auto gshop = EB2::makeShop(cubesphere);

    // Build index space
    int max_level_here = 0;
    int max_coarsening_level = 100;
    EB2::Build(gshop, geom.back(), max_level_here, max_level_here + max_coarsening_level);
}
