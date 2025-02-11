#include <rincflo.H>

/********************************************************************************
 * Function to create an annular cylinder EB.                                     *
 ********************************************************************************/
void Rincflo::MakeEB_annulus()
{
    // Initialise annulus parameters
    int direction = 0;
    Real outer_radius = 0.0002;
    Real inner_radius = 0.0001;
    Vector<Real> outer_centervec(SpaceDim);
    Vector<Real> inner_centervec(SpaceDim);

    // Get annulus information from inputs file.                               *
    ParmParse pp("annulus");

    pp.query("direction", direction);
    pp.query("outer_radius", outer_radius);
    pp.query("inner_radius", inner_radius);
    pp.getarr("outer_center", outer_centervec, 0, SpaceDim);
    pp.getarr("inner_center", inner_centervec, 0, SpaceDim);
    Array<Real, SpaceDim> outer_center = 
        {AMREX_D_DECL(outer_centervec[0], outer_centervec[1], outer_centervec[2])};
    Array<Real, SpaceDim> inner_center = 
        {AMREX_D_DECL(inner_centervec[0], inner_centervec[1], inner_centervec[2])};

    // MakeEB_annulus: outer and inner cylinders must have same center coordinate per direction
    AMREX_ASSERT(outer_center[direction] == inner_center[direction]);

    // Compute distance between cylinder centres
    Real offset = 0.0;
    for(int i = 0; i < SpaceDim; i++)
        {offset += pow(outer_center[i] - inner_center[i], 2);}
    offset = sqrt(offset); 

    // Check that the inner cylinder is fully contained in the outer one
    Real smallest_gap_width = outer_radius - inner_radius - offset;
    AMREX_ASSERT(smallest_gap_width >= 0.0);

    // Compute standoff - measure of eccentricity
    Real standoff = 100 * smallest_gap_width / (outer_radius - inner_radius);
    AMREX_ASSERT((standoff >= 0) && (standoff <= 100));
    
    // Print info about annulus
    Print() << "\n";
    Print() << " Direction:       " << direction << std::endl;
    Print() << " Outer radius:    " << outer_radius << std::endl;
    Print() << " Inner radius:    " << inner_radius << std::endl;
    #if (AMREX_IS_2D)
    Print() << format("Outer center:   {:f}, {:f}.\n", outer_center[0], outer_center[1]);
    Print() << format("Inner center:   {:f}, {:f}.\n", inner_center[0], inner_center[1]);
    #else
    Print() << format("Outer center:   {:f}, {:f}, {:f}.\n", outer_center[0], outer_center[1], outer_center[2]);
    Print() << format("Inner center:   {:f}, {:f}, {:f}.\n", inner_center[0], inner_center[1], inner_center[2]);
    #endif
    Print() << " Offset:          " << offset << std::endl;
    Print() << " Smallest gap:    " << smallest_gap_width << std::endl;
    Print() << " Standoff:        " << standoff << std::endl;

    // Build the annulus implifict function as a union of two cylinders
    EB2::CylinderIF outer_cyl(outer_radius, direction, outer_center, true);
    EB2::CylinderIF inner_cyl(inner_radius, direction, inner_center, false);
    auto annulus = EB2::makeUnion(outer_cyl, inner_cyl);

    // Generate GeometryShop
    auto gshop = EB2::makeShop(annulus);

    // Build index space
    int max_level_here = 0;
    int max_coarsening_level = 100;
    EB2::Build(gshop, geom.back(), max_level_here, max_level_here + max_coarsening_level);
}
