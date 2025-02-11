#include <rincflo.H>

/****************************************************************************
 * Function to create a simple rectangular box with EB walls.               *
 ****************************************************************************/
void Rincflo::MakeEB_box()
{
    ParmParse pp("amr");
    int amr_max_lev;
    pp.query("max_level", amr_max_lev);

    ParmParse pp2("geo_eb");
    int required_coarsening_level=amr_max_lev;
    pp2.query("req_lev", required_coarsening_level);
    int max_coarsening_level=20;
    pp2.query("max_lev", max_coarsening_level);

    if (required_coarsening_level < amr_max_lev )
    {
        Print() << format(
            "ERROR?? amr_max= {:d}, req={:d}, max_coarse={:d}.\n", 
            amr_max_lev, required_coarsening_level, max_coarsening_level);
    }

    // Get box information from inputs file
    ParmParse pp3("box");
    AMREX_D_TERM(
    Real xlo=0.0;,
    Real ylo=0.0;,
    Real zlo=0.0;)
    AMREX_D_TERM(
    Real xhi=1.0;,
    Real yhi=1.0;,
    Real zhi=1.0;)
    bool fluid_inside=false;
    AMREX_D_TERM(
    pp3.query("xlo", xlo);,
    pp3.query("ylo", ylo);,
    pp3.query("zlo", zlo);)
    AMREX_D_TERM(
    pp3.query("xhi", xhi);,
    pp3.query("yhi", yhi);,
    pp3.query("zhi", zhi);)
    pp3.query("fluid_inside", fluid_inside);

    EB2::BoxIF box({AMREX_D_DECL(xlo,ylo,zlo)},{AMREX_D_DECL(xhi,yhi,zhi)}, fluid_inside);
    auto gshop = EB2::makeShop(box);
    EB2::Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level);      
}
