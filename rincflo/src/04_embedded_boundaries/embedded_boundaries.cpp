#include <rincflo.H>

// *********************************************************************************************************************
void Rincflo::MakeEBGeometry()
{
   /******************************************************************************
   * rincflo.geometry=<string> specifies the EB geometry. <string> can be one of    *
   * lattice, box, cylinder, annulus, sphere, spherecube...
   ******************************************************************************/
    #define FUNC_NAME "Rincflo::MakeEBGeometry"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    ParmParse pp("rincflo");

    std::string geom_type;
    pp.query("geometry", geom_type);

    // Report the number of grid cells and max levels
    AMREX_D_TERM(
    const long Nx = Geom(0).Domain().length(0);,
    const long Ny = Geom(0).Domain().length(1);,
    const long Nz = Geom(0).Domain().length(2);)
    #if (AMREX_IS_2D)
    Print() << format("Number of grid cells: (Nx, Ny) = ({:d}, {:d}).\n", Nx, Ny);
    #elif (AMREX_IS_3D)
    Print() << format("Number of grid cells: (Nx, Ny, Nz) = ({:d}, {:d}, {:d}).\n", Nx, Ny, Nz);
    #endif
    Print() << format("Max levels: {:d}.\n", max_level);

   /******************************************************************************
   *  CONSTRUCT EB                                                              *
   ******************************************************************************/
    if(geom_type == "")
    {
        Print() << "No EB geometry declared in inputs =>  Will build all regular geometry and set max_level=0.\n";
        MakeEB_regular();
        max_level = 0;
        m_has_boundary = false;
    }
    else if(geom_type == "lattice")
    {
        Print() << "Building lattice geometry.\n";
        MakeEB_lattice();
    }
    else if(geom_type == "box")
    {
        Print() << "Building box geometry.\n";
        MakeEB_box();
    }
    else if(geom_type == "cylinder")
    {
        Print() << "Building cylinder geometry.\n";
        MakeEB_cylinder();
    }
    else if(geom_type == "annulus")
    {
        Print() << "Building annulus geometry.\n";
        MakeEB_annulus();
    }
    else if(geom_type == "sphere")
    {
        Print() << "Building sphere geometry.\n";
        MakeEB_sphere();
    }
    else if(geom_type == "spherecube")
    {
        Print() << "Building spherecube geometry.\n";
        MakeEB_spherecube();
    }
    else if(geom_type == "tuscan")
    {
        Print() << "Building tuscan geometry.\n";
        MakeEB_tuscan();
    }
    else if(geom_type == "randomfibers")
    {
        Print() << "Building random fibers geometry.\n";
        MakeEB_random_fibers();
    }
    else
    {
        const string msg = 
        "Rincflo::MakeEBGeometry - Unknown geometry type.\n"
        "Allowed types are: none, box, cylinder, annulus, sphere, spherecube, tuscan, randomfibers, lattice.\n"
        "Enter Rincflo.geometry = \"\" to build all regular geometry.\n";
        Abort(msg);
    }
    Print() << "Done making the geometry ebfactory.\n";
    CheckCutCells();
    Print() << "Checked cutcells\n";

    

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::CheckCutCells()
{
    #define FUNC_NAME "Rincflo::CheckCutCells"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

   for (int lev = 0; lev <= finest_level; lev++)
    {
        auto const& fact = EBFactory(lev);
        auto& ld = *m_leveldata[lev];

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
            Array4<EBCellFlag const> const& flag = flagfab.const_array();
            Array4<Real const> vfrac = fact.getVolFrac().const_array(mfi);
            // Array4<Real const> afrac = fact.getBndryArea().const_array(mfi);
            Box const& bx = mfi.tilebox();
            const bool cover_multiple_cuts = m_cover_multiple_cuts;

            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (flag(i,j,k).isBoundary(cover_multiple_cuts) && (vfrac(i,j,k) < vfrac_almost_zero) )
                    {Print() << format("Rincflo::CheckCutCells() - Warning: Cut cell found with vfrac almost zero! vfrac={:8.2e}.\n", vfrac(i,j,k));}
                if (flag(i,j,k).isBoundary(cover_multiple_cuts) && (vfrac(i,j,k) > vfrac_almost_one) )
                    {Print() << format("Rincflo::CheckCutCells() - Warning: Cut cell found with vfrac almost one! vfrac={:0.9f}.\n", vfrac(i,j,k));}
            };
            ParallelFor(bx, func);
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
