#include <rincflo.H>

void Rincflo::WriteMyEBSurface()
{
    #if (AMREX_IS_3D)
    Print() << "Writing the geometry to a vtp file.\n" << std::endl;

    // Only write at the finest level!
    int lev = finest_level; 
    WriteEBSurface(grids[lev],dmap[lev],geom[lev],&EBFactory(lev));
    #endif  
}
