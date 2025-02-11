#include <rincflo.H>
#include <rincflo_derive_K.H>

// *********************************************************************************************************************
Real Rincflo::CalcCellArea (int lev) const
{
    // Grid dimensions along each axis; guaranteed to have at least 2 dimensions
    const Real dx = Geom(lev).CellSize(0);
    const Real dy = Geom(lev).CellSize(1);
    return dx * dy;
}

// *********************************************************************************************************************
Real Rincflo::CalcCellVolume (int lev) const
{
    // In two dimensions, use the nominal height for dz; in three dimensions, use the real height
    const Real dz = AMREX_D_PICK(1.0, m_height, Geom(lev).CellSize(2));
    return CalcCellArea(lev) * dz;
}

// *********************************************************************************************************************
// Compute derived geometry attributes: cell_type, area, volume, area_over_volume
void Rincflo::CalcGeometry()
{
    #define FUNC_NAME "Rincflo::CalcGeometry"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Embedded boundaries
        #if (AMREX_USE_EB)
        const auto& fact = EBFactory(lev);
        const auto& flags_mf = fact.getMultiEBCellFlagFab();
        #endif
        const bool cover_multiple_cuts = m_cover_multiple_cuts;

        // The linear step size (same in all directions)
        const Real dx = Geom(lev).CellSize(0);
        // The inverse step size
        const Real dx_inv = 1.0 / dx;
        // The multiplicative factor for the area of a cell; in square meters
        Real cell_area = CalcCellArea(lev);
        // The multiplicative factor for the volume of a cell; in cubic meters
        Real cell_volume = CalcCellVolume(lev);

        // The level data
        auto& ld = *m_leveldata[lev];

        #if (DEBUG_RFB)
        int mfi_counter {0};
        #endif
        DEBUG_PRINT(format("Set up EB and level data for level {:d}.\n", lev));

        // Initialize the cell type to 0.0 before the iterations
        ld.cell_type.setVal(0.0);

        // Iterate over multifabs
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for(MFIter mfi(ld.cell_type, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            #if (AMREX_USE_EB)
            // Get the type of this box (e.g. regular, single valued, covered)
            const EBCellFlagFab& flags_box = flags_mf[mfi];
            auto const& typ = flags_box.getType(bx);
            #endif

            // Build fabs for the output fields (not const)
            Array4<Real> const& cell_type = ld.cell_type.array(mfi);
            Array4<Real> const& area = ld.area.array(mfi);
            Array4<Real> const& volume = ld.volume.array(mfi);
            Array4<Real> const& area_over_volume = ld.area_over_volume.array(mfi);
            // Array4<Real> const& cell_centroid = ld.cell_centroid.array(mfi);
            // Array4<Real> const& bdry_centroid = ld.bdry_centroid.array(mfi);
            // Array4<Real> const& bdry_normal = ld.bdry_normal.array(mfi);

            #if (AMREX_USE_EB)
            if (typ == FabType::regular)
            // Regular cells (all liquid)
            {
                // DEBUG_PRINT("Processing regular region.\n");
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // These cells are all liquid; zero surface area and full volume
                    cell_type(i,j,k) = static_cast<Real>(CellType::liquid);
                    area(i,j,k) = 0.0;
                    volume(i,j,k) = cell_volume;
                    area_over_volume(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
            // Covered cells get default values for all solid / no liquid
            else if (typ == FabType::covered)
            {
                // DEBUG_PRINT("Processing covered region.\n");
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cell_type(i,j,k) = static_cast<Real>(CellType::solid);
                    area(i,j,k) = 0.0;
                    volume(i,j,k) = 0.0;
                    area_over_volume(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
            // The boundary cells
            else
            {
                // DEBUG_PRINT("Processing single valued or multivalued region.\n");
                // Calculate the volume fraction (liquid) and area of the boundary
                Array4<Real const> const& vfrac = fact.getVolFrac().const_array(mfi);
                // DEBUG_PRINT("Built vfrac const array.\n");
                Array4<Real const> const& afrac = fact.getBndryArea().const_array(mfi);
                // DEBUG_PRINT("Built afrac const array.\n");
                // DEBUG_PRINT("Built vfrac and afrac const arrays.\n");
                const auto& flag = flags_box.const_array();
                // These cells may be liquid, boundary or solid; must check all possibilities.
                // Treat multi-valued cells as being covered.

                // Make a preliminary pass to assign the type of each cell
                auto func_type = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // These cells are all liquid
                    if (flag(i,j,k).isRegular())
                        {cell_type(i,j,k) = static_cast<Real>(CellType::liquid);}
                    // These are cut cells (boundary)
                    else if (flag(i,j,k).isBoundary(cover_multiple_cuts))
                    {
                        // Is this cell almost all liquid?
                        if (vfrac(i,j,k) < m_eb2_small_volfrac)
                            {cell_type(i,j,k) = static_cast<Real>(CellType::liquid);}
                        // Is this cell almost all solid?
                        else if (vfrac(i,j,k) > m_eb2_large_volfrac)
                            {cell_type(i,j,k) = static_cast<Real>(CellType::solid);}
                        // Treat this cell is a boundary cell
                        else
                            {cell_type(i,j,k) = static_cast<Real>(CellType::boundary);}
                    }
                    // This category includes covered cells and multivalued cells (which we treat as covered); they're all solid
                    else
                        {cell_type(i,j,k) = static_cast<Real>(CellType::solid);}
                };
                ParallelFor(bx, func_type);

                // Make a second pass to assign the area, volume, and area_over_volume of each cell
                auto func_av = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // These cells are all liquid
                    if (cell_type(i,j,k) == static_cast<Real>(CellType::liquid))
                    {
                        area(i,j,k) = 0.0;
                        volume(i,j,k) = cell_volume;
                        area_over_volume(i,j,k) = 0.0;
                    }
                    // These are cut cells (boundary)
                    else if (cell_type(i,j,k) == static_cast<Real>(CellType::boundary))
                    {
                        area(i,j,k) = afrac(i,j,k) * cell_area;
                        volume(i,j,k) = vfrac(i,j,k) * cell_volume;
                        area_over_volume(i,j,k) = afrac(i,j,k) / vfrac(i,j,k) * dx_inv;
                    }
                    // These are covered cells (all solid)
                    else if (cell_type(i,j,k) == static_cast<Real>(CellType::solid))
                    {
                        area(i,j,k) = 0.0;
                        volume(i,j,k) = 0.0;
                        area_over_volume(i,j,k) = 0.0;
                    }
                };
                ParallelFor(bx, func_av);
            }
        } // for / mf_iter
        #else
        auto func = 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // These cells are all liquid; zero surface area and full volume
            cell_type(i,j,k) = static_cast<Real>(CellType::liquid);
            area(i,j,k) = 0.0;
            volume(i,j,k) = cell_volume;
            area_over_volume(i,j,k) = 0.0;
        };
        ParallelFor(bx, func);
        #endif
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
