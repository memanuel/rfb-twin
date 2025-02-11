#include <rincflo.H>
#include <rincflo_derive_K.H>

// *********************************************************************************************************************
// Compute pot_diff, overpot, current, curr_dens. Note that soc is usually computed in ImposeRedoxConcConstratint. 
// It's also updated here to ensure it's up to date.
void Rincflo::CalcDerivedReact()
{
    #define FUNC_NAME "Rincflo::CalcDerivedReact"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Quit early if this isn't a reaction model
    if (!is_Reaction())
    {
        Print() << "Warning - called CalcDerivedReact() but not a reaction simulation. Returning early.\n";
        return;
    }
    
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Embedded boundaries
        #if (AMREX_USE_EB)
        const auto& fact = EBFactory(lev);
        const auto& flags_mf = fact.getMultiEBCellFlagFab();
        #endif
        const  bool cover_multiple_cuts = m_cover_multiple_cuts;

        // The linear cell size (same in all directions)
        const Real dx = Geom(lev).CellSize(0);
        #if (AMREX_IS_2D)
        // The multiplicative factor for the area of a cell
        const Real cell_area = dx;
        #else
        // The multiplicative factor for the area of a cell
        const Real cell_area = dx * dx;
        #endif
        // The multiplicative factor for the volume of a cell
        const Real cell_volume = cell_area * dx;

        // The level data
        auto& ld = *m_leveldata[lev];

        // Alias the input fields from this level; these are const
        const auto& ld_conc = ld.conc;
        const auto& ld_conc_o = ld.conc_o;
        const auto& ld_epotS = ld.epotS;
        const auto& ld_epotL = ld.epotL;
        const auto& ld_pot_diff = ld.pot_diff;
        const auto& ld_overpot = ld.overpot;
        const auto& ld_source = ld.source;
        const auto& ld_area = ld.area;
        const auto& ld_volume = ld.volume;

        // Alias the output (derived) fields from these levels; need to write to these so not const
        auto& ld_soc = ld.soc;
        auto& ld_source_mol = ld.source_mol;
        auto& ld_current = ld.current;
        auto& ld_curr_dens = ld.curr_dens;
        
        // DEBUG_PRINT(format("Set up EB and level data for level {:d}.\n", lev));

        // Constant NaN so we can record missing values
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        // Iterate over multifabs
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for(MFIter mfi(ld_conc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            #if (AMREX_USE_EB)
            // Get the type of this box (e.g. regular, single valued, covered)
            const EBCellFlagFab& flags_box = flags_mf[mfi];
            auto const& typ = flags_box.getType(bx);
            bool is_liquid = fabIsLiquid(typ);
            bool is_solid = fabIsSolid(typ);
            bool is_boundary = fabIsBoundary(typ);
            #endif

            // Build constant arrays for the input fields (const)
            Array4<Real const> const& conc = ld_conc.const_array(mfi);
            Array4<Real const> const& conc_o = ld_conc_o.const_array(mfi);
            Array4<Real const> const& epotS = ld_epotS.const_array(mfi);
            Array4<Real const> const& epotL = ld_epotL.const_array(mfi);
            Array4<Real const> const& pot_diff = ld_pot_diff.const_array(mfi);
            Array4<Real const> const& overpot = ld_overpot.const_array(mfi);
            Array4<Real const> const& source = ld_source.const_array(mfi);
            Array4<Real const> const& area = ld_area.const_array(mfi);
            Array4<Real const> const& volume = ld_volume.const_array(mfi);

            // Build non-constant arrays for the output fields
            Array4<Real> const& soc = ld_soc.array(mfi);
            Array4<Real> const& source_mol = ld_source_mol.array(mfi);
            Array4<Real> const& current = ld_current.array(mfi);
            Array4<Real> const& curr_dens = ld_curr_dens.array(mfi);

            #if (AMREX_USE_EB)
            // This box contains all liquid cells
            if (is_liquid)
            {
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Calculate the state of charge (SOC) of the cell
                    soc(i,j,k) = conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
                    // Current on non-boundary cells is zero by convention
                    source_mol(i,j,k) = 0.0;
                    current(i,j,k) = 0.0;
                    curr_dens(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
            // This box contains all solid cells
            else if (is_solid)
            {
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    soc(i,j,k) = NaN;
                    source_mol(i,j,k) = 0.0;
                    current(i,j,k) = 0.0;
                    curr_dens(i,j,k) = NaN;
                };
                ParallelFor(bx, func);
            }
            // Boxes with a mix of solid, liquid and boundary cells; must check all possibilities
            else if (is_boundary)
            {
                const auto& flag = flags_box.const_array();
                auto func = 
                [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Regular (liquid) cells have a SOC and zero current
                    if (flag(i,j,k).isRegular()) 
                    {
                        soc(i,j,k) = conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
                        source_mol(i,j,k) = 0.0;
                        current(i,j,k) = 0.0;
                        curr_dens(i,j,k) = NaN;
                    }
                    // Covered (solid) cells have no SOC and zero current
                    else if (flag(i,j,k).isCovered()) 
                    {
                        soc(i,j,k) = NaN;
                        source_mol(i,j,k) = 0.0;
                        current(i,j,k) = 0.0;
                        curr_dens(i,j,k) = NaN;
                    }
                    // Cut cells (boundary) have both SOC and source / current terms
                    else if (flag(i,j,k).isBoundary(cover_multiple_cuts))
                    {
                        // SOC
                        soc(i,j,k) = conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1));
                        // Source term in moles / second
                        // Volume is in m^3 and source is in mol / m^3 / sec
                        // Therefore source * volume is in units of mol / second
                        source_mol(i,j,k) = source(i,j,k) * volume(i,j,k);
                        // Calculate current in amperes from source term
                        current(i,j,k) = m_nF_const * source_mol(i,j,k);
                        // Calculate current density from current and area
                        curr_dens(i,j,k) = current(i,j,k) / area(i,j,k);
                    }
                };
                ParallelFor(bx, func);
            }
        } // for / mf_iter
        #else
        auto func = 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Current on non-boundary cells is zero by convention
            source_mol(i,j,k) = 0.0;
            current(i,j,k) = 0.0;
            curr_dens(i,j,k) = 0.0;
        };
        ParallelFor(bx, func);
        #endif
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
Real Rincflo::CalcTotalCurrent () const
{
    const auto& current = m_leveldata[finest_level]->current;
    return current.sum(0);
}

// *********************************************************************************************************************
Real Rincflo::CalcAvgCurrentDensity ()
{
    // Always calculate total current and area at the finest level
    int lev = finest_level;

    // The area of the embedded boundary
    const auto& fact = EBFactory(lev);
    const auto& aeb_mcf = fact.getBndryArea();
    MultiFab aeb = aeb_mcf.ToMultiFab(0.0,0.0);
    Real total_area = aeb.sum(0);

    // The total current
    Real total_curr = CalcTotalCurrent();

    // The average current density
    Real avg_current_density = total_curr / total_area;
    return avg_current_density;
}
