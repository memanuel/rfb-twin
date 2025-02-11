#include <rincflo.H>
// #include <rincflo_derive_K.H>

// *********************************************************************************************************************
Real Rincflo::CalcAverageVolumeFraction(int lev) const
{
    #define FUNC_NAME "Rincflo::CalcAverageVolumeFraction"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    MultiFab const& vfrac = EBFactory(lev).getVolFrac();
    // The sum of the volume fractions on this level
    const Real sum_vfrac = vfrac.sum(0);
    // The grid size as a real
    const Real grid_size = CalcGridSizeReal(lev);
    // All the cells on a level have equal size, so the mean vfrac is the simple arithmetic mean of vfrac
    Real mean_vfrac {sum_vfrac / grid_size};

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return mean_vfrac;
}

// *********************************************************************************************************************
Real Rincflo::CalcTotalSurfaceArea(int lev) const
{
    #define FUNC_NAME "Rincflo::CalcTotalSurfaceArea(lev)"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The linear cell size (same in all directions)
    const Real dx = Geom(lev).CellSize(0);
    const Real dy = Geom(lev).CellSize(1);
    const Real dz = AMREX_D_PICK(1.0, m_height, Geom(lev).CellSize(2));

    // The multiplicative factor for the area of a cell (assumes grid cells squares / cubes)
    const Real cell_area = AMREX_D_PICK(1.0, dx*dz, dx*dx);

    // Get FABs for the cell area and boundary flags
    const auto& fact = EBFactory(lev);
    // Boundary area
    const auto& ba_fab = fact.getBndryArea().ToMultiFab(0.0, 0.0);
    // Boundary flags
    const auto& flags_fab = fact.getMultiEBCellFlagFab();
    // Are we covering multiply cut cells?
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    // Calculate the area in each MPI domain on this level
    auto sum_func = 
    [cell_area, cover_multiple_cuts]
    AMREX_GPU_HOST_DEVICE 
    (   Box const& bx, 
        Array4<Real const> const& ba, 
        Array4<EBCellFlag const> const& flags) -> Real
    {
        Real area = 0.0;
        auto func_b = 
        [=, &area] (int i, int j, int k) noexcept
        {
            if (flags(i,j,k).isBoundary(cover_multiple_cuts))
                {area += ba(i,j,k,0) * cell_area;}
        };
        Loop(bx, func_b);
        return area;
    };

    // Accumulate the total area over one MPI worker
    Real sum_area = ReduceSum(ba_fab, flags_fab, 0, sum_func);

    // Sum the area over all the MPI workers
    ReduceRealSum(sum_area);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum_area;
}

// *********************************************************************************************************************
Real Rincflo::CalcTotalSurfaceArea() const
{
    #define FUNC_NAME "Rincflo::CalcTotalSurfaceArea()"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Delegate with level=0
    int lev {0};
    Real sum_area = CalcTotalSurfaceArea(lev);
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum_area;
}

// *********************************************************************************************************************
Real Rincflo::CalcNetFluxCurrent() const
{
    #define FUNC_NAME "Rincflo::CalcNetFluxCurrent"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Analysis is at the top level (whole domain)
    int lev {0};

    // The velocity of the flow at this level
    const auto& vel = m_leveldata[lev]->velocity;

    // The state of charge at this level
    const auto& soc = m_leveldata[lev]->soc;

    // The species concentrations at this level
    const auto& conc = m_leveldata[lev]->conc;

    // Number of variables in new MultiFabs
    constexpr int nvar {1};
    // Number of ghost cells in new MultiFabs
    const int ngrow {soc.nGrow()};

    // Surface area of one cell normal to the x-direction
    const Real normal_surface_cell {CalcCellArea(lev)};

    // MultiFab with normal area in each cell
    MultiFab normal_area_mf;
    normal_area_mf.define(soc.boxArray(), soc.DistributionMap(), nvar, ngrow);
    normal_area_mf.setVal(normal_surface_cell);

    // MultiFab with soc flux in each cell
    MultiFab soc_flux;
    // Initialize the flux to have the same footprint as the state of charge
    soc_flux.define(soc.boxArray(), soc.DistributionMap(), nvar, ngrow);
    // Set soc_flux to zero; skipping this step leads to garbage values in the flux
    soc_flux.setVal(0.0);
    // Add the product of vel[0] and soc to soc_flux; vel[0] = vel_x
    MultiFab::AddProduct(soc_flux, soc, 0, vel, 0, 0, 1, ngrow);

    // Arguments for SumOverBoundaryCells
    constexpr int dir {0};
    constexpr bool lo_in {true};
    constexpr bool lo_out {false};
    constexpr int comp {0};
    constexpr bool local {false};
    constexpr bool is_nodal {false};

    // Calculate flow of liquid at the inlet and outlet
    Real soc_flux_in  = SumOverBoundaryCells(dir, lo_in , soc_flux, comp, is_nodal, local);
    Real soc_flux_out = SumOverBoundaryCells(dir, lo_out, soc_flux, comp, is_nodal, local);
    // The net flux in units of meter^3 / sec * soc 
    Real net_flux = soc_flux_out - soc_flux_in;
    // Muliplier from dimensionless SOC flux to current in amps
    const Real flux_multiplier = normal_surface_cell * m_conc_redox_tot * m_nF_const;
    // The equivalent current in amps
    Real equiv_current = net_flux * flux_multiplier;

    // Print flux in / out in milliamps if verbosity is high
    if (m_verbose > 1)
    {
        Real current_in_ma  = soc_flux_in  * flux_multiplier * 1.0E3;
        Real current_out_ma = soc_flux_out * flux_multiplier * 1.0E3;
        Real equiv_current_ma = equiv_current * 1.0E3;
        Print() << format("current_in    = {:12.9f} mA.\n", current_in_ma);
        Print() << format("current_out   = {:12.9f} mA.\n", current_out_ma);
        Print() << format("equiv_current = {:12.9f} mA.\n", equiv_current_ma);
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME

    return equiv_current;
}

// *********************************************************************************************************************
Real Rincflo::CalcConservedRedoxConc(bool verbose) const
{
    #define FUNC_NAME "Rincflo::CalcNetFluxCurrent"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Analysis is at the top level (whole domain)
    int lev {0};

    // The species concentrations at this level
    const auto& conc = m_leveldata[lev]->conc;
    // Number of ghost cells in new MultiFabs
    // const int ngrow = nghost_state();
    const int ngrow = 0;
    // Number of variables in new MultiFab to be built
    constexpr int nvar {1};
    // MultiFab with total concentration of redox species in each cell, which should be conserved
    MultiFab conc_tot;

    // Initialize the redox_conc_tot have the same footprint as the concentration
    conc_tot.define(conc.boxArray(), conc.DistributionMap(), nvar, ngrow);
    // Initialize it to zero
    conc_tot.setVal(0.0);
    // Add the first two components of concentration: oxidized and reduced species
    constexpr int srccomp = 0;
    constexpr int dstcomp = 0;
    constexpr int numcomp = 1;
    MultiFab::Add(conc_tot, conc, srccomp+0, dstcomp, numcomp, ngrow);
    MultiFab::Add(conc_tot, conc, srccomp+1, dstcomp, numcomp, ngrow);

    // For a mysterious reason, the following code fails with a memory corruption error
    // malloc(): invalid size (unsorted)
    // MultiFab::Add(conc_tot, conc, srccomp, dstcomp, 2, ngrow);
    // The above is less efficient but it works :)

    // Calculate the mean of the total redox concentration
    constexpr int comp = 0;
    constexpr bool local = false;
    Real tot_mean = VolumeWeightedAvg(lev, conc_tot, comp, local);

    // Placeholders for the min, max over liquid and boundary regions
    Real tot_min_liquid, tot_max_liquid, tot_min_boundary, tot_max_boundary;
    
    // Calculate the min / max in the regions only when required
    if (verbose)
    {
        // Calculate the min and max over liquid cells
        tot_min_liquid = MinOverLiquid(lev, conc_tot, comp, local);
        tot_max_liquid = MaxOverLiquid(lev, conc_tot, comp, local);

        // Calculate the min and max over boundary cells - only when there is a boundary
        if (has_eb())
        {
            tot_min_boundary = MinOverBoundary(lev, conc_tot, comp, local);
            tot_max_boundary = MaxOverBoundary(lev, conc_tot, comp, local);
        }
    }

    // Calculate the standard deviation of the total redox concentration
    // conc_tot will be modified in place to contain the difference from the mean
    conc_tot.plus(-tot_mean, dstcomp, 1, ngrow);
    // square the difference in place
    MultiFab::Multiply(conc_tot, conc_tot, dstcomp, dstcomp, 1, ngrow);
    // Rename this Fab to make it clear that it's now the squared difference
    const auto& conc_diff2 = conc_tot;
    // compute the volume weighted average of the squared differences, then take the square root
    Real tot_var = VolumeWeightedAvg(lev, conc_diff2, dstcomp, local);
    Real tot_std = std::sqrt(tot_var);
    // Compute the error due to the mean not matching the expected conserved total
    Real cons_err = std::abs(tot_mean - m_conc_redox_tot);
    // Compute the RMS error
    Real rms_err = std::sqrt(tot_var + cons_err*cons_err);

    // Report detailed diagnostics on conserved redox concentration if requested
    if (verbose)
    {
        // Calculate relative RMS error
        Real rms_err_rel = rms_err / m_conc_redox_tot;
        // Calculate maximum relative error
        Real max_err_liquid = std::max(m_conc_redox_tot - tot_min_liquid, tot_max_liquid - m_conc_redox_tot);
        Real max_err_boundary = has_eb() ?  
            std::max(m_conc_redox_tot - tot_min_boundary, tot_max_boundary - m_conc_redox_tot) : 0.0;
        Real max_err = std::max(max_err_liquid, max_err_boundary);
        Real max_err_rel = max_err / m_conc_redox_tot;
        // Report the summary statistics
        Print() << format("redox_conc summary statistics:\n");
        Print() << format("mean           = {:12.9f}.\n", tot_mean);
        Print() << format("std            = {:5.3e}.\n", tot_std);
        Print() << format("cons. error    = {:5.3e}.\n", cons_err);
        Print() << format("RMS error      = {:5.3e}.\n", rms_err);
        Print() << format("RMS err (rel)  = {:5.3e}.\n", rms_err_rel);
        Print() << format("Max err (rel)  = {:5.3e}.\n", max_err_rel);
        Print() << format("liquid range   = [{:12.9f}, {:12.9f}].\n", tot_min_liquid, tot_max_liquid);
        if (has_eb())
            {Print() << format("boundary range = [{:12.9f}, {:12.9f}].\n", tot_min_boundary, tot_max_boundary);}

    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME

    return rms_err;
}