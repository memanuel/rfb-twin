#include <rincflo.H>

// ************************************************************************************************
void Rincflo::Analysis()
{
    // Status message for analysis output
    PrintStars();
    Print() << "Analysis\n\n";

    // Compute the geometry
    CalcGeometry();

    // Stages of the analysis
    AnalyzeGeometry();
    AnalyzeFlow();

    // Only analyze the current and potential if this is a reaction model
    if (is_Reaction()) 
    {
        // Update derived reaction quantities before analyzing them
        CalcDerivedReact();
        AnalyzeCurrent();
        AnalyzePotential();
    }
}

// *********************************************************************************************************************
long Rincflo::CalcGridSize (int lev) const
{
    // Calculate the number of grid cells
    AMREX_D_TERM(
    const long Nx = Geom(lev).Domain().length(0);,
    const long Ny = Geom(lev).Domain().length(1);,
    const long Nz = Geom(lev).Domain().length(2);)
    return AMREX_D_TERM(Nx, * Ny, * Nz);
}

// *********************************************************************************************************************
void Rincflo::AnalyzeGeometry() const
{
    #define FUNC_NAME "Rincflo::AnalyzeGeometry"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    
    // Analysis is at the top level (whole domain)
    int lev {0};

    // The volume of one grid cell; convert from meter^3 to cubic centimeters
    Real cell_volume_cc = CalcCellVolume(lev) * 1.0E6;

    // Number of grid cells
    Real grid_size = CalcGridSizeReal(lev);

    // The volume of liquid and solid in cubic centimeters
    Real volume_domain = grid_size * cell_volume_cc;

    // Get the volume of the liquid from the level data
    const auto& volume = m_leveldata[lev]->volume;
    // Convert the volume of the liquid from meter^3 to cubic centimeters
    Real volume_liquid = volume.sum() * 1.0E6;
    // The mean volume fraction (i.e. porosity)
    Real porosity = volume_liquid / volume_domain;

    // The remaining volume is occupied by the solid
    Real volume_solid = volume_domain - volume_liquid;

    // The length of the domain along each axis; in centimeters
    constexpr Real m_to_cm {100.0};
    const Real Lx = Geom(lev).CellSize(0) * Geom(lev).Domain().length(0) * m_to_cm;
    const Real Ly = Geom(lev).CellSize(1) * Geom(lev).Domain().length(1) * m_to_cm;
    const Real Lz = AMREX_D_PICK(m_height, m_height, Geom(lev).CellSize(2) * Geom(lev).Domain().length(1)) * m_to_cm;
    // The nominal surface area along each axis; in square centimeters
    Real area_nominal_xy = Lx * Ly;
    Real area_nominal_yz = Ly * Lz;
    Real area_nominal_xz = Lx * Lz;

    // Get the surface area of the embedded boundary from the level data
    const auto& area = m_leveldata[lev]->area;
    // Convert the surface area from meter^2 to cm^2
    Real area_wire = area.sum() * (m_to_cm * m_to_cm);
    // Compute the area per volume in cm^-1
    Real area_per_volume = area_wire / volume_domain;
    // The total number of boxes
    long n_box_tot {0};
    // The total number of cells
    long n_cell_tot {0};
    // The total number of boundary cells
    long n_cell_bdry_tot {0};

    // Report results - grid summary
    Print() << format("*** Grid Summary ***\n");
    Print() << format("{:6s} : {:8s} : {:10s} : {:8s} : {:10s}\n", "Level", "Boxes", "Cells", "\% Domain", "Cut Cells");
    for (int lev=0; lev <= max_level; ++lev)
    {
        const auto& ba = grids[lev];
        const auto& cell_type = m_leveldata[lev]->cell_type;
        // Number of boxes and cells in the box array on this level 
        long n_box = ba.size();
        long n_cell = ba.numPts();

        // Number of cells in the domain on this level
        long n_cell_dom = Geom(lev).Domain().numPts();
        Real domain_frac = static_cast<Real>(n_cell) / static_cast<Real>(n_cell_dom);
        // The number of boundary cells; sum over cell_type as a hack
        long n_cell_bdry = static_cast<long>(SumOverBoundary(lev, cell_type, 0, false)) / static_cast<long>(CellType::boundary);
        // The grand total of boxes, cells and boundary cells
        n_box_tot += n_box;
        n_cell_tot += n_cell;
        n_cell_bdry_tot += n_cell_bdry;
        // Report results for this level
        Print() << format("{:6d} : {:8d} : {:10d} : {:8.6f} : {:10d}\n", lev, n_box, n_cell, domain_frac, n_cell_bdry);
    }
    // Report total number of cells
    Print() << format("Total  : {:8d} : {:10d} : {:8s} : {:10d}\n", n_box_tot, n_cell_tot, "", n_cell_bdry_tot);


    // Report results - volume and surface area
    Print() << format("\n*** Geometry Analysis ***\n");
    Print() << format("volume - liquid       = {:12.6e} cm^3.\n", volume_liquid);
    Print() << format("volume - solid        = {:12.6e} cm^3.\n", volume_solid);
    Print() << format("volume - domain       = {:12.6e} cm^3.\n", volume_domain);
    Print() << format("porosity              = {:0.9f}.\n", porosity);
    Print() << format("nominal area (XY)     = {:12.6e} cm^2.\n", area_nominal_xy);
    Print() << format("nominal area (YZ)     = {:12.6e} cm^2.\n", area_nominal_yz);
    // Print() << format("nominal area (XZ)     = {:12.6e} cm^2.\n", area_nominal_xz);
    Print() << format("wire surface area     = {:12.6e} cm^2.\n", area_wire);
    Print() << format("area per volume       = {:12.6e} cm^-1.\n", area_per_volume);
    Print() << "\n";

    // Print minimum volume fractions table
    if (m_verbose > 0)
    {
        // Header rows
        Print() << format("Minimum volume fractions for liquid and solid\n");
        Print() << format("{:6s} : {:^12s} : {:^12s}\n", "lev", "liq", "sol");

        // Get minimum of liquid and solid fraction; do this by computing min and max of vfrac
        for(int lev = 0; lev <= finest_level; lev++)
        {
            const auto& ld = *m_leveldata[lev];
            // The volume of one grid cell in cubic meters
            Real cell_volume_m3 = CalcCellVolume(lev);
            // These multifabs are not local
            bool local = false;
            // The minimum vfrac of liquid (a.ka. vfrac)
            Real vfrac_liq_min = MinOverBoundary(lev, ld.volume, local) / cell_volume_m3;
            // The minimum vfrac of solid (a.k.a. 1.0 - vfrac)
            Real vfrac_sol_min = 1.0 - MaxOverBoundary(lev, ld.volume, local) / cell_volume_m3;
            Print() << format("{:6d} : {: 12.6e} : {: 12.6e}\n", lev, vfrac_liq_min, vfrac_sol_min);
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return;
}

// *********************************************************************************************************************
void Rincflo::AnalyzeFlow()
{
    #define FUNC_NAME "Rincflo::AnalyzeFlow"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    
    // Analysis is at the top level (whole domain)
    int lev {0};

    // Number of variables in new MultiFabs
    constexpr int nvar {1};

    // Vectors to store the mean, min and max velmag by level
    vector<Real> velmag_mean(finest_level+1);
    vector<Real> velmag_min(finest_level+1);
    vector<Real> velmag_max(finest_level+1);
    // Calculate summary statistics of velmag on each level
    for (int lev=0; lev <= finest_level; ++lev)
    {
        // The velocity of the flow at this level
        const auto& vel = m_leveldata[lev]->velocity;
        // Number of ghost cells in new MultiFab
        const int ngrow {m_leveldata[lev]->velocity.nGrow()};
        // The velocity magnitude at this level
        MultiFab velmag(vel.boxArray(), vel.DistributionMap(), nvar, ngrow);
        CalcVelMag(lev, velmag, vel);

        // Arguments for sum, min and max operations
        constexpr int comp {0};
        constexpr bool local {false};
        // Conversion factor from m/s to cm/s
        constexpr Real mps_to_cmps = 100.0;

        // Summary statistics for velmag
        long n_cell = CountLiquidCells(lev, local);
        velmag_mean[lev] = SumOverLiquid(lev, velmag, comp, local) / n_cell * mps_to_cmps;
        velmag_min[lev] = MinOverLiquid(lev, velmag, comp, local) * mps_to_cmps;
        velmag_max[lev] = MaxOverLiquid(lev, velmag, comp, local) * mps_to_cmps;
    }

    // The velocity of the flow at the top level
    const auto& vel = m_leveldata[lev]->velocity;

    // The pressure at the top level
    const auto& pressure = m_leveldata[lev]->pressure;

    // Number of ghost cells in new MultiFab
    const int ngrow {m_leveldata[lev]->pressure.nGrow()};
    // Calculate an adjusted pressure that includes the background pressure
    MultiFab pressure_adj(pressure.boxArray(), pressure.DistributionMap(), nvar, ngrow);
    CalcAdjustedPressure(lev, pressure_adj, pressure, false);

    // Surface area of one cell normal to the x-direction
    const Real normal_surface_cell {CalcCellArea(lev)};

    // MultiFab with normal area in each cell
    MultiFab normal_area_mf;
    normal_area_mf.define(pressure.boxArray(), pressure.DistributionMap(), nvar, pressure.nGrow());
    normal_area_mf.setVal(normal_surface_cell);

    // MultiFab with normal force due to pressure in each cell
    MultiFab normal_force_mf;
    normal_force_mf.define(pressure.boxArray(), pressure.DistributionMap(), nvar, pressure.nGrow());
    normal_force_mf.setVal(normal_surface_cell);
    MultiFab::Multiply(normal_force_mf, pressure_adj, 0, 0, 1, 0);

    // Arguments for SumOverBoundaryCells
    constexpr int dir {0};
    constexpr bool lo_in {true};
    constexpr bool lo_out {false};
    constexpr int comp {0};
    constexpr bool local {false};
    constexpr bool vel_is_nodal {false};
    constexpr bool pressure_is_nodal {true};

    // Conversion factor from meter^3 / sec to cubic centimeters / hour
    constexpr Real m3ps_to_ccph {1.0E6 * 3600.0};

    // Calculate flow of liquid at the inlet and outlet
    m_flow_in = SumOverBoundaryCells(dir, lo_in , vel, comp, vel_is_nodal, local) * normal_surface_cell;
    m_flow_out = SumOverBoundaryCells(dir, lo_out, vel, comp, vel_is_nodal, local) * normal_surface_cell;
    // Convert from m^3/s to cubic centimeters per hour
    Real flow_in_ccph  = m_flow_in * m3ps_to_ccph;
    Real flow_out_ccph = m_flow_out * m3ps_to_ccph;

    // Calculate area of inlet and outlet
    Real normal_surface_in  = SumOverBoundaryCells(dir, lo_in , normal_area_mf, comp, pressure_is_nodal, local);
    Real normal_surface_out = SumOverBoundaryCells(dir, lo_out, normal_area_mf, comp, pressure_is_nodal, local);

    // Calculate normal force due to pressure at inlet and outlet
    Real normal_force_in  = SumOverBoundaryCells(dir, lo_in , normal_force_mf, comp, pressure_is_nodal, local);
    Real normal_force_out = SumOverBoundaryCells(dir, lo_out, normal_force_mf, comp, pressure_is_nodal, local);

    // Calculate pressure at inlet and outlet
    Real pressure_in  = normal_force_in / normal_surface_in;
    Real pressure_out = normal_force_out / normal_surface_out;
    Real pressure_drop = pressure_in - pressure_out;
    
    // Calculate nominal flow time
    const long Ny = Geom(lev).Domain().length(1);
    const long Nz = AMREX_D_PICK(1, 1, Geom(lev).Domain().length(2));
    Real u_sum = SumOverBoundaryCells(dir, lo_out, vel, comp, vel_is_nodal, local);
    Real u_nominal = u_sum / (Ny * Nz);
    m_t_flow = geom[0].ProbLength(0) / u_nominal;

    // Report results
    Print() << format("\n*** Flow Analysis ***\n");
    Print() << format("flow in            = {:12.6f} ml/hour.\n", flow_in_ccph);
    Print() << format("flow out           = {:12.6f} ml/hour.\n", flow_out_ccph);
    Print() << format("pressure at inlet  = {:12.6f} Pa.\n", pressure_in);
    Print() << format("pressure at outlet = {:12.6f} Pa.\n", pressure_out);
    Print() << format("pressure drop      = {:12.6f} Pa.\n", pressure_drop);
    Print() << format("nominal flow time  = {:12.6f} s.\n", m_t_flow);
    Print() << format("velocity magnitude summary by level (cm/s):\n");
    Print() << format("{:3s} : {:12s} : {:12s} : {:12s}\n", "lev", "  mean", "  min", "  max");
    for (int lev=0; lev<= finest_level; ++lev)
    {
        Print() << format("{:3d} : {:12.6f} : {:12.6f} : {:12.6f}\n", 
            lev, velmag_mean[lev], velmag_min[lev], velmag_max[lev]);
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::AnalyzeCurrent() const
{
    #define FUNC_NAME "Rincflo::AnalyzeCurrent"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Conversion factor from amps to milliamps
    constexpr Real amp_to_milliamp {1.0E3};
    // Conversion factor from m^2 to cm^2
    constexpr Real m2_to_cm2 {1.0E4};

    // Note - there are member variables that maintain m_current, m_curr_flux, m_current_mt, m_max_current, etc.
    // However, it's important that we can call the analysis routine when loading from a checkpoint
    // WITHOUT running any reaction time steps. Therefore we calculate everything from scratch here.

    // Calculate the total current and the equivalent current of the net flux of redox species
    Real current_mA = CalcTotalCurrent() * amp_to_milliamp;
    Real curr_flux_mA = CalcNetFluxCurrent() * amp_to_milliamp;
    // The current from mass transport; only applicable in Nernst model
    Real current_mt_mA = 0.0;
    // The relative difference between the two currents; only applicable in Nernst model
    Real curr_diff_rel = 0.0;

    // Deal with the Nernst model
    if (m_reaction_model == ReactionModel::Nernst)
    {
        current_mt_mA = current_mA;
        current_mA = CalcTotalCurrent_BV() * amp_to_milliamp; 
        curr_diff_rel = (current_mt_mA - current_mA) / max(current_mA, current_mt_mA);
    }

    // Calculate the maximum current at 100% utilization
    Real max_current_mA = m_flow_out * m_conc_redox_tot * m_F_const * m_n_electrons * amp_to_milliamp;

    // Calculate the utilization
    Real utilization = current_mA / max_current_mA;

    // Calculate the flux difference relative to the current
    Real flux_diff_rel = (curr_flux_mA - current_mA) / current_mA;

    // Report the results
    Print() << format("\n*** Current Analysis ***\n");
    Print() << format("Total Current =  {:9.6f} mA.\n", current_mA);
    Print() << format("Max Current   =  {:9.6f} mA (at 100% utilization).\n", max_current_mA);
    Print() << format("Utilization   =  {:9.6f}.\n", utilization);
    Print() << format("Net SOC Flux  =  {:9.6f} mA.\n", curr_flux_mA);
    Print() << format("Flux Diff Rel =  {:9.3e}.\n", flux_diff_rel);
    if (m_reaction_model == ReactionModel::Nernst)
    {
        Print() << format("Current (MT)  =  {:9.6f} mA.\n", current_mt_mA);
        Print() << format("Curr Diff Rel =  {:9.3e}.\n", curr_diff_rel);
    }

    // Calculate conserved total redox concentration if verbosity is high
    if (m_verbose > 0)
        {CalcConservedRedoxConc(true);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::AnalyzePotential() const
{
    #define FUNC_NAME "Rincflo::AnalyzePotential"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Only the finest level is used
    const int lev = finest_level;

    // Get the potential difference and overpotential, both in volts
    const auto& ld = *m_leveldata[lev];
    const auto& pot_diff = ld.pot_diff;
    const auto& overpot = ld.overpot;

    // Conversion factor from volts to millivolts
    constexpr Real volt2mv {1.0E3};
    // Applied voltage in millivolts
    Real V_app = m_DVapp * volt2mv;

    // Compute the mean, min, and max pot_diff, in millivolts where applicable; or parrot back assumed value
    Real pot_diff_mean_mV = is_PotComputed() ? AreaWeightedAvg(lev, pot_diff, 0, false) * volt2mv : V_app;
    Real pot_diff_min_mV = is_PotComputed() ? MinOverBoundary(lev, pot_diff, 0) * volt2mv : V_app;
    Real pot_diff_max_mV = is_PotComputed() ? MaxOverBoundary(lev, pot_diff, 0) * volt2mv : V_app;

    // Compute the mean, min, and max overpot, in millivolts; this can always be calculated from SOC
    // even in models like SpecialRedox and Nernst where it is inferred from the SOC.
    Real overpot_mean_mV = AreaWeightedAvg(lev, overpot, 0, false) * volt2mv;
    Real overpot_min_mV = MinOverBoundary(lev, overpot, 0) * volt2mv;
    Real overpot_max_mV = MaxOverBoundary(lev, overpot, 0) * volt2mv;

    Print() << format("\n*** Potential Analysis ***\n");
    // Table headers
    Print() << format("{:10s} : {:10s} : {:10s} : {:10s}\n", "Field", "Mean", "Min", "Max");
    Print() << format("{:10s} : {:10.3f} : {:10.3f} : {:10.3f}\n", 
        "PotDiff", pot_diff_mean_mV, pot_diff_min_mV, pot_diff_max_mV);
    Print() << format("{:10s} : {:10.3f} : {:10.3f} : {:10.3f}\n", 
        "Overpot", overpot_mean_mV, overpot_min_mV, overpot_max_mV);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return;
}
