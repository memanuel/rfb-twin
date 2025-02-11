// Local imports
#include <rincflo.H>

// Included names
using cnpy::npy_save, cnpy::make_npy_array, cnpy::npz_save, cnpy::npz_t;
namespace fs = std::filesystem;

// ************************************************************************************************
void Rincflo::WriteNumpyAll()
// Write out numpy arrays (.npy files) for analysis in Python
// Uses cnpy library to convert vectors of int or double to Numpy .npy files.
{
    #define FUNC_NAME "Rincflo::WriteNumpyAll"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Status update
    if (m_numpy_verbose > 0) {Print() << format("Writing numpy data...\n");}

    // The MPI process number of this thread
    const int proc {MyProc()};
    // The MPI process number of the IO thread
    const int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    const bool is_io {IOProcessor()};

    // Always update geometry before writing to numpy; need cell type
    CalcGeometry();

    // Write out the scalars and grid geometry
    WriteNumpyScalars();

    // Get the shape of the numpy arrays
    const Rincflo::NumpyShapes ns = GetNumpyShapes();

    // Alias the spatial dimension for legibility
    constexpr size_t dim = SpaceDim;
    // The number of levels we are writing out to numpy
    const int num_levels = max_level+1;
    // Alias the number of species for legibility
    const size_t ns = m_nspec;

    // Preallocate empty vectors for the key metadata at each level: cell_type and refinement
    if (m_numpy_verbose > 1) {Print() << format("Preallocating numpy arrays.\n");}

    // The type of each cell (liquid, boundary, solid)
    vector<NpyArray> cell_type_npy(num_levels);
    // The number of levels of refinement for each cell
    vector<NpyArray> refinement_npy(num_levels);

    // The (i, j, k) index coordinates of each cell
    // vector<NpyArray> idx_npy(num_levels);
    // The (x, y, z) position at the center of each cell
    // vector<NpyArray> pos_npy(num_levels);

    // Preallocate empty NpyArray objects for 
    // The area of each cell
    vector<NpyArray> area_npy(num_levels);
    // The volume of each cell
    vector<NpyArray> volume_npy(num_levels);
    // The fluid velocity (vx, vy, vz) in each cell
    vector<NpyArray> velocity_npy(num_levels);
    // The fluid pressure in each cell - nodal version
    vector<NpyArray> pressure_node_npy(num_levels);
    // The fluid pressure - cell-centered version
    vector<NpyArray> pressure_cell_npy(num_levels);
    // The gradient of the fluid pressure in each cell
    vector<NpyArray> gradp_npy(num_levels);
    // The state of charge in each cell
    vector<NpyArray> soc_npy(num_levels);
    // The species concentrations in each cell
    vector<NpyArray> conc_npy(num_levels);
    // The current in each cell
    vector<NpyArray> current_npy(num_levels);
    // The overpotential
    vector<NpyArray> overpot_npy(num_levels);
    // The electric potential in the liquid
    vector<NpyArray> epotL_npy(num_levels);
    // The electric potential in the solid
    vector<NpyArray> epotS_npy(num_levels);

    // Iterate through the levels and compute the data on each level
    for (int lev=0; lev < num_levels; ++lev)
    {
        // Status
        if (m_numpy_verbose > 1) {Print() << format("Populating data on level {:d}.\n", lev);}

        // AMReX level data
        const auto& ld {*m_leveldata[lev]};

        // BoxArray shared by all these level data
        BoxArray const& ba {grids[lev]};

        // Get a single cell-centered Box for each MultiFab
        // minimalBox() computes a single box to enclose all the boxes
        // enclosedCells() converts it to a cell-centered Box
        const Box bx_1g {ba.minimalBox().enclosedCells()};

        // BoxArray with one grid
        const BoxArray ba_1g(bx_1g);

        // Map all the processes to the IO Processor
        // This way all the levels will be on the same processor for upsampling without loading from disk
        const Vector<int> pmap {proc_io};

        // DistributionMapping with one grid spread to just the IO Processor
        const DistributionMapping dm_1g(pmap);
        // Status
        if (m_numpy_verbose > 2) {Print() << format("Built single grid objects bx_1g, ba_1g, pmap, dm_1g on level {:d}.\n", lev);}

        // FabFactory for single grid MultiFab objects
        #if (AMREX_USE_EB)
        const auto& geom_1g {geom[lev]};
        const auto fact_ptr_1g = makeEBFabFactory(geom_1g, ba_1g, dm_1g, ngrow(), EBSupport::full);
        const auto& fact_1g {*fact_ptr_1g};
        #else
        const auto& fact_1g = new FArrayBoxFactory();
        #endif
        // Status
        if (m_numpy_verbose > 2) {Print() << format("Built EB factory on level {:d}.\n", lev);}

        // When we write to numpy, we don't want to grow the arrays
        constexpr int ng = 0;

        // Build the level data single grid struct for this level
        LevelDataNumpyOutput ld_1g(ba_1g, dm_1g, fact_1g, dim, ns, ng);
        if (m_numpy_verbose > 1) {Print() << format("Built LevelDataNumpyOutput ld_1g on level {:d}.\n", lev);}

        // Initialize MFs to NAN signify unfilled cells
        ld_1g.cell_type.setVal(NaN);
        ld_1g.area.setVal(NaN);
        ld_1g.volume.setVal(NaN);
        ld_1g.velocity.setVal(NaN);
        ld_1g.pressure.setVal(NaN);
        ld_1g.gradp.setVal(NaN);
        ld_1g.soc.setVal(NaN);
        ld_1g.current.setVal(NaN);
        ld_1g.overpot.setVal(NaN);
        ld_1g.epotL.setVal(NaN);
        ld_1g.epotS.setVal(NaN);

        // Perform a parallel copy from each working MultiFab to the single grid MultiFab on the communicator rank.
        // arguments to ParallelCopy: src, src_comp, dest_comp, num_comp
        // OR: src, src_comp, dest_comp, num_comp, src_nghost, dst_nghost

        // Copy cell type data into MultiFab with one grid
        ld_1g.cell_type.ParallelCopy(ld.cell_type, 0, 0, 1);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of cell type.\n");}

        // Copy cell area into MultiFab with one grid
        ld_1g.area.ParallelCopy(ld.area, 0, 0, 1);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of cell area.\n");}

        // Copy cell volume into MultiFab with one grid
        ld_1g.volume.ParallelCopy(ld.volume, 0, 0, 1);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of cell volume.\n");}

        // Copy velocity data into MultiFab with one grid
        ld_1g.velocity.ParallelCopy(ld.velocity, 0, 0, dim);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of velocity.\n");}

        // Compute adjusted pressure that includes the background pressure from the inlet / outlet BCs
        MultiFab pressure_adj(ld.pressure.boxArray(), ld.pressure.DistributionMap(), 1, 0);
        CalcAdjustedPressure(lev, pressure_adj, ld.pressure, true);
        // Copy pressure data into MultiFab with one grid
        ld_1g.pressure.ParallelCopy(pressure_adj, 0, 0, 1);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of adjusted pressure.\n");}

        // Compute adjusted gradp that includes the background pressure from the inlet / outlet BCs
        MultiFab gradp_adj(ld.gradp.boxArray(), ld.gradp.DistributionMap(), SpaceDim, 0);
        CalcAdjustedGradp(lev, gradp_adj, ld.gradp);
        // Copy adjusted pressure gradient data into MultiFab with one grid; no ghost
        ld_1g.gradp.ParallelCopy(gradp_adj, 0, 0, dim);
        if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of gradp.\n");}

        // Only copy reaction data if we've actually simulated the reaction
        if (is_Reaction())
        {
            // Need to update derived reaction properties including soc before copying output
            CalcDerivedReact();
            if (m_numpy_verbose > 1) {Print() << format("Completed CalcDerivedReact().\n");}

            // Copy conc data into MultiFab with one grid
            ld_1g.conc.ParallelCopy(ld.conc, 0, 0, ns);
            if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of conc.\n");}

            // Copy soc data into MultiFab with one grid
            ld_1g.soc.ParallelCopy(ld.soc, 0, 0, 1);
            if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of soc.\n");}

            // Copy current data into MultiFab with one grid
            ld_1g.current.ParallelCopy(ld.current, 0, 0, 1);
            if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of current.\n");}

            // Copy overpotential into MultiFab with one grid; no ghost
            ld_1g.overpot.ParallelCopy(ld.overpot, 0, 0, 1);
            if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of overpot.\n");}

            // Copy epotL data into MultiFab with one grid; has ghost.  Only need this when computing potential
            if (is_PotComputed())
            {
                ld_1g.epotL.ParallelCopy(ld.epotL, 0, 0, 1);
                if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of epotL.\n");}

                ld_1g.epotS.ParallelCopy(ld.epotS, 0, 0, 1);
                if (m_numpy_verbose > 2) {Print() << format("Completed parallel copy of epotS.\n");}
            }
        }   // if is_Reaction

        // Status
        if (m_numpy_verbose > 1) {Print() << format("Loaded single grid data on level {:d}.\n", lev);}

        // Only run this block on IO rank
        if (is_io)
        {
            // The cell types on this level; if no embedded boundaries, all the cells have type 0
            cell_type_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // The number of levels of refinement
            refinement_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // Get the cell type and refinement for this level
            MultiFab_CellTypeToNumpy(cell_type_npy, refinement_npy, ld_1g.cell_type, lev);
            // When we upsample, we don't want to upsample everywhere, just where the refinement level requires it
            bool upsample_all = false;

            // Initialize the cell area
            area_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the cell area
            MultiFab_ToNumpy(area_npy, ld_1g.area, lev);
            if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(area).\n");}
            // Upsample the cell area when available; area is extensive
            if (lev > 0)
                {NumpyUpsample(area_npy, refinement_npy, lev, true, upsample_all);}

            // Initialize the cell volume
            volume_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the cell volume
            MultiFab_ToNumpy(volume_npy, ld_1g.volume, lev);
            if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(volume).\n");}
            // Upsample the cell volume when available; volume is extensive
            if (lev > 0)
                {NumpyUpsample(volume_npy, refinement_npy, lev, true, upsample_all);}

            // Mask for liquid + boundary
            uint8_t mask_lb = static_cast<uint8_t>(CellType::liquid) + static_cast<uint8_t>(CellType::boundary);

            // Initialize vectors for index, position
            // idx_npy[lev] = make_npy_array<int32_t>(shape_npy_vector[lev]);
            // pos_npy[lev] = make_npy_array<Real>(shape_npy_vector[lev]);
            // Extract the index and positition vectors
            // MultiFab_IndexToNumpy(idx_npy[lev], pos_npy[lev], ld_1g.cell_type, lev);

            // Initialize velocity
            velocity_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
            // Extract the velocity
            MultiFab_ToNumpy(velocity_npy, ld_1g.velocity, lev);
            if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(velocity).\n");}
            // Upsample the velocity from the coarse level when available; velocity is intensive
            if (lev > 0)
                {NumpyUpsample(velocity_npy, refinement_npy, lev, false, upsample_all);}
            // Filter the velocity to only include liquid and boundary cells
            NumpyFilter(velocity_npy[lev], cell_type_npy[lev], mask_lb);

            // Initialize pressure (nodal)
            pressure_node_npy[lev] = make_npy_array<Real>(ns.shape_npy_pressure[lev]);
            // Extract the pressure
            MultiFab_ToNumpy(pressure_node_npy, ld_1g.pressure, lev);
            if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(pressure) (nodal layout).\n");}

            // Status - flow variables
            if (m_numpy_verbose > 2) {Print() << format("Copied and upsampled area, volume, velocity, nodal pressure.\n");}

            // Initialize pressure (cell centered)
            pressure_cell_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Convert pressure from nodal to cell-centered
            NumpyNodeToCell(pressure_node_npy[lev], pressure_cell_npy[lev]);
            if (m_numpy_verbose > 2) {Print() << format("Converted pressure from nodal to cell-centered.\n");}
            // Upsample the pressure when available; pressure is intensive
            if (lev > 0)
                {NumpyUpsample(pressure_cell_npy, refinement_npy, lev, false, upsample_all);}
            // Filter pressure to only include liquid and boundary cells
            NumpyFilter(pressure_cell_npy[lev], cell_type_npy[lev], mask_lb);
            if (m_numpy_verbose > 2) {Print() << format("Filtered pressure.\n");}

            // Initialize gradp
            gradp_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
            // Extract the pressure gradient
            MultiFab_ToNumpy(gradp_npy, ld_1g.gradp, lev);
            if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(gradp).\n");}
            // Upsample the pressure gradient from the coarse level when available; gradp is intensive
            if (lev > 0)
                {NumpyUpsample(gradp_npy, refinement_npy, lev, false, upsample_all);}
            // Filter the pressure gradient to only include liquid and boundary cells
            NumpyFilter(gradp_npy[lev], cell_type_npy[lev], mask_lb);
            if (m_numpy_verbose > 2) {Print() << format("Filtered pressure gradient.\n");}

            // Only write out the reaction quantities if we've actually simulated the reaction
            if (is_Reaction())
            {
                // Initialize conc
                conc_npy[lev] = make_npy_array<Real>(ns.shape_npy_conc[lev]);
                // Extract the conc
                MultiFab_ToNumpy(conc_npy, ld_1g.conc, lev);
                if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(conc).\n");}
                // Upsample the conc when available; conc is intensive
                if (lev > 0)
                    {NumpyUpsample(conc_npy, refinement_npy, lev, false, upsample_all);}
                // Filter the conc to only include liquid and boundary cells
                NumpyFilter(conc_npy[lev], cell_type_npy[lev], mask_lb);
                if (m_numpy_verbose > 2) {Print() << format("Filtered conc.\n");}

                // Initialize SOC
                soc_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                // Extract the SOC
                MultiFab_ToNumpy(soc_npy, ld_1g.soc, lev);
                if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(soc).\n");}
                // Upsample the SOC when available; SOC is intensive
                if (lev > 0)
                    {NumpyUpsample(soc_npy, refinement_npy, lev, false, upsample_all);}
                // Filter SOC to only include liquid and boundary cells
                NumpyFilter(soc_npy[lev], cell_type_npy[lev], mask_lb);
                if (m_numpy_verbose > 2) {Print() << format("Filtered soc.\n");}

                // Mask for boundary only
                uint8_t mask_b = static_cast<uint8_t>(CellType::boundary);

                // Initialize current
                current_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                // Extract the current
                MultiFab_ToNumpy(current_npy, ld_1g.current, lev);
                if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(current).\n");}
                // Upsample the current when available; current is extensive
                if (lev > 0)
                    {NumpyUpsample(current_npy, refinement_npy, lev, true, upsample_all);}
                // Filter current to only include boundary cells
                NumpyFilter(current_npy[lev], cell_type_npy[lev], mask_b);
                if (m_numpy_verbose > 2) {Print() << format("Filtered current.\n");}

                // Initialize overpot
                overpot_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                // Extract the overpotential
                MultiFab_ToNumpy(overpot_npy, ld_1g.overpot, lev);
                if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(overpot).\n");}
                // Upsample the overpotential when available; overpotential is intensive
                if (lev > 0)
                    {NumpyUpsample(overpot_npy, refinement_npy, lev, false, upsample_all);}
                // Filter overpotential to include only boundary cells
                NumpyFilter(overpot_npy[lev], cell_type_npy[lev], mask_b);
                if (m_numpy_verbose > 2) {Print() << format("Filtered overpot.\n");}

                // Handle the electric potential when applicable
                if (is_PotComputed())
                {
                    // Initialize electric potential in liquid
                    epotL_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                    MultiFab_ToNumpy(epotL_npy, ld_1g.epotL, lev);
                    if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(epotL).\n");}
                    if (lev > 0)
                        {NumpyUpsample(epotL_npy, refinement_npy, lev, false, upsample_all);}
                    NumpyFilter(epotL_npy[lev], cell_type_npy[lev], mask_b);
                    if (m_numpy_verbose > 2) {Print() << format("Filtered epotL.\n");}

                    // Initialize electric potential in solid
                    epotS_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                    MultiFab_ToNumpy(epotS_npy, ld_1g.epotS, lev);
                    if (m_numpy_verbose > 2) {Print() << format("Completed MultiFab_ToNumpy(epotS).\n");}
                    if (lev > 0)
                        {NumpyUpsample(epotS_npy, refinement_npy, lev, false, upsample_all);}
                    NumpyFilter(epotS_npy[lev], cell_type_npy[lev], mask_b);
                    if (m_numpy_verbose > 2) {Print() << format("Filtered epotS.\n");}
                }   // if is_PotComputed
            } // if is_Reaction
        } // if is_io
    }   // for / lev

    // Status
    if (m_numpy_verbose > 0) {Print() << format("Completed copying and upsampling of data to numpy arrays.\n");}

    // Iterate through the levels, convert to friendly units, and write out the data to numpy
    // attribute    unit            conversion_factor
    // length:      cm              1.0E2
    // area:        cm^2            1.0E4
    // volume:      cm^3            1.0E6
    // velocity:    cm/sec          1.0E2
    // pressure:    pascals         1.0E0
    // current:     milliamps       1.0E3
    // curr_dens:   milliamps/cm^2  1.0E-1
    // source_mol:  moles/sec       1.0E0
    // epotL:       millivolts      1.0E3
    // epotS:       millivolts      1.0E3
    // overpot:     millivolts      1.0E3

    // Only run this block on IO rank
    if (is_io)
    {
        // Iterate through the levels and write to numpy
        for (int lev=0; lev < num_levels; ++lev)
        {
            // The directory to write checkpoints, e.g. numpy/rstep_1234567
            string out_dir_top = NumpyOutputDir();
            const char* out_dir_top_c = out_dir_top.c_str();
        
            // Directory where numpy data files for this level will be saved
            const string out_dir = format("{:s}/L{:d}", out_dir_top_c, lev);
            const char* out_dir_c = out_dir.c_str();

            // Create directory for numpy files if missing
            fs::create_directory(out_dir);

            // String to hold the filename about to be saved
            string fname = "";

            // Status
            if (m_numpy_verbose > 0) {Print() << format("Writing out data to {:s}.\n", out_dir_c);}

            // Write out index to numpy
            // idx_npy[lev].save<int32_t>.save(format("{:s}/idx.npy", out_dir));

            // Write out cell type to numpy
            fname = format("{:s}/cell_type.npy", out_dir_c);
            cell_type_npy[lev].save<uint8_t>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out refinement to numpy
            fname = format("{:s}/refinement.npy", out_dir_c);
            refinement_npy[lev].save<uint8_t>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved refinement.npy");}

            // Convert position to cm; conversion factor is 1.0E2
            // pos_npy[lev].rescale(1.0E2);
            // Write out position to numpy
            // fname = format("{:s}/pos.npy", out_dir);
            // pos_npy[lev].save<Real>(fname);

            // Convert area to cm^2; conversion factor is 1.0E4
            area_npy[lev].rescale<Real>(1.0E4);
            // Write out area to numpy
            fname = format("{:s}/area.npy", out_dir_c);
            area_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Convert volume to cm^3; conversion factor is 1.0E6
            volume_npy[lev].rescale<Real>(1.0E6);
            // Write out volume to numpy
            fname = format("{:s}/volume.npy", out_dir_c);
            volume_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Convert velocity to cm/sec; conversion factor is 1.0E2
            velocity_npy[lev].rescale<Real>(1.0E2);
            // Write out velocity to numpy
            fname = format("{:s}/velocity.npy", out_dir_c);
            velocity_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out nodal pressure to numpy; units are Pascals, don't rescale
            fname = format("{:s}/pressure_node.npy", out_dir_c);
            pressure_node_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out cell-centered pressure to numpy; units are Pascals, don't rescale
            fname = format("{:s}/pressure_cell.npy", out_dir_c);
            pressure_cell_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out pressure gradient to numpy; units are Pascals / meter = N / m^3, don't rescale
            fname = format("{:s}/gradp.npy", out_dir_c);
            gradp_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Only save files related to the reaction when running the reaction simulation.
            if (is_Reaction())
            {
                // Write out conc to numpy; units are in millimolar, don't rescale
                fname = format("{:s}/conc.npy", out_dir_c);
                conc_npy[lev].save<Real>(fname);
                if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

                // Write out state of charge to numpy; units are dimensionless, don't rescale (obviously)
                fname = format("{:s}/soc.npy", out_dir_c);
                soc_npy[lev].save<Real>(fname);
                if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

                // Convert current from amps to milliamps; conversion factor is 1.0E3
                current_npy[lev].rescale(1.0E3);
                // Write out current to numpy
                fname = format("{:s}/current.npy", out_dir_c);
                current_npy[lev].save<Real>(fname);
                if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

                // Convert overpotential from volts to millivolts; conversion factor is 1.0E3
                overpot_npy[lev].rescale(1.0E3);
                // Write out the overpotential
                fname = format("{:s}/overpot.npy", out_dir_c);
                overpot_npy[lev].save<Real>(fname);
                if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

                // Write out the electric potential if applicable
                if (is_PotComputed())
                {
                    // Save electric potential in liquid; convert to millivolts
                    epotL_npy[lev].rescale(1.0E3);
                    fname = format("{:s}/epotL.npy", out_dir_c);
                    epotL_npy[lev].save<Real>(fname);
                    if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

                    // Save electric potential in solid; convert to millivolts
                    epotS_npy[lev].rescale(1.0E3);
                    fname = format("{:s}/epotS.npy", out_dir_c);
                    epotS_npy[lev].save<Real>(fname);
                    if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}
                }   // if is_PotComputed
            }   // if is_Reaction
        }   // for / lev
    }   // if is_io
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
