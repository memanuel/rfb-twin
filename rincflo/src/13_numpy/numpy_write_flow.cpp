// Local imports
#include <rincflo.H>

// Included names
using cnpy::npy_save, cnpy::make_npy_array, cnpy::npz_save, cnpy::npz_t;
namespace fs = std::filesystem;

// ************************************************************************************************
void Rincflo::WriteNumpyVelocity()
// Write out numpy array velocity.npy
{
    #define FUNC_NAME "Rincflo::WriteNumpyVelocity"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The MPI process number of this thread
    const int proc {MyProc()};
    // The MPI process number of the IO thread
    const int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    const bool is_io {IOProcessor()};

    // Get the shape of the numpy arrays
    const Rincflo::NumpyShapes ns = GetNumpyShapes();
    // Alias the spatial dimension for legibility
    constexpr size_t dim = SpaceDim;
    // The number of levels we are writing out to numpy
    const int num_levels = max_level+1;
    // Alias the number of species for legibility
    const size_t nsp = m_nspec;

    // The type of each cell (liquid, boundary, solid)
    vector<NpyArray> cell_type_npy(num_levels);
    // The number of levels of refinement for each cell
    vector<NpyArray> refinement_npy(num_levels);
    // The fluid velocity (vx, vy, vz) in each cell
    vector<NpyArray> velocity_npy(num_levels);

    // Iterate through the levels and compute the data on each level
    for (int lev=0; lev < num_levels; ++lev)
    {
        // AMReX level data
        const auto& ld {*m_leveldata[lev]};
        // BoxArray shared by all these level data
        BoxArray const& ba {grids[lev]};
        // Get a single cell-centered Box for each MultiFab
        const Box bx_1g {ba.minimalBox().enclosedCells()};
        // BoxArray with one grid
        const BoxArray ba_1g(bx_1g);
        // Map all the processes to the IO Processor
        const Vector<int> pmap {proc_io};
        // DistributionMapping with one grid spread to just the IO Processor
        const DistributionMapping dm_1g(pmap);

        // FabFactory for single grid MultiFab objects
        #if (AMREX_USE_EB)
        const auto& geom_1g {geom[lev]};
        const auto fact_ptr_1g = makeEBFabFactory(geom_1g, ba_1g, dm_1g, ngrow(), EBSupport::full);
        const auto& fact_1g {*fact_ptr_1g};
        #else
        const auto& fact_1g = new FArrayBoxFactory();
        #endif

        // When we write to numpy, we don't want to grow the arrays
        constexpr int ng = 0;
        // Build the cell_type MultiFab with a single grid
        MultiFab cell_type  (ba_1g, dm_1g, 1  , ng, MFInfo(), fact_1g);
        // The velocity MultiFab with a single grid
        MultiFab velocity   (ba_1g, dm_1g, dim, ng, MFInfo(), fact_1g);

        // Initialize MFs to NAN signify unfilled cells
        cell_type.setVal(NaN);
        velocity.setVal(NaN);

        // Copy cell type data into MultiFab with one grid
        cell_type.ParallelCopy  (ld.cell_type, 0, 0, 1  );
        // Copy velocity data into MultiFab with one grid
        velocity.ParallelCopy   (ld.velocity,  0, 0, dim);

        // Only run this block on IO rank
        if (is_io)
        {
            // The cell types on this level; if no embedded boundaries, all the cells have type 0
            cell_type_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // The number of levels of refinement
            refinement_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // Get the cell type and refinement for this level
            MultiFab_CellTypeToNumpy(cell_type_npy, refinement_npy, cell_type, lev);
            // When we upsample, we don't want to upsample everywhere, just where the refinement level requires it
            bool upsample_all = false;

            // Mask for liquid + boundary
            uint8_t mask_lb = static_cast<uint8_t>(CellType::liquid) + static_cast<uint8_t>(CellType::boundary);

            // Initialize velocity
            velocity_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
            // Extract the velocity
            MultiFab_ToNumpy(velocity_npy, velocity, lev);
            // Upsample the velocity from the coarse level when available; velocity is intensive
            if (lev > 0)
                {NumpyUpsample(velocity_npy, refinement_npy, lev, false, upsample_all);}
            // Filter the velocity to only include liquid and boundary cells
            NumpyFilter(velocity_npy[lev], cell_type_npy[lev], mask_lb);
        } // if is_io
    }   // for / lev

    // Iterate through the levels, convert to friendly units, and write out the data to numpy
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

           // Convert velocity to cm/sec; conversion factor is 1.0E2
            velocity_npy[lev].rescale<Real>(1.0E2);

            // Write out velocity to numpy
            fname = format("{:s}/velocity.npy", out_dir_c);
            velocity_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}
        } // for / lev
    } // if is_io
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::WriteNumpyPressure()
// Write out numpy array velocity.npy
{
    #define FUNC_NAME "Rincflo::WriteNumpyPressure"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The MPI process number of this thread
    const int proc {MyProc()};
    // The MPI process number of the IO thread
    const int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    const bool is_io {IOProcessor()};

    // Get the shape of the numpy arrays
    const Rincflo::NumpyShapes ns = GetNumpyShapes();
    // Alias the spatial dimension for legibility
    constexpr size_t dim = SpaceDim;
    // The number of levels we are writing out to numpy
    const int num_levels = max_level+1;
    // Alias the number of species for legibility
    const size_t nsp = m_nspec;

    // The type of each cell (liquid, boundary, solid)
    vector<NpyArray> cell_type_npy(num_levels);
    // The number of levels of refinement for each cell
    vector<NpyArray> refinement_npy(num_levels);
    // The fluid pressure in each cell - nodal version
    vector<NpyArray> pressure_node_npy(num_levels);
    // The fluid pressure - cell-centered version
    vector<NpyArray> pressure_cell_npy(num_levels);
    // The gradient of the fluid pressure in each cell
    vector<NpyArray> gradp_npy(num_levels);

    // Iterate through the levels and compute the data on each level
    for (int lev=0; lev < num_levels; ++lev)
    {
        // AMReX level data
        const auto& ld {*m_leveldata[lev]};
        // BoxArray shared by all these level data
        BoxArray const& ba {grids[lev]};
        // Get a single cell-centered Box for each MultiFab
        const Box bx_1g {ba.minimalBox().enclosedCells()};
        // BoxArray with one grid
        const BoxArray ba_1g(bx_1g);
        // Map all the processes to the IO Processor
        const Vector<int> pmap {proc_io};
        // DistributionMapping with one grid spread to just the IO Processor
        const DistributionMapping dm_1g(pmap);

        // FabFactory for single grid MultiFab objects
        #if (AMREX_USE_EB)
        const auto& geom_1g {geom[lev]};
        const auto fact_ptr_1g = makeEBFabFactory(geom_1g, ba_1g, dm_1g, ngrow(), EBSupport::full);
        const auto& fact_1g {*fact_ptr_1g};
        #else
        const auto& fact_1g = new FArrayBoxFactory();
        #endif

        // When we write to numpy, we don't want to grow the arrays
        constexpr int ng = 0;
        // The cell_type MultiFab with a single grid
        MultiFab cell_type  (ba_1g, dm_1g, 1  , ng, MFInfo(), fact_1g);
        // The pressure MultiFab with a single grid
        MultiFab pressure   (convert(ba_1g, IntVect::TheNodeVector()), 
                                     dm_1g, 1 , ng, MFInfo(), fact_1g);
        // The gradp MultiFab (pressure gradient) with a single grid
        MultiFab gradp      (ba_1g, dm_1g, dim, ng, MFInfo(), fact_1g);

        // Initialize MFs to NAN signify unfilled cells
        cell_type.setVal(NaN);
        pressure.setVal(NaN);
        gradp.setVal(NaN);

        // Copy cell type data into MultiFab with one grid
        cell_type.ParallelCopy  (ld.cell_type, 0, 0, 1  );

        // Compute adjusted pressure that includes the background pressure from the inlet / outlet BCs
        MultiFab pressure_adj(ld.pressure.boxArray(), ld.pressure.DistributionMap(), 1, 0);
        CalcAdjustedPressure(lev, pressure_adj, ld.pressure, true);
        // Copy adjusted pressure data into MultiFab with one grid
        pressure.ParallelCopy(pressure_adj, 0, 0, 1);

        // Compute adjusted gradp that includes the background pressure from the inlet / outlet BCs
        MultiFab gradp_adj(ld.gradp.boxArray(), ld.gradp.DistributionMap(), SpaceDim, 0);
        CalcAdjustedGradp(lev, gradp_adj, ld.gradp);
        // Copy adjusted pressure gradient data into MultiFab with one grid; no ghost
        gradp.ParallelCopy(gradp_adj, 0, 0, dim);

        // Only run this block on IO rank
        if (is_io)
        {
            // The cell types on this level; if no embedded boundaries, all the cells have type 0
            cell_type_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // The number of levels of refinement
            refinement_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // Get the cell type and refinement for this level
            MultiFab_CellTypeToNumpy(cell_type_npy, refinement_npy, cell_type, lev);
            // When we upsample, we don't want to upsample everywhere, just where the refinement level requires it
            bool upsample_all = false;

            // Mask for liquid + boundary
            uint8_t mask_lb = static_cast<uint8_t>(CellType::liquid) + static_cast<uint8_t>(CellType::boundary);

            // Initialize pressure (nodal)
            pressure_node_npy[lev] = make_npy_array<Real>(ns.shape_npy_pressure[lev]);
            // Extract the pressure
            MultiFab_ToNumpy(pressure_node_npy, pressure, lev);

            // Initialize pressure (cell centered)
            pressure_cell_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Convert pressure from nodal to cell-centered
            NumpyNodeToCell(pressure_node_npy[lev], pressure_cell_npy[lev]);
            // Upsample the pressure when available; pressure is intensive
            if (lev > 0)
                {NumpyUpsample(pressure_cell_npy, refinement_npy, lev, false, upsample_all);}
            // Filter pressure to only include liquid and boundary cells
            NumpyFilter(pressure_cell_npy[lev], cell_type_npy[lev], mask_lb);

            // Initialize gradp
            gradp_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
            // Extract the pressure gradient
            MultiFab_ToNumpy(gradp_npy, gradp, lev);
            // Upsample the pressure gradient from the coarse level when available; gradp is intensive
            if (lev > 0)
                {NumpyUpsample(gradp_npy, refinement_npy, lev, false, upsample_all);}
            // Filter the pressure gradient to only include liquid and boundary cells
            NumpyFilter(gradp_npy[lev], cell_type_npy[lev], mask_lb);
        } // if is_io
    }   // for / lev

    // Iterate through the levels, convert to friendly units, and write out the data to numpy
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
        } // for / lev
    } // if is_io
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
