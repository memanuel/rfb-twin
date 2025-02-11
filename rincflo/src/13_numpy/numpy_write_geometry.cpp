// Local imports
#include <rincflo.H>

// Included names
using cnpy::npy_save, cnpy::make_npy_array, cnpy::npz_save, cnpy::npz_t;
namespace fs = std::filesystem;

// ************************************************************************************************
void Rincflo::WriteNumpyGeometry()
// Write out numpy arrays cell_type.npy and refinement.npy.
{
    #define FUNC_NAME "Rincflo::WriteNumpyGeometry"
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
    // The area of each cell
    vector<NpyArray> area_npy(num_levels);
    // The volume of each cell
    vector<NpyArray> volume_npy(num_levels);

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
        MultiFab cell_type(ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
        // The area MultiFab with a single grid
        MultiFab area     (ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
        // The volume MultiFab with a single grid
        MultiFab volume   (ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);

        // Initialize MFs to NAN signify unfilled cells
        cell_type.setVal(NaN);
        area.setVal(NaN);
        volume.setVal(NaN);

        // Copy cell type data into MultiFab with one grid
        cell_type.ParallelCopy(ld.cell_type, 0, 0, 1);
        area.ParallelCopy     (ld.area,      0, 0, 1);
        volume.ParallelCopy   (ld.volume,    0, 0, 1);

        // Only copy from the single grid MultiFab to the numpy array on the IO rank
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

            // Initialize the cell area
            area_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the cell area
            MultiFab_ToNumpy(area_npy, area, lev);
            // Upsample the cell area when available; area is extensive
            if (lev > 0)
                {NumpyUpsample(area_npy, refinement_npy, lev, true, upsample_all);}

            // Initialize the cell volume
            volume_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the cell volume
            MultiFab_ToNumpy(volume_npy, volume, lev);
            // Upsample the cell volume when available; volume is extensive
            if (lev > 0)
                {NumpyUpsample(volume_npy, refinement_npy, lev, true, upsample_all);}
        }   // if is_io
    }   // for / lev

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

            // Write out cell type to numpy
            fname = format("{:s}/cell_type.npy", out_dir_c);
            cell_type_npy[lev].save<uint8_t>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out refinement to numpy
            fname = format("{:s}/refinement.npy", out_dir_c);
            refinement_npy[lev].save<uint8_t>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

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
        }   // if is_io
    }   // for / lev
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
