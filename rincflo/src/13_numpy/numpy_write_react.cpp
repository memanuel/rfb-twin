// Local imports
#include <rincflo.H>

// Included names
using cnpy::npy_save, cnpy::make_npy_array, cnpy::npz_save, cnpy::npz_t;
namespace fs = std::filesystem;

// ************************************************************************************************
void Rincflo::WriteNumpyConc()
// Write out numpy arrays conc.npy, soc.npy
{
    #define FUNC_NAME "Rincflo::WriteNumpyConc"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Return early if not simulating the reaction
    if (!is_Reaction())
    {
        Print() << "Warning - called WriteNumpyConc() but not simulating rection. Returning early.\n";
        return;
    }

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
    // The species concentrations in each cell
    vector<NpyArray> conc_npy(num_levels);
    // The state of charge in each cell
    vector<NpyArray> soc_npy(num_levels);

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
        // The conc MultiFab with a single grid
        MultiFab conc     (ba_1g, dm_1g, nsp , ng, MFInfo(), fact_1g);
        // The state of charge MultiFab with a single grid
        MultiFab soc        (ba_1g, dm_1g, 1  , ng, MFInfo(), fact_1g);

        // Initialize MFs to NAN signify unfilled cells
        cell_type.setVal(NaN);
        conc.setVal(NaN);
        soc.setVal(NaN);

        // Copy cell type data into MultiFab with one grid
        cell_type.ParallelCopy  (ld.cell_type, 0, 0, 1  );
        // Copy conc data into MultiFab with one grid
        conc.ParallelCopy(ld.conc, 0, 0, nsp);
        // Copy soc data into MultiFab with one grid
        soc.ParallelCopy(ld.soc, 0, 0, 1);

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

            // Initialize conc
            conc_npy[lev] = make_npy_array<Real>(ns.shape_npy_conc[lev]);
            // Extract the conc
            MultiFab_ToNumpy(conc_npy, conc, lev);
            // Upsample the conc when available; conc is intensive
            if (lev > 0)
                {NumpyUpsample(conc_npy, refinement_npy, lev, false, upsample_all);}
            // Filter the conc to only include liquid and boundary cells
            NumpyFilter(conc_npy[lev], cell_type_npy[lev], mask_lb);

            // Initialize SOC
            soc_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the SOC
            MultiFab_ToNumpy(soc_npy, soc, lev);
            // Upsample the SOC when available; SOC is intensive
            if (lev > 0)
                {NumpyUpsample(soc_npy, refinement_npy, lev, false, upsample_all);}
            // Filter SOC to only include liquid and boundary cells
            NumpyFilter(soc_npy[lev], cell_type_npy[lev], mask_lb);
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

            // Write out conc to numpy; units are in millimolar, don't rescale
            fname = format("{:s}/conc.npy", out_dir_c);
            conc_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}

            // Write out state of charge to numpy; units are dimensionless, don't rescale (obviously)
            fname = format("{:s}/soc.npy", out_dir_c);
            soc_npy[lev].save<Real>(fname);
            if (m_numpy_verbose > 0) {Print() << format("Saved {:s}.\n", fname.c_str());}
        } // for / lev
    } // if is_io
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::WriteNumpyCurrent()
// Write out numpy arrays current.npy, overpot.npy, epotL.npy, epotS.npy
{
    #define FUNC_NAME "Rincflo::WriteNumpyCurrent"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Return early if not simulating the reaction
    if (!is_Reaction())
    {
        Print() << "Warning - called WriteNumpyCurrent() but not simulating rection. Returning early.\n";
        return;
    }

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
        MultiFab cell_type  (ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
        // The current MultiFab with a single grid
        MultiFab current    (ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
        // The overpot MultiFab with a single grid
        MultiFab overpot    (ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
        // Placeholder for the epotL MultiFab with a single grid; populate it when applicable
        MultiFab epotL;
        // Placeholder for the epotS MultiFab with a single grid; populate it when applicable
        MultiFab epotS;

        // Initialize MFs to NAN signify unfilled cells
        cell_type.setVal(NaN);
        current.setVal(NaN);
        overpot.setVal(NaN);

        // Copy cell type data into MultiFab with one grid
        cell_type.ParallelCopy  (ld.cell_type, 0, 0, 1  );
        // Copy current data into MultiFab with one grid
        current.ParallelCopy(ld.current, 0, 0, 1);
        // Copy overpotential into MultiFab with one grid; no ghost
        overpot.ParallelCopy(ld.overpot, 0, 0, 1);

        // Initialize epotL and epotS when potential is calculated
        if (is_PotComputed())
        {
            epotL.define(ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
            epotS.define(ba_1g, dm_1g, 1, ng, MFInfo(), fact_1g);
            epotL.setVal(NaN);
            epotS.setVal(NaN);
            epotL.ParallelCopy(ld.epotL, 0, 0, 1);
            epotS.ParallelCopy(ld.epotS, 0, 0, 1);
        }

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

            // Mask for boundary only
            uint8_t mask_b = static_cast<uint8_t>(CellType::boundary);

            // Initialize current
            current_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the current
            MultiFab_ToNumpy(current_npy, current, lev);
            // Upsample the current when available; current is extensive
            if (lev > 0)
                {NumpyUpsample(current_npy, refinement_npy, lev, true, upsample_all);}
            // Filter current to only include boundary cells
            NumpyFilter(current_npy[lev], cell_type_npy[lev], mask_b);

            // Initialize overpot
            overpot_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
            // Extract the overpotential
            MultiFab_ToNumpy(overpot_npy, overpot, lev);
            // Upsample the overpotential when available; overpotential is intensive
            if (lev > 0)
                {NumpyUpsample(overpot_npy, refinement_npy, lev, false, upsample_all);}
            // Filter overpotential to include only boundary cells
            NumpyFilter(overpot_npy[lev], cell_type_npy[lev], mask_b);

            // Handle the electric potential when applicable
            if (is_PotComputed())
            {
                // Initialize electric potential in liquid
                epotL_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                MultiFab_ToNumpy(epotL_npy, epotL, lev);
                if (lev > 0)
                    {NumpyUpsample(epotL_npy, refinement_npy, lev, false, upsample_all);}

                // Initialize electric potential in solid
                epotS_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                MultiFab_ToNumpy(epotS_npy, epotS, lev);
                if (lev > 0)
                    {NumpyUpsample(epotS_npy, refinement_npy, lev, false, upsample_all);}
            }
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
        } // for / lev
    } // if is_io
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
