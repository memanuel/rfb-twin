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
    if (m_numpy_verbose > 0) {Print() << format("Writing numpy data for all arrays...\n");}

    // The MPI process number of this thread
    int proc {MyProc()};
    // The MPI process number of the IO thread
    int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    bool is_io {IOProcessor()};

    // Write out the scalars and the domain geometry
    WriteNumpyScalars();
    
    // Geometry quantities
    // Update geometry to get cell type
    CalcGeometry();
    // Write out the cell type and refinement level
    WriteNumpyGeometry();

    // Flow quantities
    // Write out the velocity
    WriteNumpyVelocity();
    // Write out the pressure (nodal and cell centerd) and pressure gradient
    WriteNumpyPressure();

    // Reaction quantities - only if we are simulating the reaction
    if (is_Reaction())
    {
        // Need to update derived reaction quantities (e.g. SOC) before writing out
        CalcDerivedReact();
        // Write out the conc concentrations and state of charge
        WriteNumpyConc();
        // Write out the current
        WriteNumpyCurrent();
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}    

// ************************************************************************************************
void Rincflo::WriteNumpyScalars()
// Write out scalars used in loading numpy data files as well as the grid geometry.
{
    #define FUNC_NAME "Rincflo::WriteNumpyScalars"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The MPI process number of this thread
    int proc {MyProc()};
    // The MPI process number of the IO thread
    int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    bool is_io {IOProcessor()};

    // These tasks only happen on the IO processor
    if (is_io)
    {
        // The output directory for this simulation
        const string dir_top = NumpyOutputDir();
        const char* dir_top_c = dir_top.c_str();

        // Status
        if (m_numpy_verbose > 0) {Print() << format("Saving scalars.npz to {:s}\n", dir_top_c);}
        Print();

        // Create output directories if missing
        fs::create_directory("numpy");
        fs::create_directory(dir_top);

        // Write out the scalar data
        string zipname = format("{:s}/scalars.npz", dir_top_c);
        vector<size_t> scalar_shape {1};
        // First entry is in write mode "w" to create a new file scalars.npz, overwriting the old one
        npz_save<int>(zipname, "nstep", &m_nstep, scalar_shape, "w");
        // Remaining entries in append mode "a" to add data elements to scalars.npz file
        npz_save<int>(zipname, "rstep", &m_rstep, scalar_shape, "a");
        npz_save<Real>(zipname, "cur_time", &m_cur_time, scalar_shape, "a");
        npz_save<Real>(zipname, "dt", &m_dt, scalar_shape, "a");
        npz_save<Real>(zipname, "prev_dt", &m_prev_dt, scalar_shape, "a");
        npz_save<Real>(zipname, "prev_prev_dt", &m_prev_prev_dt, scalar_shape, "a");
    }

    // Get the shape of the numpy arrays
    const Rincflo::NumpyShapes ns = GetNumpyShapes();

    // The number of levels we are writing out to numpy
    const int num_levels = max_level+1;
    
    // Write out the grid shape and domain lo / hi bounds on the MPI communicator rank
    if (is_io)
    {
        // The directory to write checkpoints, e.g. numpy/rstep_1234567
        string out_dir_top = NumpyOutputDir();
        const char* out_dir_top_c = out_dir_top.c_str();
        fs::create_directory(out_dir_top);
        
        // Status
        if (m_numpy_verbose > 1) {Print() << format("Saving grid_shape, domain_lo, domain_hi.\n");}

        // Save the grid shape and domain bounds in the top level directory
        cnpy::npy_save(format("{:s}/grid_shape.npy", out_dir_top_c), ns.grid_shape.data(), ns.shape_of_grid_metadata);
        cnpy::npy_save(format("{:s}/domain_lo.npy",  out_dir_top_c), ns.domain_lo.data(),  ns.shape_of_domain);
        cnpy::npy_save(format("{:s}/domain_hi.npy",  out_dir_top_c), ns.domain_hi.data(),  ns.shape_of_domain);

        for (int lev=0; lev < num_levels; ++lev)
        {
            // Directory where numpy data files for this level will be saved
            string out_dir = format("{:s}/L{:d}", out_dir_top_c, lev);
                
            // Create directories for numpy files if missing
            fs::create_directory(out_dir);
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
