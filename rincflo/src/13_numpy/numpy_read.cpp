// Local imports
#include <rincflo.H>
#include <cnpy.H>

// Included names
using cnpy::NpyArray, cnpy::npz_t, cnpy::npy_load, cnpy::npz_load, cnpy::make_npy_array;
namespace fs = std::filesystem;

// ************************************************************************************************
void Rincflo::ReadNumpy()
// Write out numpy arrays (.npy files) for analysis in Python
// Uses cnpy library to read Numpy .npy files into objects of type NypArray.
{
    #define FUNC_NAME "Rincflo::ReadNumpy"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Status update
    Print() << format("Reading numpy data...\n");

    // The MPI process number of this thread
    int proc {MyProc()};
    // The MPI process number of the IO thread
    int proc_io {IOProcessorNumber()};
    // Is this the IO processor?
    bool is_io {IOProcessor()};

    // The spatial dimension
    constexpr int dim = SpaceDim;
    // The number of levels
    const int num_levels = max_level + 1;
    // Alias number of species for legibility
    const size_t nsp = m_nspec;
    // Alias the level offset for reading flow data
    const int offset_flow = m_numpy_lev_offset_flow;
    // Alias offset the level offset for reading reaction data
    const int offset_react = m_numpy_lev_offset_react;

    // Read in the scalar data; need to do this on ALL the MPI ranks, not just the IO processor
    string zipname = format("{:s}/scalars.npz", NumpyInputDir_active().c_str(), max_level);
    npz_t scalars_npz = npz_load(zipname);
    m_nstep = scalars_npz["nstep"].as_scalar<int>();
    m_rstep = scalars_npz["rstep"].as_scalar<int>();
    m_cur_time = scalars_npz["cur_time"].as_scalar<Real>();
    m_dt = scalars_npz["dt"].as_scalar<Real>();
    m_prev_dt = scalars_npz["prev_dt"].as_scalar<Real>();
    m_prev_prev_dt = scalars_npz["prev_prev_dt"].as_scalar<Real>();

    // Respect precedence of nstep and rstep in the input file (otherwise these get clobbered by numpy)
    ParmParse pp("rincflo");
    pp.query("nstep", m_nstep);
    pp.query("rstep", m_rstep);

    // Since we just loaded m_rstep, apply this value to m_start_rsteps as well
    m_start_rsteps = m_rstep;

    if (m_numpy_verbose > 0) {Print() << format("Loaded scalars.npz.\n");}
    
    // Set time arrays for use in fillpatching
    UpdateTimeArrays();

    // Always update geometry before reading from numpy; need cell type
    CalcGeometry();

    // Load the scalar data including step counts and times
    if (m_numpy_verbose > 0) {Print() << format("Loaded scalars.npz.\n");}

    // Preallocate empty vectors for the numpy arrays at each level

    // The cell type
    vector<NpyArray> cell_type_npy(num_levels);
    // The refinement level
    vector<NpyArray> refinement_npy(num_levels);
    // The fluid velocity (vx, vy, vz) in each cell in cm/sec
    vector<NpyArray> velocity_npy(num_levels);
    // The fluid pressure in each cell
    vector<NpyArray> pressure_npy(num_levels);
    // The pressure gradient in each cell
    vector<NpyArray> gradp_npy(num_levels);
    // The species concentrations in each cell
    vector<NpyArray> conc_npy(num_levels);
    // The current in milliamps
    vector<NpyArray> current_npy(num_levels);
    // The electric potential in the liquid in millivolts
    vector<NpyArray> epotL_npy(num_levels);
    // The electric potential in the solid in millivolts
    vector<NpyArray> epotS_npy(num_levels);
    // The overpotential in millivolts
    vector<NpyArray> overpot_npy(num_levels);

    if (m_numpy_verbose > 0) {Print() << format("Created empty vector<NpyArray>.\n");}

    // Get the shape of the numpy arrays
    Rincflo::NumpyShapes ns = GetNumpyShapes();

    // Iterate through the levels and load the data on each level
    for (int lev=0; lev < num_levels; ++lev)
    {
        // Status
        if (m_numpy_verbose > 1) {Print() << format("Reading data on level {:d}.\n", lev);}

        // AMReX level data (modifiable)
        auto& ld {*m_leveldata[lev]};

        // BoxArray shared by all these level data
        BoxArray const& ba {grids[lev]};

        // Get a single cell-centered Box for each MultiFab
        const Box bx_1g {ba.minimalBox().enclosedCells()};

        // BoxArray with one grid
        const BoxArray ba_1g(bx_1g);

        // Map all the processes to the IO Processor
        Vector<int> pmap {proc_io};
        
        // DistributionMapping with one grid spread to just the IO Processor
        const DistributionMapping dm_1g(pmap);

        // Status
        if (m_numpy_verbose > 2) {Print() << format("Built single grid objects bx_1g, ba_1g, pmap, dm_1g on level {:d}.\n", lev);}

        // FabFactory for single grid MultiFab objects on this level
        #if (AMREX_USE_EB)
        // Vector<int> nghost_vec {nghost_eb_basic(), nghost_eb_volume(), nghost_eb_full()};
        const auto fact_ptr_1g = makeEBFabFactory(
            geom[lev], ba_1g, dm_1g, ngrow(), EBSupport::full);
        const auto& fact_1g {*fact_ptr_1g};
        #else
        const auto& fact = new FArrayBoxFactory();
        #endif

        // Status
        if (m_numpy_verbose > 2) {Print() << format("Built EB factory on level {:d}.\n", lev);}

        // Number of ghost cells for state variables: velocity, density, conc, epotL, epotS
        const int ng = nghost_state();

        // Build the level data single grid struct for this level
        LevelDataNumpyInput ld_1g(ba_1g, dm_1g, fact_1g, dim, nsp, ng);
        if (m_numpy_verbose > 2) {Print() << format("Built LevelDataNumpyInput ld_1g on level {:d}.\n", lev);}

        // Initialize MFs to zero
        ld_1g.cell_type.setVal(0.0);
        ld_1g.velocity.setVal(0.0);
        ld_1g.pressure.setVal(0.0);
        ld_1g.gradp.setVal(0.0);
        ld_1g.conc.setVal(0.0);
        ld_1g.current.setVal(0.0);
        ld_1g.overpot.setVal(0.0);
        if (is_PotComputed() )
        {
            ld_1g.epotL.setVal(0.0);
            ld_1g.epotS.setVal(0.0);
        }
        if (m_numpy_verbose > 2) {Print() << format("Initialized ld_1g arrays to zero.\n");}

        // Copy cell type data into MultiFab with one grid
        ld_1g.cell_type.ParallelCopy(ld.cell_type, 0, 0, 1);

        // Iterate through the levels, read the data from numpy, and convert from friendly units to MKS
        // attribute    unit            conversion_factor
        // velocity:    cm/sec          1.0E-2
        // source:      millimoles/sec  1.0E-3

        // Only run this block on IO rank
        if (is_io)
        {
            // Do we need to load flow data on this level?
            bool load_flow = (lev + offset_flow < m_numpy_lev_flow);

            // Top level directory with flow data
            const string npy_dir_flow_top = NumpyInputDir_flow();
            // Top level directory with reaction data
            const string npy_dir_react_top = NumpyInputDir_react();

            // Directory where numpy flow data files for this level were saved
            const string npy_dir_flow = format("{:s}/L{:d}", npy_dir_flow_top.c_str(), lev + offset_flow);
            // Directory where numpy reaction data files for this level were saved
            const string npy_dir_react = format("{:s}/L{:d}", npy_dir_react_top.c_str(), lev + offset_react);

            // The name of the file containing the numpy data to be loaded
            string fname;

            // The cell types on this level; if no embedded boundaries, all the cells have type 0
            cell_type_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // The number of levels of refinement in the data file on this level
            refinement_npy[lev] = make_npy_array<uint8_t>(ns.shape_npy_scalar[lev]);
            // Get the cell type and refinement for this level
            MultiFab_CellTypeToNumpy(cell_type_npy, refinement_npy, ld_1g.cell_type, lev);
            // We want to upsample all the cells in this routine regardless of the refinement level
            bool upsample_all = true;

            // Read flow inputs from numpy when applicable
            if (load_flow)
            {
                // Read velocity from numpy; convert from cm/sec to m/sec with conversion factor 1.0E-2
                fname = format("{:s}/velocity.npy", npy_dir_flow.c_str());
                velocity_npy[lev] = npy_load(fname);

                // Read pressure from numpy; units are Pascals, don't rescale
                fname = format("{:s}/pressure_node.npy", npy_dir_flow.c_str());
                pressure_npy[lev] = npy_load(fname);

                // Read pressure gradient from numpy; units are Pascals / meter, don't rescale
                fname = format("{:s}/gradp.npy", npy_dir_flow.c_str());
                gradp_npy[lev] = npy_load(fname);

                // Status
                if (m_numpy_verbose > 1) {Print() << format("Loaded flow data on level {:d}: velocity, pressure, gradp.\n", lev);}
            }
            // Otherwise upsample the flow data
            else
            {
                // Create and upsample the velocity
                velocity_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
                NumpyUpsample(velocity_npy, refinement_npy, lev, false, upsample_all);

                // Create and upsample the pressure
                pressure_npy[lev] = make_npy_array<Real>(ns.shape_npy_pressure[lev]);
                NumpyUpsample(pressure_npy, refinement_npy, lev, false, upsample_all);

                // Create and upsample the pressure gradient
                gradp_npy[lev] = make_npy_array<Real>(ns.shape_npy_vector[lev]);
                NumpyUpsample(gradp_npy   , refinement_npy, lev, false, upsample_all);

                // Status
                if (m_numpy_verbose > 1) {Print() << format("Upsampled flow data on level {:d}: velocity, pressure, gradp.\n", lev);}
            }
            
            // Populate MultiFab of flow data from numpy; when numpy data is unavailable it will be upsampled
            Numpy_ToMultiFab(velocity_npy, ld_1g.velocity, lev, 1.0E-2);
            Numpy_ToMultiFab(pressure_npy, ld_1g.pressure, lev, 1.0);
            Numpy_ToMultiFab(gradp_npy   , ld_1g.gradp   , lev, 1.0);

            // Do we need to populate reaction data from numpy?
            bool need_react = (!m_react_reinit_all_conc);
            // Do we need epotL?
            bool need_epotL = (need_react && is_PotComputed());
            // Do we need epotS?
            bool need_epotS = (need_epotL && m_solve_epotS);
            // Do we need overpot and current? Only when simulating a reaction in the Nernst model
            bool need_opc = (need_react && (m_reaction_model == ReactionModel::Nernst));

            // Do we need to load reaction data on this level?
            bool load_react = (lev + offset_react < m_numpy_lev_react) && need_react;
            // Do we need to load epotL data on this level?
            bool load_epotL = load_react && need_epotL;
            // Do we need to load epotS data on this level?
            bool load_epotS = load_react && need_epotS;
            // Do we need to load overpot and current?
            bool load_opc = load_react && need_opc;

            // Load the basic reaction data
            if (load_react)
            {
                // Read conc from numpy; units are in millimolar, don't rescale
                fname = format("{:s}/conc.npy", npy_dir_react.c_str());
                conc_npy[lev] = npy_load(fname);

                // Also load overpotential and current for the Nernst model only
                if (load_opc)
                {
                    // Read overpotential from numpy; convert from millivolts to volts
                    fname = format("{:s}/overpot.npy", npy_dir_react.c_str());
                    overpot_npy[lev] = npy_load(fname);

                    // Read current from numpy; convert from milliamps to amps
                    fname = format("{:s}/current.npy", npy_dir_react.c_str());
                    current_npy[lev] = npy_load(fname);
                }
            }
            // Otherwise upsample the basic reaction data
            else if (need_react)
            {
                // Create and upsample the conc concentrations
                conc_npy[lev] = make_npy_array<Real>(ns.shape_npy_conc[lev]);
                NumpyUpsample(conc_npy , refinement_npy, lev, false, upsample_all);

                // Upsample overpotential and current for the Nernst model only
                if (need_opc)
                {
                    // Create and upsample the current
                    overpot_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                    NumpyUpsample(overpot_npy, refinement_npy, lev, false, upsample_all);

                    // Create and upsample the overpotential
                    current_npy[lev] = make_npy_array<Real>(ns.shape_npy_scalar[lev]);
                    NumpyUpsample(current_npy, refinement_npy, lev, true , upsample_all);
                }
            }

            // Populate basic reaction data from numpy to MultiFab
            if (need_react)
            {
                Numpy_ToMultiFab(conc_npy , ld_1g.conc , lev, 1.0);
                // Also populate overpot and current in the Nernst model
                if (need_opc)
                {
                    Numpy_ToMultiFab(overpot_npy, ld_1g.overpot, lev, 1.0E-3);
                    Numpy_ToMultiFab(current_npy, ld_1g.current, lev, 1.0E-3);
                }
            }

            // Electric potential in the liquid; convert from millivolts to volts
            if (load_epotL)
            {                
                fname = format("{:s}/epotL.npy", npy_dir_react.c_str());
                epotL_npy[lev] = npy_load(fname);
            }
            else if (need_epotL)
                {NumpyUpsample(epotL_npy, refinement_npy, lev, false, upsample_all);}
            if (need_epotL)
                {Numpy_ToMultiFab(epotL_npy, ld_1g.epotL, lev, 1.0E-3);}

            // Electric potential in the solid; convert from millivolts to volts; only when solving potential in the solid
            if (load_epotS)
            {
                fname = format("{:s}/epotS.npy", npy_dir_react.c_str());
                epotS_npy[lev] = npy_load(fname);
            }
            else if (need_epotS)
                {NumpyUpsample(epotS_npy, refinement_npy, lev, false, upsample_all);}
            if (need_epotS)
                {Numpy_ToMultiFab(epotS_npy, ld_1g.epotS, lev, 1.0E-3);}
        }   // if (is_io)

        // The density is constant
        ld.density.setVal(m_ro_0);

        // Perform a parallel copy from the single grid MultiFab on the communicator rank to each working MultiFab.
        // arguments to ParallelCopy: src, src_comp, dest_comp, num_comp, src_nghost, dst_nghost

        // Flow variables
        ld.velocity.ParallelCopy(ld_1g.velocity, 0, 0, dim, ng, ng);
        ld.pressure.ParallelCopy(ld_1g.pressure, 0, 0,   1,  0, 0);
        ld.gradp.ParallelCopy   (ld_1g.gradp,    0, 0, dim,  0, 0);

        // Reaction state variables
        ld.conc.ParallelCopy  (ld_1g.conc,   0, 0,  nsp, ng, ng);

        // Copy the electric potential if applicable
        if (is_PotComputed() )
        {
            ld.epotL.ParallelCopy(ld_1g.epotL, 0, 0, 1, ng, ng);
            ld.epotS.ParallelCopy(ld_1g.epotS, 0, 0, 1, ng, ng);
        }
        // When not computing the electric potential, initialize epotL and epotS consistent
        // with the applied voltage m_DVapp and the assumptions of the special model
        else
        {
            ld.epotL.setVal(0.0);
            ld.epotS.setVal(m_DVapp);
        }

        // Reaction cut cells
        ld.current.ParallelCopy(ld_1g.current, 0, 0, 1, 0, 0);
        ld.overpot.ParallelCopy(ld_1g.overpot, 0, 0, 1, 0, 0);

        // Populate the old versions of the MultiFabs
        ld.density_o.setVal(m_ro_0);
        ld.velocity_o.ParallelCopy(ld.velocity, 0, 0,  dim, ng, ng);
        ld.conc_o.ParallelCopy(ld.conc    , 0, 0,  nsp, ng, ng);
        ld.epotL_o.ParallelCopy(ld.epotL      , 0, 0,   1, ng, ng);
        ld.epotS_o.ParallelCopy(ld.epotS      , 0, 0,   1, ng, ng);

        if (m_numpy_verbose > 1) {Print() << format("Completed parallel copy to working MultiFabs from single grid MFs on level {:d}.\n", lev);}

    }   // for(lev)

    // Perform a round of fillpatching on all the levels
    for (int lev=0; lev < num_levels; ++lev)
    {
        // AMReX level data (modifiable)
        auto& ld {*m_leveldata[lev]};
        // Time for current and previous reaction step; used for fill patching
        Real time = m_cur_time;
        Real time_o = m_cur_time - m_dt;
        // Number of ghost cells for fill patching
        const int ng = nghost_state();

        // Run fillpatching on fluid flow variables
        fillpatch_velocity(lev, time  , ld.velocity,   ng);
        fillpatch_velocity(lev, time_o, ld.velocity_o, ng);
        fillpatch_gradp   (lev, time,   ld.gradp,       0);
        fillpatch_density (lev, time,   ld.density,    ng);
        fillpatch_density (lev, time,   ld.density_o,  ng);

        // Run fillpatching on reaction variables
        fillpatch_conc (lev, time  , ld.conc   , ng);
        fillpatch_conc (lev, time_o, ld.conc_o , ng);
        fillpatch_overpot(lev, time  , ld.overpot  ,  0);
        if (is_PotComputed() )
        {
            fillpatch_epotL(lev, time  , ld.epotL  , ng);
            fillpatch_epotS(lev, time  , ld.epotS  , ng);
            fillpatch_epotL(lev, time_o, ld.epotL_o, ng);
            fillpatch_epotS(lev, time_o, ld.epotS_o, ng);
        }

        if (m_numpy_verbose > 1) {Print() << format("Completed fillpatching on level {:d}.\n", lev);}
    }

    // Scale velocity if applicable
    if (m_vel_scale_factor != 1.0)
        {ScaleInputVelocity();}

    // Call redistribution routine to update internal ghost values on MultiFabs
    AMREX_EB_ONLY(InitialRedistribution());

    // Update both solved and derived reaction quantities
    CalcSourceTermAll();
    CalcDerivedReact();

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::ScaleInputVelocity()
{
    #define FUNC_NAME "Rincflo::scale_input_vel"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Status message
    Print() << format("Scaling input velocity by {:12.9f}.\n", m_vel_scale_factor);

    // The number of levels
    const int num_levels = max_level+1;

    // Iterate through the levels and compute the data on each level
    for (int lev=0; lev < num_levels; ++lev)
    {
        // AMReX level data (modifiable)
        auto& ld {*m_leveldata[lev]};

        // All of these MultiFabs must be scaled: velocity, pressure, pressure gradient
        ld.velocity.mult(m_vel_scale_factor);
        ld.velocity_o.mult(m_vel_scale_factor);
        ld.pressure.mult(m_vel_scale_factor);
        ld.gradp.mult(m_vel_scale_factor);
        // mac_phi is the cell-centered pressure
        ld.mac_phi.mult(m_vel_scale_factor);
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
