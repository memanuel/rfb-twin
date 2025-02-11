// Library imports
#include <rincflo.H>

// ************************************************************************************************
string Rincflo::NumpyInputDir_flow() const
{
    // The default directory for numpy flow data when not specfied in the config file
    string dir_top_dflt = format("numpy/step_{:0{}d}",  m_nstep, m_chk_min_digits);

    // Choose the input directory if specified, otherwise use the default
    return (m_numpy_input_dir_flow.length() > 0) ? m_numpy_input_dir_flow : dir_top_dflt;
}

// ************************************************************************************************
string Rincflo::NumpyInputDir_react() const
{
    // The default directory for numpy flow data when not specfied in the config file
    string dir_top_dflt = format("numpy/rstep_{:0{}d}",  m_rstep, m_chk_min_digits);

    // Choose the input directory if specified, otherwise use the default
    return (m_numpy_input_dir_react.length() > 0) ? m_numpy_input_dir_react: dir_top_dflt;
}

// ************************************************************************************************
string Rincflo::NumpyInputDir_active() const
{
    return is_ActiveSimReact() ? NumpyInputDir_react() : NumpyInputDir_flow();
}

// ************************************************************************************************
string Rincflo::NumpyOutputDir() const
{
    // The default directory for numpy data when not specfied in the config file
    string dir_top_dflt = is_ActiveSimReact() ? 
        format("numpy/rstep_{:0{}d}", m_rstep, m_chk_min_digits) :
        format("numpy/step_{:0{}d}",  m_nstep, m_chk_min_digits);

    // Choose the input directory if specified, otherwise use the default
    return (m_numpy_output_dir.length() > 0) ? m_numpy_output_dir : dir_top_dflt;
}

// ************************************************************************************************
Rincflo::NumpyShapes Rincflo::GetNumpyShapes() const
{
    // The spatial dimension
    constexpr size_t dim = SpaceDim;
    // The number of levels we are writing out to numpy
    const int num_levels = max_level+1;
    // Alias number of species for legibility
    const size_t nsp = m_nspec;

    // Initialize empty arrays of the right size
    NumpyShapes ns(num_levels, dim);

    // Iterate through the levels and save the sizes and shapes of numpy arrays for scalar and vector fields
    for (int lev=0; lev < num_levels; ++lev)
    {
        // The geometry object for this level
        const auto& geom {Geom(lev)};
        // The domain object for the geometry
        const auto& domain {geom.Domain()};

        // Save the lo and hi domain; convert from meters to cm.  
        for (int dir = 0; dir < dim; ++dir)
        {
            // Row multiplier is 3 even in 2D simulations b/c we write out nominal height in z
            int r = lev * 3;
            // Multiply geometry by 100 to convert from meters to cm
            ns.domain_lo[r + dir] = geom.ProbLo(dir) * 100.0;
            ns.domain_hi[r + dir] = geom.ProbHi(dir) * 100.0;
        }

        // Save the nominal height in z-axis when running a 2D sim
        #if (AMREX_IS_2D)
        // Row number for z component of this level in the domain bounds array
        int domain_idx_z = lev * 3 + 2;
        ns.domain_lo[domain_idx_z] = 0.0;
        ns.domain_hi[domain_idx_z] = m_height * 100.0; 
        #endif

        // Number of grid points along x, y, z axes
        const size_t nx = dim > 0 ? domain.length(0) : 1;
        const size_t ny = dim > 1 ? domain.length(1) : 1;
        const size_t nz = dim > 2 ? domain.length(2) : 1;

        // Write out the grid shape entries
        const size_t r = lev * dim;
        AMREX_D_TERM(
        ns.grid_shape[r+0] = nx;,
        ns.grid_shape[r+1] = ny;,
        ns.grid_shape[r+2] = nz;)

        // Shape of numpy array to store a scalar field
        ns.shape_npy_scalar[lev] = {AMREX_D_DECL(nx, ny, nz)};
        // Shape of numpy array to store a vector field
        ns.shape_npy_vector[lev] = {AMREX_D_DECL(nx, ny, nz), dim};
        // Shape of numpy array to store concentrations
        ns.shape_npy_conc[lev] = {AMREX_D_DECL(nx, ny, nz), nsp};
        // Shape of numpy array to store the pressure; this is nodal so it's bigger
        ns.shape_npy_pressure[lev] = {AMREX_D_DECL((nx+1), (ny+1), (nz+1))};
    }
    return ns;
}

// ************************************************************************************************
void Rincflo::MultiFab_IndexToNumpy(
    NpyArray& idx_npy, NpyArray& pos_npy, const MultiFab& mf_1g, int lev)
// Write the index values and grid centers to idx_npy and pos_npy; these are aligned as a vector field
{
    #define FUNC_NAME "Rincflo::MultiFab_IndexToNumpy"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Check that level is in permissible range
    if (lev < 0 || lev > max_level)
        {Abort(format("Rincflo::MultiFab_IndexToNumpy: Bad level = {:d}, must be in range [0, {:d}].\n", lev, max_level));}

    // The geometry object for this level
    const auto& geom {Geom(lev)};
    // The domain object for the geometry
    const auto& domain {geom.Domain()};
    
    // Number of grid points along x, y, z axes
    AMREX_D_TERM(
    const size_t nx = domain.length(0);,
    const size_t ny = domain.length(1);,
    const size_t nz = domain.length(2);)

    // Alias the shape of the numpy arrays
    const vector<size_t>& shape_idx = idx_npy.shape;
    const vector<size_t>& shape_pos = pos_npy.shape;
    // The expected shape of the two numpy arrays
    const vector<size_t> shape {AMREX_D_DECL(nx, ny, nz)};

    // Fail if the shape of either array does not match expected shape
    if (shape_idx != shape)
        {Abort("MultiFab_IndexToNumpy: bad shape_idx, does not match expected shape.\n");}
    if (shape_pos != shape)
        {Abort("MultiFab_IndexToNumpy: bad shape_pos, does not match expected shape.\n");}

    // Data pointers for the index and position
    int32_t* idx = idx_npy.data<int32_t>();
    Real* pos = pos_npy.data<Real>();

    // Low end of the domain in x, y, z axes
    AMREX_D_TERM(
    const Real x_lo {geom.ProbLo(0)};,
    const Real y_lo {geom.ProbLo(1)};,
    const Real z_lo {geom.ProbLo(2)};)

    // Grid spacing in x, y, z axes
    AMREX_D_TERM(
    const Real dx {geom.CellSize(0)};,
    const Real dy {geom.CellSize(1)};,
    const Real dz {geom.CellSize(2)};)

    // The spatial dimension
    constexpr int dim = SpaceDim;
    
    // Iterate through the boxes in the single grid
    for(MFIter mfi(mf_1g, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        // The valid box of grid cells
        const auto& bx = mfi.tilebox();

        // Loop over the box
        auto func = 
        [=, this, &idx_npy, &pos_npy] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Row index for vector entry
            int rv {NumpyRowVector(AMREX_D_DECL(nx, ny, nz), dim, AMREX_D_DECL(i, j, k), 0)};

            // Row index for x, y, z entries
            AMREX_D_TERM(
            const int rx {rv + 0};,
            const int ry {rv + 1};,
            const int rz {rv + 2};)

            // Coordinates at center of grid cell
            AMREX_D_TERM(
            double x = x_lo + (i + 0.5) * dx;,
            double y = y_lo + (j + 0.5) * dy;,
            double z = z_lo + (k + 0.5) * dz;)

            // Write out grid index entries [i, j, k]
            AMREX_D_TERM(
            idx[rx] = i;,
            idx[ry] = j;,
            idx[rz] = k;)

            // Write out position at center of grid cell [x, y, z]
            AMREX_D_TERM(
            pos[rx] = x;,
            pos[ry] = y;,
            pos[rz] = z;)
        };
        ParallelFor(bx, func);
    } // for / mfi
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::MultiFab_CellTypeToNumpy_top(NpyArray& cell_type_npy, const MultiFab& cell_type_mf_1g)
// Write out the cell_type array at the top level (lev=0). Start with the real valued MultiFab cell_type_mf_1g.
// Convert the cell types from amrex::Real to uint8_t; write to cell_type in place.
{
    #define FUNC_NAME "Rincflo::MultiFab_CellTypeToNumpy_top"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The top level is 0
    constexpr int lev {0};
    // The geometry object for this level
    const auto& geom {Geom(lev)};
    // The domain object for the geometry
    const auto& domain {geom.Domain()};

    // Number of grid points along x, y, z axes
    AMREX_D_TERM(
    const size_t nx = domain.length(0);,
    const size_t ny = domain.length(1);,
    const size_t nz = domain.length(2);)

    // Alias the shape of the cell type array
    const vector<size_t>& shape_npy = cell_type_npy.shape;
    // The expected shape of the cell type arrays
    const vector<size_t> shape {AMREX_D_DECL(nx, ny, nz)};

    // Fail if the shape of either array does not match expected shape
    if (shape_npy != shape)
        {Abort("MultiFab_CellTypeToNumpy_top: bad shape of cell_type_npy, does not match expected shape.\n");}

    // The pointer to the cell type; modifiable with array semantics cell_type_arr = rhs
    uint8_t* cell_type_arr = cell_type_npy.data<uint8_t>();
    
    // Iterate through the boxes in the single grid
    for(MFIter mfi(cell_type_mf_1g, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        // The valid box of grid cells
        const Box& bx = mfi.tilebox();
        // The cell type as a const array
        const auto& cell_type = cell_type_mf_1g.const_array(mfi);

        // Loop over the box
        auto func = 
        [=, this, &cell_type_npy] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // The value at this entry
            Real val = cell_type(i,j,k);
            // Is this entry valid?
            bool is_ok = (0 <= val) && (val < 256) && (!isnan(val));
            // Row index for scalar entry
            int r {NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i, j, k))};
            // Write out the value if it was valid; otherwise write 0 as a placeholder
            cell_type_arr[r] = is_ok ? static_cast<uint8_t>(val) : 0;
        };
        ParallelFor(bx, func);
    } // for / mfi
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::MultiFab_CellTypeToNumpy(
        vector<NpyArray>& cell_type_npy, vector<NpyArray>& refinement_npy, const MultiFab& cell_type_mf_1g, int lev)
// Write out the cell_type array at a level > 0.  Initialize by upsampling the previous level.
// Then convert the real valued data in cell_type_mf_1g to uint8 on the boxes where it's defined.
{
    #define FUNC_NAME "Rincflo::MultiFab_CellTypeToNumpy"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Check that level is in permissible range
    if (lev < 0 || lev > max_level)
        {Abort(format("Rincflo::MultiFab_CellTypeToNumpy: Bad level = {:d}, must be in range [0, {:d}].\n", lev, max_level));}

    // Alias the cell type for the fine level (current)
    NpyArray& cell_type_npy_fine = cell_type_npy[lev];

    // Delegate to MultiFab_CellTypeToNumpy_top on level 0
    if (lev == 0)
        {return MultiFab_CellTypeToNumpy_top(cell_type_npy_fine, cell_type_mf_1g);}

    // If we get here, we're guarenteed that lev > 0
    // Alias the cell_type for the coarse level (previous)
    const NpyArray& cell_type_npy_crse = cell_type_npy[lev-1];

    // The pointers to the cell type; accessible with array semantics
    uint8_t* cell_type_fine = cell_type_npy_fine.data<uint8_t>();
    const uint8_t* cell_type_crse = cell_type_npy_crse.const_data<uint8_t>();

    // Alias the refinement for the current and previous levels
    NpyArray& refinement_npy_fine = refinement_npy[lev];
    const NpyArray& refinement_npy_crse = refinement_npy[lev-1];

    // The pointers to the refinement level; accessible with array semantics
    uint8_t* refinement_fine = refinement_npy_fine.data<uint8_t>();
    const uint8_t* refinement_crse = refinement_npy_crse.const_data<uint8_t>();

    // Alias the dimension
    constexpr int dim = SpaceDim;

    // The geometry objects at the fine and coarse levels
    const auto& geom_fine {Geom(lev)};
    const auto& geom_crse {Geom(lev-1)};

    // The domain objects at the fine and coarse levels
    const auto& domain_fine {geom_fine.Domain()};
    const auto& domain_crse {geom_crse.Domain()};

    // Number of grid points along x, y, z axes on the fine level
    const size_t nx_fine = (dim > 0) ? domain_fine.length(0) : 1;
    const size_t ny_fine = (dim > 1) ? domain_fine.length(1) : 1;
    const size_t nz_fine = (dim > 2) ? domain_fine.length(2) : 1;

    // Number of grid points along x, y, z axes on the coarse level
    const size_t nx_crse = (dim > 0) ? domain_crse.length(0) : 1;
    const size_t ny_crse = (dim > 1) ? domain_crse.length(1) : 1;
    const size_t nz_crse = (dim > 2) ? domain_crse.length(2) : 1;

    // The refinement ratio in AMReX between levels is always 2
    constexpr int ratio = 2;

    // First iterate over the ENTIRE fine domain and fill in provisional values from the coarse domain
    for (int i_fine = 0; i_fine < nx_fine; ++i_fine) 
    {
        // x coordinate on the coarse grid
        int i_crse = i_fine / ratio;
        for (int j_fine = 0; j_fine < ny_fine; ++j_fine)
        {
            // y coordinate on the coarse grid
            int j_crse = j_fine / ratio;
            for (int k_fine = 0; k_fine < nz_fine; ++k_fine)
            {
                // z coordinate on the coarse grid
                int k_crse = k_fine / ratio;
                // Row index for scalar entry on the fine level (current)
                int r_fine {NumpyRowScalar(AMREX_D_DECL(nx_fine, ny_fine, nz_fine), AMREX_D_DECL(i_fine, j_fine, k_fine))};
                // Row index for scalar entry on the coarse level (prev)
                int r_crse {NumpyRowScalar(AMREX_D_DECL(nx_crse, ny_crse, nz_crse), AMREX_D_DECL(i_crse, j_crse, k_crse))};
                // Write out the fine value from the coarse array
                cell_type_fine[r_fine] = cell_type_crse[r_crse];
                // Inherit the level of refinment from the prior level
                refinement_fine[r_fine] = refinement_crse[r_crse];
            }
        }
    }

    // Iterate through the boxes in the single grid
    for(MFIter mfi(cell_type_mf_1g, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        // The valid box of grid cells
        const Box& bx = mfi.tilebox();
        // The cell type as a const array
        const auto& cell_type = cell_type_mf_1g.const_array(mfi);

        // Loop over the box
        auto func = 
        [=, this, &cell_type_npy_fine, &refinement_npy_fine] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // The value at this entry
            Real val = cell_type(i,j,k);
            // Is this entry valid?
            bool is_ok = (!isnan(val)) && (0 <= val) && (val < 256);
            // Write out the value if it's valid
            if (is_ok)
            {
                // Row index for scalar entry on the fine level
                int r{NumpyRowScalar(AMREX_D_DECL(nx_fine, ny_fine, nz_fine), AMREX_D_DECL(i, j, k))};
                // Write out cell type to the numpy array
                cell_type_fine[r] = static_cast<uint8_t>(val);
                // Write out the refinment for this level
                refinement_fine[r] = static_cast<uint8_t>(lev);
            }
        };
        ParallelFor(bx, func);
    } // for / mfi
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
// Convert a MultiFab of data to a vector<NpyArray> that can be saved to numpy.
// Type NpyArray is essentially a flattened C++ vector with some additional metadata; it's defined in cnpy.
void Rincflo::MultiFab_ToNumpy(vector<NpyArray>& data_npy, const MultiFab& mf_1g, int lev, Real conversion_factor)
{
    #define FUNC_NAME "Rincflo::MultiFab_ToNumpy"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Check that level is in permissible range
    if (lev < 0 || lev > max_level)
        {Abort(format("Rincflo::MultiFab_ToNumpy: Bad level = {:d}, must be in range [0, {:d}].\n", lev, max_level));}

    // The spatial dimension
    constexpr int dim = SpaceDim;

    // Numpy data on this level
    NpyArray& data_npy_lev = data_npy[lev];

    // Alias the shape of the numpy array
    const vector<size_t>& shape_npy = data_npy_lev.shape;

    // Number of dimensions in the numpy array
    const int dim_npy = shape_npy.size();

    // Is this a scalar numpy array?
    const bool is_scalar_array = (dim_npy == dim);

    // Is this a vector numpy array?
    const bool is_vector_array = (dim_npy == dim + 1);

    // Fail if dim_npy is not either dim (scalar nuppy array) or dim+1 (vector numpy array)
    if (!(is_scalar_array || is_vector_array))
        {Abort(format("Rincflo::MultiFab_ToNumpy. Bad numpy dimension! dim={:d}, dim_npy={:d}.\n", dim, dim_npy));}

    // Get the integers nx, ny, nz, nc with the size of the array
    const size_t nx = (dim > 0) ? shape_npy[0] : 1;
    const size_t ny = (dim > 1) ? shape_npy[1] : 1;
    const size_t nz = (dim > 2) ? shape_npy[2] : 1;
    const size_t nc = is_vector_array ? shape_npy[dim] : 1;

    // Fail if the number of components does not match between the numpy shape and the MultiFab
    if (nc != mf_1g.n_comp)
        {Abort(format("Rincflo::MultiFab_ToNumpy. Number of components doesn't match! numpy={:d}, MultiFab={:d}.\n", nc, mf_1g.n_comp));}

    // The data pointer; modifiable
    Real* data = data_npy_lev.data<Real>();

    // Iterate through the boxes in the single grid
    for(MFIter mfi(mf_1g, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        // The valid box of grid cells
        const Box& bx = mfi.tilebox();

        // Handle scalar arrays
        if (is_scalar_array)
        {
            // The scalar field (phi is a stand-in for a generic scalar field)
            const auto& phi = mf_1g.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Row index for scalar entry
                int r {NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i, j, k))};
                // Write out scalar value
                data[r] = phi(i,j,k) * conversion_factor;
            };
            ParallelFor(bx, func);
        }
        // Handle vector arrays
        else if (is_vector_array)
        {
            // The writable vector field (F is a stand-in for a generic vector field)
            const auto& F = mf_1g.const_array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Row index for vector entry
                int rv {NumpyRowVector(AMREX_D_DECL(nx, ny, nz), nc, AMREX_D_DECL(i, j, k), 0)};
                // Write out vector field components in this grid cell
                for (int c = 0; c < nc; ++c) 
                    {data[rv++] = F(i, j, k, c) * conversion_factor;}
            };
            ParallelFor(bx, func);
        }

    } // for / mfi
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::Numpy_ToMultiFab(
    const vector<NpyArray>& data_npy, MultiFab& mf_1g, int lev, Real conversion_factor)
{
    #define FUNC_NAME "Rincflo::Numpy_ToMultiFab"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Check that level is in permissible range
    if (lev < 0 || lev > max_level)
        {Abort(format("Rincflo::Numpy_ToMultiFab: Bad level = {:d}, must be in range [0, {:d}].\n", lev, max_level));}

    // The spatial dimension
    constexpr int dim = SpaceDim;

    // Numpy data on this level
    const NpyArray& data_npy_lev = data_npy[lev];

    // Alias the shape of the numpy array
    const vector<size_t>& shape_npy = data_npy_lev.shape;

    // Number of dimensions in the numpy array
    const size_t dim_npy = shape_npy.size();

    // Is this a scalar numpy array?
    const bool is_scalar_array = (dim_npy == dim);

    // Is this a vector numpy array?
    const bool is_vector_array = (dim_npy == dim + 1);

    // Fail if dim_npy is not either dim (scalar nuppy array) or dim+1 (vector numpy array)
    if (!(is_scalar_array || is_vector_array))
        {Abort(format("Rincflo::Numpy_ToMultiFab. Bad numpy dimension! dim={:d}, dim_npy={:d}.\n", dim, dim_npy));}

    // Get the integers Nx, Ny, Nz, Nc with the size of the numpy array
    const size_t nx = (dim > 0) ? shape_npy[0] : 1;
    const size_t ny = (dim > 1) ? shape_npy[1] : 1;
    const size_t nz = (dim > 2) ? shape_npy[2] : 1;
    const size_t nc = is_vector_array ? shape_npy[dim] : 1;

    // Fail if the number of components does not match between the numpy shape and the MultiFab
    if (nc != mf_1g.n_comp)
        {Abort(format("Rincflo::Numpy_ToMultiFab. Number of components doesn't match! numpy={:d}, MultiFab={:d}.\n", nc, mf_1g.n_comp));}

    // The data array; constant
    const Real* data = data_npy_lev.const_data<Real>();

    // Iterate through the boxes in the single grid
    for(MFIter mfi(mf_1g, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        // The valid box of grid cells
        const Box& bx = mfi.tilebox();

        // Handle scalar arrays
        if (is_scalar_array)
        {
            // The writable scalar field (phi is a stand-in for a generic field)
            const auto& phi = mf_1g.array(mfi);

            ParallelFor(bx, 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Row index for scalar entry
                int r {NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i, j, k))};
                // Read in scalar value if the value is good
                if (!isnan(data[r]))
                    {phi(i, j, k) = data[r] * conversion_factor;}
            });
        }
        else if (is_vector_array)
        {
            // The writable vector field (F is a stand-in for a generic field)
            const auto& F = mf_1g.array(mfi);

            ParallelFor(bx, 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Row index for vector entry (first component)
                int rv {NumpyRowVector(AMREX_D_DECL(nx, ny, nz), nc, AMREX_D_DECL(i, j, k), 0)};
                // Write out vector field components in this grid cell
                for (int c = 0; c < nc; ++c) 
                {
                    if (!isnan(data[rv+c]))
                        {F(i, j, k, c) = data[rv+c] * conversion_factor;}
                }
            });
        }
    } // for / mfi

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::NumpyUpsample(
    vector<NpyArray>& data_npy, const vector<NpyArray>& refinement_npy, int lev, bool is_extensive, bool upsample_all)
// Upsample numpy data from data_coarse to data_fine; these are both NpyArray instances.
{
    #define FUNC_NAME "Rincflo::NumpyUpsample"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Check that level is in permissible range
    if (lev < 0 || lev > max_level)
        {Abort(format("Rincflo::NumpyUpsample: Bad level = {:d}, must be in range [0, {:d}].\n", lev, max_level));}

    // Alias dimension
    constexpr int dim = SpaceDim;

    // The coarse numpy data; const
    const Real* data_crse = data_npy[lev-1].const_data<Real>();
    // The fine numpy data; modifiable
    Real* data_fine = data_npy[lev].data<Real>();
    // The refinement at the fine level (the level we're writing to)
    const uint8_t* refinement_fine = refinement_npy[lev].const_data<uint8_t>();

    // Shape of coarse and fine
    const vector<size_t>& shape_crse = data_npy[lev-1].shape;
    const vector<size_t>& shape_fine = data_npy[lev].shape;
    // Fail if different number of dimensions
    if (shape_fine.size() != shape_crse.size())
    {
        string msg = format("Rincflo::NumpyUpsample. Inconsistent dimensions between fine ({:d}) and coarse ({:d}).\n", 
            shape_fine.size(), shape_crse.size());
        Abort(msg);
    }
    // Alias the number of dimensions in the numpy data; shared by coarse and fine
    const int dim_npy = shape_crse.size();

    // Is this a scalar numpy array?
    const bool is_scalar_array = (dim_npy == dim);
    // Is this a vector numpy array?
    const bool is_vector_array = (dim_npy == dim + 1);

    // Fail if shape isn't either a scalar or vector array
    if (!(is_scalar_array || is_vector_array))
        {Abort(format("Rincflo::NumpyUpsample. Bad numpy dimension! dim={:d}, dim_npy={:d}.\n", dim, dim_npy));}

    // Number of grid points along x, y, z axes in the coarse domain
    const size_t nx_crse = (dim > 0) ? shape_crse[0] : 1;
    const size_t ny_crse = (dim > 1) ? shape_crse[1] : 1;
    const size_t nz_crse = (dim > 2) ? shape_crse[2] : 1;

    // Number of grid points along x, y, z axes in the fine domain
    const size_t nx_fine = (dim > 0) ? shape_fine[0] : 1;
    const size_t ny_fine = (dim > 1) ? shape_fine[1] : 1;
    const size_t nz_fine = (dim > 2) ? shape_fine[2] : 1;

    // The refinement ratio is always 2 in AMReX
    constexpr int ratio = 2;

    // Total number of cells in domain at coarse and fine level, both as a real number
    const Real num_cells_coarse_r 
        {static_cast<Real>(nx_crse) * static_cast<Real>(ny_crse) * static_cast<Real>(nz_crse)};
    const Real num_cells_fine_r 
        {static_cast<Real>(nx_fine) * static_cast<Real>(ny_fine) * static_cast<Real>(nz_fine)};

    // The multiplier for extensive quantities; just set this to 1.0 for intensive to avoid branching    
    const Real extensive_factor = is_extensive ? (num_cells_coarse_r / num_cells_fine_r) : 1.0;

    // Number of components
    const size_t nc_crse = is_vector_array ? shape_crse[dim] : 1;
    const size_t nc_fine = is_vector_array ? shape_fine[dim] : 1;

    // Fail if inconsistent number of components
    if (nc_fine != nc_crse)
        {Abort(format("Rincflo::NumpyUpsample. Number of components doesn't match! coarse={:d}, fine={:d}.\n", nc_crse, nc_fine));}

    // Alias shared number of components
    const int& nc = nc_crse;

    // Status
    if (m_numpy_verbose > 2)
    {
        Print() << format("NumpyUpsample. lev={:d}. {:s} array.\n", lev, (is_scalar_array) ? "scalar" : "vector");
        if (is_scalar_array)
        {
            Print() << format("Coarse shape ({:d}, {:d}, {:d}).\n", nx_crse, ny_crse, nz_crse);
            Print() << format("Fine shape   ({:d}, {:d}, {:d}).\n", nx_fine, ny_fine, nz_fine);
        }
        if (is_vector_array)
        {
            Print() << format("Coarse shape ({:d}, {:d}, {:d}, {:d}).\n", nx_crse, ny_crse, nz_crse, nc);
            Print() << format("Fine shape   ({:d}, {:d}, {:d}, {:d}).\n", nx_fine, ny_fine, nz_fine, nc);
        }
    }

    // Iterate over the fine data
    for (int i_fine=0; i_fine < nx_fine; ++i_fine)
    {
        // x index on the coarse domain
        int i_crse = i_fine / ratio;
        for (int j_fine=0; j_fine < ny_fine; ++j_fine)
        {
            // y index on the coarse domain
            int j_crse = j_fine / ratio;
            for (int k_fine=0; k_fine < nz_fine; ++k_fine)
            {
                // z index on the coarse domain
                int k_crse = k_fine / ratio;

                // The row index on the coarse domain
                int r_crse {NumpyRowVector(AMREX_D_DECL(nx_crse, ny_crse, nz_crse), nc, AMREX_D_DECL(i_crse, j_crse, k_crse), 0)};

                // The row index on the fine domain for vector data
                int r_fine {NumpyRowVector(AMREX_D_DECL(nx_fine, ny_fine, nz_fine), nc, AMREX_D_DECL(i_fine, j_fine, k_fine), 0)};

                // The row index on the fine domain for scalar data on the fine domain; used to check the refinement level
                int i_ref {NumpyRowScalar(AMREX_D_DECL(nx_fine, ny_fine, nz_fine), AMREX_D_DECL(i_fine, j_fine, k_fine))};

                // Is the refinement level of this cell smaller than the level? Then it needs refinement
                if (upsample_all || (refinement_fine[i_ref] < lev))
                {
                    // Copy from coarse to fine for all components, with extensive factor
                    for (int c = 0; c < nc; ++c)
                        {data_fine[r_fine++] = data_crse[r_crse++] * extensive_factor;}
                }
            }
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// ************************************************************************************************
void Rincflo::NumpyFilter(NpyArray& data_npy, const NpyArray& cell_type_npy, uint8_t mask)
// Filter a numpy data array in place; keep only elements that match the given mask.
// If an element does not match, overwrite it with NaN.
{
    #define FUNC_NAME "Rincflo::NumpyFilter"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The spatial dimension
    constexpr int dim = SpaceDim;

    // Alias the shape of the numpy array
    const auto& shape_npy = data_npy.shape;

    // Number of dimensions in the numpy array
    const int dim_npy = shape_npy.size();

    // Is this a scalar numpy array?
    const bool is_scalar_array = (dim_npy == dim);

    // Is this a vector numpy array?
    const bool is_vector_array = (dim_npy == dim + 1);

    // Fail if dim_npy is not either dim (scalar nuppy array) or dim+1 (vector numpy array)
    if (!(is_scalar_array || is_vector_array))
        {Abort(format("Rincflo::NumpyFilter. Bad numpy dimension! dim={:d}, dim_npy={:d}.\n", dim, dim_npy));}

    // Get the integers nx, ny, nz, nc with the size of the array
    const size_t nx = (dim > 0) ? shape_npy[0] : 1;
    const size_t ny = (dim > 1) ? shape_npy[1] : 1;
    const size_t nz = (dim > 2) ? shape_npy[2] : 1;
    const size_t nc = is_vector_array ? shape_npy[dim] : 1;

    // The data pointer
    Real* const data = data_npy.data<Real>();

    // The pointer to the cell type; accessible with array semantics
    const uint8_t* cell_type = cell_type_npy.const_data<uint8_t>();
    
    // The number of grid cells (multiply spatial dimensions but not components)
    const size_t grid_sz = nx * ny * nz;
    // Fail if the number of elements in cell_type does not match what is expected
    if (cell_type_npy.num_vals != grid_sz)
        {Abort(format("Rincflo::NumpyFilter. Bad cell_type size! Expected {:d}, got {:d}.\n", grid_sz, cell_type_npy.num_vals));}

    // The expected number of data entries in the numpy array
    const size_t data_sz = grid_sz * nc;
    // Fail if the data size does not match what is expected
    if (data_npy.num_vals != data_sz)
        {Abort(format("Rincflo::NumpyFilter. Bad data size! Expected {:d}, got {:d}.\n", data_sz, data_npy.num_vals));}

    // Iterate through array, overwriting non-matching elements with NaN
    for (size_t i = 0; i < grid_sz; ++i)
    {
        // Does this cell type not match the mask?
        if (!(cell_type[i] & mask))
        {
            // The row number of the first of dim elements in the numpy vector array
            size_t r {nc * i};
            // Overwrite these elements with NaN, one for each component
            for (int c = 0; c < nc; ++c)
                {data[r++] = NaN;}
        }
    }
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
void Rincflo::NumpyNodeToCell(const NpyArray& data_node_npy, NpyArray& data_cell_npy)
{
    #define FUNC_NAME "Rincflo::NumpyNodeToCell"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The spatial dimension
    constexpr int dim = SpaceDim;

    // Alias the shape of the nodal and cell-centered arrays
    const auto& shape_node = data_node_npy.shape;
    const auto& shape_cell = data_cell_npy.shape;

    // Fail if dimensions are not equal
    if (shape_cell.size() != shape_node.size())
        {Abort(format("Rincflo::NumpyNodeToCell. Dimension of shape_node doesn't match shape_cell!\n"));}

    // Number of dimensions in the numpy arrays
    const int dim_npy = shape_node.size();

    // Is this a scalar numpy array?
    const bool is_scalar_array = (dim_npy == dim);
    // Is this a vector numpy array?
    // const bool is_vector_array = (dim_npy == dim + 1);

    // Fail if it isn't a scalar array
    if (!is_scalar_array)
        {Abort(format("Rincflo::NumpyNodeToCell. Bad numpy dimension, expected a scalar array! dim={:d}, dim_npy={:d}.\n", dim, dim_npy));}

    // Get the integers nx, ny, nz, nc with the size of the cell-centered array
    const size_t nx = (dim > 0) ? shape_cell[0] : 1;
    const size_t ny = (dim > 1) ? shape_cell[1] : 1;
    const size_t nz = (dim > 2) ? shape_cell[2] : 1;
    // const size_t nc = is_vector_array ? shape_cell[dim] : 1;

    // The expected shape of the nodal array
    vector<size_t> shape_node_expected {AMREX_D_DECL(nx+1, ny+1, nz+1)};
    // vector<size_t> shape_node_expected = is_vector_array ? vector<size_t>{nx+1, ny+1, nz+1, nc} : vector<size_t>{nx+1, ny+1, nz+1};
    // Fail if shape of nodal array does not match
    if (shape_node != shape_node_expected)
    {
        string shape_node_str = AMREX_D_PICK(
            format("({:d})", shape_node[0]), 
            format("({:d},{:d})", shape_node[0], shape_node[1]), 
            format("({:d},{:d},{:d})",  shape_node[0], shape_node[1], shape_node[2]));
        string shape_node_expected_str = AMREX_D_PICK(
            format("({:d})", nx+1), 
            format("({:d},{:d})", nx+1, ny+1), 
            format("({:d},{:d},{:d})", nx+1, ny+1, nz+1));
        Abort(format("Rincflo::NumpyNodeToCell. Bad shape of nodal array! Got {:s}, expected {:s}.\n", 
            shape_node_str.c_str(), shape_node_expected_str.c_str()));
    }

    // The data pointers to each array
    const Real* data_node = data_node_npy.const_data<Real>();
    Real* data_cell = data_cell_npy.data<Real>();

    // The upper bound on the offset to each axis for sampling nodal data
    size_t di_max = (dim > 0) ? 2 : 1;
    size_t dj_max = (dim > 1) ? 2 : 1;
    size_t dk_max = (dim > 2) ? 2 : 1;

    // Multiplicative factor - reciprocal of the number of touching nodal cells on each cell-centered cell
    // This is 1/4 in 2D and 1/8 in 3D
    Real factor = AMREX_D_PICK(0.5, 0.25, 0.125);

    // Iterate over the nodal and cell-centered arrays
    for (size_t i_cell=0; i_cell < nx; ++i_cell)
    {
        for (size_t j_cell=0; j_cell < ny; ++j_cell)
        {
            for (size_t k_cell=0; k_cell < nz; ++k_cell)
            {
                // The row index on the cell array
                int r_cell = NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i_cell, j_cell, k_cell));

                // Initialize cell value to zero
                data_cell[r_cell] = 0.0;

                // Iterate over touching nodal cells
                for (size_t di=0; di < di_max; ++di)
                {
                    size_t i_node = i_cell + di;
                    for (size_t dj=0; dj < dj_max; ++dj)
                    {
                        size_t j_node = j_cell + dj;
                        for (int dk=0; dk < dk_max; ++dk)
                        {
                            size_t k_node = k_cell + dk;
                            // The row index on the node array
                            int r_node = NumpyRowScalar(AMREX_D_DECL(nx+1, ny+1, nz+1), AMREX_D_DECL(i_node, j_node, k_node));
                            // Increment cell centered array
                            data_cell[r_cell] += data_node[r_node];
                        }   // for / dk
                    }   // for / dj
                }   // for / di
                // Re-scale the cell-centered data by 1/4 (2D) or 1/8 (3D)
                data_cell[r_cell] *= factor;
            }   // for / k_cell
        }   // for / j_cell
    }   // for / i_cell

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME    
}
