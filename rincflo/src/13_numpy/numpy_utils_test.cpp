// ************************************************************************************************
void Rincflo::Numpy_ToMultiFab(
    const vector<NpyArray>& data_npy, MultiFab& mf_1g, int lev_np, int lev_mf, Real conversion_factor)
{
    #define FUNC_NAME "Rincflo::Numpy_ToMultiFab"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // The spatial dimension
    constexpr int dim = SpaceDim;

    // Numpy data on this level
    const NpyArray& data_npy_lev = data_npy[lev_np];

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

    // // The geometry object for this level
    // const auto& geom {Geom(lev_mf)};
    // // The domain object for the geometry
    // const auto& domain {geom.Domain()};
    // // Number of grid points along x, y, z axes of the MultiFab
    // const size_t nx_mf = dim > 0 ? domain.length(0) : 1;
    // const size_t ny_mf = dim > 1 ? domain.length(1) : 1;
    // const size_t nz_mf = dim > 2 ? domain.length(2) : 1;

    // AMReX refinement always doubles the number of cells on each additional level
    const int ratio {1 << (lev_mf - lev_np)};

    // The data array; constant
    const Real* data = data_npy_lev.const_data<Real>();
    // The cell type array; constant
    // const uint8_t* cell_type = cell_type_npy[lev].const_data<uint8_t>();

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
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i_mf, int j_mf, int k_mf) noexcept
            {
                // The i, j, k components corresponding to the numpy array
                int i_np {i_mf / ratio}, j_np {j_mf / ratio}, k_np {k_mf / ratio};
                // Row index for scalar entry
                int r {NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i_np, j_np, k_np))};
                // Read in scalar value if the value is good
                if (!isnan(data[r]))
                    {phi(i_mf, j_mf, k_mf) = data[r] * conversion_factor;}
            };
            ParallelFor(bx, func);
        }
        else if (is_vector_array)
        {
            // The writable vector field (F is a stand-in for a generic field)
            const auto& F = mf_1g.array(mfi);
            auto func = 
            [=, this] AMREX_GPU_DEVICE (int i_mf, int j_mf, int k_mf) noexcept
            {
                // The i, j, k components corresponding to the numpy array
                int i_np {i_mf / ratio}, j_np {j_mf / ratio}, k_np {k_mf / ratio};            
                // Row index for vector entry (first component)
                int rv {NumpyRowVector(AMREX_D_DECL(nx, ny, nz), nc, AMREX_D_DECL(i_np, j_np, k_np), 0)};
                // Write out vector field components in this grid cell
                for (int c = 0; c < nc; ++c) 
                {
                    if (!isnan(data[rv+c]))
                        {F(i_mf, j_mf, k_mf, c) = data[rv+c] * conversion_factor;}
                }
            };
            ParallelFor(bx, func);
        }
    } // for / mfi

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

