// rincflo_numpy.H
// Fragment of Rincflo.H
// All the methods in Rincflo related to numpy checkpoints.
#pragma once
#include <rincflo_macros.H>

public:

// Write out numpy data files
void WriteNumpyScalars();
void WriteNumpyGeometry();
void WriteNumpyVelocity();
void WriteNumpyPressure();
void WriteNumpyConc();
void WriteNumpyCurrent();
void WriteNumpyAll();

// Read in numpy data files
void ReadNumpy();

// The row number in a numpy array for a scalar field with shape (nx, ny) in 2D or (nx, ny, nz) in 3D
inline int NumpyRowScalar(
    AMREX_D_DECL(int nx, int ny, int nz), AMREX_D_DECL(int i, int j, int k)) const noexcept
    {return (AMREX_D_PICK(i, ny*i + j, ny*nz*i + nz*j + k));}

// The row number in a numpy array for a vector field with shape (nx, ny, nc) in 2D or (nx, ny, nz, nc) in 3D
inline int NumpyRowVector(
    AMREX_D_DECL(int nx, int ny, int nz), int nc, AMREX_D_DECL(int i, int j, int k), int c = 0) const noexcept
    {return NumpyRowScalar(AMREX_D_DECL(nx, ny, nz), AMREX_D_DECL(i, j, k))*nc + c;}

// Get the shape of the numpy vectors
Rincflo::NumpyShapes GetNumpyShapes() const;

// The top level directory used to read in numpy flow data
string NumpyInputDir_flow() const;
// The top level directory used to read in numpy reaction data
string NumpyInputDir_react() const;
// The top level directory used to read in numpy data for the matching type as this simulation (flow or react)
string NumpyInputDir_active() const;
// The top level directory used to write out numpy data files
string NumpyOutputDir() const;

// Export the index and position data to numpy arrays
void MultiFab_IndexToNumpy(NpyArray& idx_npy, NpyArray& pos_npy, const MultiFab& mf_1g, int lev);
// Export the cell type for the top level (level 0) to a numpy array
void MultiFab_CellTypeToNumpy_top(NpyArray& cell_type_npy, const MultiFab& cell_type_mf_1g);
// Export the cell type to a numpy array
void MultiFab_CellTypeToNumpy(vector<NpyArray>& cell_type_npy, vector<NpyArray>& refinement_npy,
        const MultiFab& cell_type_mf_1g, int lev);

// Export the data in a MultiFab to a numpy array
void MultiFab_ToNumpy(vector<NpyArray>& data_npy, const MultiFab& mf_1g, int lev, Real conversion_factor=1.0);

// Overwrite the contents of a MultiFab with data from a numpy array
void Numpy_ToMultiFab(
    const vector<NpyArray>& data_npy, MultiFab& mf_1g, int lev, Real conversion_factor=1.0);

// Upsample data in a cell centered numpy array - populate fine level by duplicating coarse level
void NumpyUpsample(
    vector<NpyArray>& data_npy, const vector<NpyArray>& refinement_npy, 
    int lev, bool is_extensive, bool upsample_all);
// Filter data in a numpy array; cell_type_npy must be aligned with data_npy
void NumpyFilter(NpyArray& data_npy, const NpyArray& cell_type_npy, uint8_t mask);
// Convert a data in a nodal array to a cell-centered array
void NumpyNodeToCell(const NpyArray& data_node_npy, NpyArray& data_cell_npy);
