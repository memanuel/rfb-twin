# *************************************************************************************************
# Construct linear operators for advection, diffusion, and time step.
# Michael S. Emanuel
# 2024-10-16
# *************************************************************************************************

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csr_array, identity

# Local imports
from utils import Geometry

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
def calc_dt(velocity: NumpyFloatArray, h: float, diff: float, cfl: float) -> float:
    """Calculate the time step from the velocity field and desired CFL parameter"""
    # Alias diffusion coefficient
    D: float = diff
    # Get the sum of velocity magnitudes
    v: float = np.nanmax(np.sum(np.abs(velocity), axis=-1))
    # Simple calculation, ignoring diffusion
    # dt_lin: float = cfl * h / v
    # Correct for diffusion; f(t) = v * t + [D * t]^1/2. We want f(dt) = cfl * h.
    # Convert this to a quadratic with the change of variables x = dt^2. Obtain
    # v * x^2 + D * x - cfl * h = 0
    x = (np.sqrt(D + 4.0 * cfl * v * h) - np.sqrt(D)) / (2.0 * v)
    dt = x * x
    # Report results
    # print('Time step calculation:')
    # print(f'v = {v:0.3e}')
    # print(f'dt_lin = {dt_lin:0.3e}')
    # print(f'dt_quad = {dt:0.3e}')
    return dt

# *************************************************************************************************
def make_advection_op(velocity: NumpyFloatArray, geom: Geometry, dt: float) -> csr_array:
    """
    Build an NxN advection matrix as a sparse matrix (COO format)
    INPUTS:
        velocity - velocity field; shape (nx, ny, nz, 3)
        geom - geometry object with grid information
        dt - time step
    """
    # Unpack the dimensions from the geometry
    nx: int = geom.nx
    ny: int = geom.ny
    nz: int = geom.nz
    # Shape of the grid
    grid_shape: tuple[int, int, int] = (nx, ny, nz)
    # Unpack the grid row numbers
    rr: NumpyInt32Array = geom.rr
    # Unpack the grid spacing in cm
    h: float = geom.h
    # Construct mask for live cells
    is_live: NumpyBoolArray = geom.is_fluid & ~geom.is_inlet

    # Number of rows in the SOC vector
    N: int = nx * ny * nz

    # Use (i, j, k) indexing for velocity and cell_type
    # Use (r, s) indexing for the advection operator A
    # The advection operator is given by
    # A = - (u dot grad) = - (u_x d/dx + u_y d/dy + u_z d/dz)

    # Upper bound on number of non-zero entries
    nnz_max: int = 6 * N
    # Counter for non-zero entries
    nnz: int = 0

    # An arbitrary entry in the advection matrix is at position (r, s) with value v
    # A[ row[p], col[p] ] = val[p]
    # Numpy array with row indices (r) of the non-zero entries
    row: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with column indices (s) of the non-zero entries
    col: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with values of the non-zero entries
    val: NumpyFloatArray = np.zeros(nnz_max, dtype=np.float64)
    # Current slice of the arrays
    slc: slice

    # Iterate over coordinates
    for c in range(3):
        # The number of slices in this coordinate, e.g. nx when c=0
        nc: int = grid_shape[c]
        # The velocity c-th components on this slice; shape e.g. (ny, nz) when c=0
        vc: NumpyFloatArray
        # Mask of live cells on this slice
        mask_live: NumpyBoolArray
        # The row index of the current slice; shape e.g. (ny, nz) when c=0
        r: NumpyInt32Array
        # The row index of the slice with x down; shape (ny, nz) when c=0
        s_lo: NumpyInt32Array
        # The row index of the slice with x up; shape (ny, nz) when c=0
        s_hi: NumpyInt32Array
        # The row index of the cell interacting with the current slice; shape e.g. (ny, nz) when c=0
        s: NumpyInt32Array
        # The diagonal entry of the advection operator on a slice; shape e.g. (ny, nz) when c=0
        diag_entry: NumpyFloatArray        

        # Iterate over slices of this coordinate
        for i in range(nc):
            # Get the velocity component and liquid mask for this slice
            if c == 0:
                vc = velocity[i, :, :, c]
                mask_live = is_live[i, :, :]
            elif c == 1:
                vc = velocity[:, i, :, c]
                mask_live = is_live[:, i, :]
            elif c == 2:
                vc = velocity[:, :, i, c]
                mask_live = is_live[:, :, i]

            # Value of i used in the stencil alongside either i_lo or i_hi; usually same as i except on the edges
            i_mi: int = int(np.clip(i, 1, nc-2))
            # Value of i to use when shifting x down and up
            i_lo: int = i_mi - 1
            i_hi: int = i_mi + 1
            # Calculate r, s_lo, s_hi based on the coordinate
            # r is the cell that is changing under advection. 
            # s_lo / s_hi are cells used in the stencil to compute ds/dx.
            if c == 0:
                r    = rr[i   , :, :]
                s_lo = rr[i_lo, :, :]
                s_hi = rr[i_hi, :, :]
            elif c == 1:
                r    = rr[:, i   , :]
                s_lo = rr[:, i_lo, :]
                s_hi = rr[:, i_hi, :]
            elif c == 2:
                r    = rr[:, :, i   ]
                s_lo = rr[:, :, i_lo]
                s_hi = rr[:, :, i_hi]

            # slice_c_lo: slice = slice(i_lo, i_lo+1)
            # slice_c_hi: slice = slice(i_hi, i_lo+1)

            # Mask of positive and negative velocity components
            mask_vel_pos: NumpyBoolArray = (vc > 0) # type: ignore
            mask_vel_neg: NumpyBoolArray = (vc < 0) # type: ignore
            # Mask of nonzero advection entries on live cells on this slice
            mask: NumpyBoolArray = (mask_vel_pos | mask_vel_neg) & mask_live

            # The number of entries with positive and negative velocity components
            nnz_slc: int = int(np.sum(mask))

            # Test whether velocity component is positive or negative. We need to use upwinding for stability.
            # When positive, we look to left; when negative, we look to right.


            # The main case is when we can look either to the left or the right depending on sign(u)
            if (0 < i) and (i < nc-1):
                # The row number of the cell interacting with cell r; in the direction dictated by upwinding
                s = s_lo * mask_vel_pos.astype(np.int32) + s_hi * mask_vel_neg.astype(np.int32)
                # Calculate the diagonal matrix entry for the advection operator
                diag_entry = -np.abs(vc) * (dt / h) # type: ignore
            # The case where we can only look to the right
            elif i == 0:
                s = s_hi
                diag_entry =  vc * (dt / h) # type: ignore
            # The case where we can only look to the left
            elif i == nc-1:
                s = s_lo
                diag_entry = -vc * (dt / h) # type: ignore

            # Set the diagonal entry for the advection operator
            slc = slice(nnz, nnz+nnz_slc)
            row[slc] = r[mask].flatten()
            col[slc] = r[mask].flatten()
            val[slc] = diag_entry[mask].flatten()
            nnz += nnz_slc
            # Set the off-diagonal entry for the advection operator
            slc = slice(nnz, nnz+nnz_slc)
            row[slc] = r[mask].flatten()
            col[slc] = s[mask].flatten()
            val[slc] = -diag_entry[mask].flatten()
            nnz += nnz_slc

    # Assemble the sparse matrix
    data: NumpyFloatArray = val[:nnz]
    coords: tuple[NumpyInt32Array, NumpyInt32Array] = (row[:nnz], col[:nnz])
    shape: tuple[int, int] = (N, N)
    dtype: np.dtype = np.float64 # type: ignore
    copy: bool = False
    A: coo_array = coo_array((data, coords), shape=shape, dtype=dtype, copy=copy)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A.tocsr()

# *************************************************************************************************
def make_diffusion_op(geom: Geometry, diff: float, dt: float) -> csr_array:
    """
    Build an NxN advection matrix as a sparse matrix (COO format)
    INPUTS:
        geom - geometry object with grid information
        diff - diffusion coefficient
        dt - time step
    """
    # Unpack the dimensions from the geometry
    nx: int = geom.nx
    ny: int = geom.ny
    nz: int = geom.nz
    # Shape of the grid
    grid_shape: tuple[int, int, int] = (nx, ny, nz)
    # Unpack the grid row numbers
    rr: NumpyInt32Array = geom.rr
    # Unpack the grid spacing
    h: float = geom.h
    # Construct mask for live cells
    is_live: NumpyBoolArray = geom.is_fluid & ~geom.is_inlet
    # Number of rows in the SOC vector
    N: int = nx * ny * nz

    # The diffusion operator is given by
    # D = diff * Laplacian = diff * (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)

    # The scale factor for diffusion operator entries in symmetric difference
    a: float = (diff * dt) / (h * h)
    # The scale factor for diffusion operator entries in one sided difference
    b: float = (diff * dt) / (h * h * h)

    # Upper bound on number of non-zero entries
    nnz_max: int = 10 * N
    # Counter for non-zero entries
    nnz: int = 0

    # An arbitrary entry in the advection matrix is at position (r, s) with value v
    # D[ row[p], col[p] ] = val[p]
    # Numpy array with row indices (r) of the non-zero entries
    row: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with column indices (s) of the non-zero entries
    col: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with values of the non-zero entries
    val: NumpyFloatArray = np.zeros(nnz_max, dtype=np.float64)
    # Current slice of the arrays
    slc: slice

    # Iterate over coordinates
    for c in range(3):
        # The number of slices in this coordinate
        nc: int = grid_shape[c]
        # Mask of liquid cells on this slice
        mask: NumpyBoolArray
        # The row index of the current slice; shape e.g. (ny, nz) when c=0
        r: NumpyInt32Array
        # The row index of the slice with x unshifted; middle of stencil
        s_mi: NumpyInt32Array
        # The row index of the slice with x shifted down 1; left of stencil
        s_lo: NumpyInt32Array
        # The row index of the slice with x shifted up 1; right of stencil
        s_hi: NumpyInt32Array
        # The row index of four columns used in one sided stencil at either edge
        s0: NumpyInt32Array
        s1: NumpyInt32Array
        s2: NumpyInt32Array
        s3: NumpyInt32Array

        # Iterate over slices of this coordinate
        for i in range(nc):
            # Value of i in the middle of the stencil; usually same as i except on the edges
            i_mi: int = int(np.clip(i, 1, nc-2))
            # Value of i for the left and right of the stencil
            i_lo: int = i_mi - 1
            i_hi: int = i_mi + 1
            # Calculate r, s_lo, s_mi, s_hi based on the coordinate
            # r is the unshifted cell. s_lo, s_mi, s_hi are cells at left, middle and right of three point stencil.
            if c == 0:
                mask = is_live[i, :, :]
                r    = rr[i   , :, :]
                s_lo = rr[i_lo, :, :]
                s_mi = rr[i_mi, :, :]
                s_hi = rr[i_hi, :, :]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[0, :, :]
                    s1 = rr[1, :, :]
                    s2 = rr[2, :, :]
                    s3 = rr[3, :, :]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[nc-1, :, :]
                    s1 = rr[nc-2, :, :]
                    s2 = rr[nc-3, :, :]
                    s3 = rr[nc-4, :, :]
            elif c == 1:
                mask = is_live[:, i, :]
                r    = rr[:, i   , :]
                s_lo = rr[:, i_lo, :]
                s_mi = rr[:, i_mi, :]
                s_hi = rr[:, i_hi, :]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[:, 0, :]
                    s1 = rr[:, 1, :]
                    s2 = rr[:, 2, :]
                    s3 = rr[:, 3, :]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[:, nc-1, :]
                    s1 = rr[:, nc-2, :]
                    s2 = rr[:, nc-3, :]
                    s3 = rr[:, nc-4, :]
            elif c == 2:
                mask = is_live[:, :, i]
                r    = rr[:, :, i   ]
                s_lo = rr[:, :, i_lo]
                s_mi = rr[:, :, i_mi]
                s_hi = rr[:, :, i_hi]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[:, :, 0]
                    s1 = rr[:, :, 1]
                    s2 = rr[:, :, 2]
                    s3 = rr[:, :, 3]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[:, :, nc-1]
                    s1 = rr[:, :, nc-2]
                    s2 = rr[:, :, nc-3]
                    s3 = rr[:, :, nc-4]

            # The number of entries on this slice (liquid cells only)
            nnz_slc: int = int(np.sum(mask))

            # The main case is when we can look either to the left or the right depending on sign(u)
            if (0 < i) and (i < nc-1):
                # The contribution to the diffusion operator from the low end of the stencil (i_lo with column s_lo)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_lo[mask].flatten()
                val[slc] = a
                nnz += nnz_slc
                # The contribution to the diffusion operator from the middle of the stencil (i_mi with column s_mi)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_mi[mask].flatten()
                val[slc] = -2.0 * a
                nnz += nnz_slc
                # The contribution to the diffusion operator from the high end of the stencil (i_hi with column s_hi)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_hi[mask].flatten()
                val[slc] = a
                nnz += nnz_slc

            # Apply the one sided stencil
            else:
                # The contribution to the diffusion operator f(x)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s0[mask].flatten()
                val[slc] =  2.0 * b
                # The contribution to the diffusion operator f(x+h) or f(x-h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s1[mask].flatten()
                val[slc] = -5.0 * b
                # The contribution to the diffusion operator f(x+2h) or f(x-2h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s2[mask].flatten()
                val[slc] =  4.0 * b
                # The contribution to the diffusion operator f(x+3h) or f(x-3h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s3[mask].flatten()
                val[slc] = -1.0 * b

    # Assemble the sparse matrix
    data: NumpyFloatArray = val[:nnz]
    coords: tuple[NumpyInt32Array, NumpyInt32Array] = (row[:nnz], col[:nnz])
    shape_mat: tuple[int, int] = (N, N)
    dtype: np.dtype = np.float64 # type: ignore
    copy: bool = False
    D: coo_array = coo_array((data, coords), shape=shape_mat, dtype=dtype, copy=copy)
    D.sum_duplicates()
    D.eliminate_zeros()
    return D.tocsr()

# *************************************************************************************************
def adjust_op_wire(A: csr_array, geom: Geometry) -> csr_array:
    """
    Adjust an operator matrix so that cells on the wire are also updated.
    Idea: for every entry in the operator matrix of the type (r, s), where 
    r is a liquid cell and s is a wire cell, add a corresponding entry (s, r) that conserves mass.
    This includes an adjustment factor for the volume ratio.
    INPUTS:
        A - operator matrix; shape (N, N)
        geom - geometry object with grid information
    RETURNS:
        B - adjusted operator matrix; shape (N, N)
    """
    # Unpack masks from the geometry
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()
    is_wire: NumpyBoolArray = geom.is_wire.flatten()
    # Unpack the volumes from the geometry
    volume: NumpyFloatArray = geom.volume.flatten()
    # Convert the original operator to COO format
    A_coo = A.tocoo(copy=True)
    # Unpack the coordinates
    rr_A: NumpyInt32Array
    ss_A: NumpyInt32Array
    rr_A, ss_A = A_coo.coords
    # Unpack the matrix entries of A
    val_rs: NumpyFloatArray = A_coo.data
    # Extract the original entries where the row is a fluid and the column is a wire cell
    mask: NumpyBoolArray = is_fluid[rr_A] & is_wire[ss_A]
    # Assemble corresponding coordinates and values for these wire entries
    rr: NumpyInt32Array = rr_A[mask].copy()
    ss: NumpyInt32Array = ss_A[mask].copy()
    val_old: NumpyFloatArray = val_rs[mask].copy()
    # The corresponding volumes of these cells
    vol_r: NumpyFloatArray = volume[rr]
    vol_s: NumpyFloatArray = volume[ss]
    # The new entries on the diagonal
    val_diag: NumpyFloatArray = -val_old * (vol_r / vol_s)
    # The new entries off the diagonal
    val_off: NumpyFloatArray = -val_diag
    
    # Create a CSR operator for the new diagonal entries on the wire
    data_diag: NumpyFloatArray = val_diag
    coords_diag: tuple[NumpyInt32Array, NumpyInt32Array] = (ss, ss)
    shape: tuple[int, int] = A.shape
    dtype: np.dtype = np.float64
    Aw_diag: csr_array = coo_array((data_diag, coords_diag), shape=shape, dtype=dtype, copy=False).tocsr(copy=True)

    # Create a CSR operator for the new off diagonal entries on the wire
    data_off: NumpyFloatArray = val_off
    coords_off: tuple[NumpyInt32Array, NumpyInt32Array] = (ss, rr)
    shape: tuple[int, int] = A.shape
    dtype: np.dtype = np.float64
    Aw_off: coo_array = coo_array((data_off, coords_off), shape=shape, dtype=dtype, copy=False).tocsr(copy=True)

    # Return the sum of the original operator and the new entries
    B: csr_array = A + Aw_diag + Aw_off
    B.sum_duplicates()
    B.eliminate_zeros()
    return B

# *************************************************************************************************
def make_time_step_op(velocity: NumpyFloatArray, geom: Geometry, diff: float, dt: float, verbose: int) -> csr_array:
    """
    Build an NxN time step operator matrix as a sparse matrix (CSR format)
    INPUTS:
        velocity - velocity field; shape (nx, ny, nz, 3)
        geom - geometry object with grid information
        diff - diffusion coefficient
        dt - time step
        verbose - level of status messages
    RETURNS:
        T - time step operator matrix; shape (N, N)        
    """
    # Calculate the number of cells
    N: int = geom.nx * geom.ny * geom.nz
    # Build the advection operator
    A: csr_array = make_advection_op(velocity=velocity, geom=geom, dt=dt)
    # Build the diffusion operator
    D: csr_array = make_diffusion_op(geom=geom, diff=diff, dt=dt)
    # The sparse identity operator
    I: csr_array = csr_array(identity(N))

    # Build the total T operator as a CSR matrix
    T: csr_array = I + A + D
    T.sum_duplicates()
    T.eliminate_zeros()

    # Status message if requested
    if verbose > 0:
        print(f'make_time_step_op()')
        print(f'Step Size:   h = {geom.h:0.hf} cm')
        print(f'Step Time:  dt = {dt:0.6f} sec')
        print(f'Diffusivity: D = {diff:0.6e} cm^2/sec')
        print('\nOperator nonzero entries:')
        print(f'Advection: nnz = {A.nnz}')
        print(f'Diffusion: nnz = {D.nnz}')
        print(f'Time Step: nnz = {T.nnz}')

    # Operator detail if requested
    if verbose > 1:
        summarize_op(Op=A.tocsr(copy=True), geom=geom, name='Advection')
        summarize_op(Op=D.tocsr(copy=True), geom=geom, name='Diffusion')

    # Convert to CSR format and return
    return T.tocsr(copy=True)

# *************************************************************************************************
def summarize_op(Op: csr_array, geom: Geometry, name: str):
    """Summarize properties of an operator"""
    # Convert the masks to 1d
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()

    # Consolidate entries and eliminate zeros    
    Op.sum_duplicates()
    Op.eliminate_zeros()

    # Take sum over rows
    row_sum: NumpyFloatArray = np.array(Op.sum(axis=1)).flatten()
    # Filter to liquid cells
    r: NumpyFloatArray = row_sum[is_fluid]

    # Count of rows
    row_width = Op.max(axis=1) - Op.min(axis=1)
    r_count: int = int(row_width.count_nonzero())

    # Summary statistics by row sum
    r_min: float = float(np.min(r))
    r_max: float = float(np.max(r))
    r_mean: float = float(np.mean(r))
    r_std: float = float(np.std(r))
    r_pct_01: float = float(np.quantile(r, 0.01))
    r_pct_99: float = float(np.quantile(r, 0.99))

    # Summary statistics by entry
    a = Op.data
    a_count: int = len(a)
    a_mean: float = float(np.mean(a))
    a_std: float = float(np.std(a))
    a_min: float = float(np.min(a))
    a_max: float = float(np.max(a))
    a_pct_01: float = float(np.quantile(a, 0.01))
    a_pct_99: float = float(np.quantile(a, 0.99))

    # Summary statistics by abs(entry)
    b = np.abs(a)
    b_mean: float = float(np.mean(b))
    b_std: float = float(np.std(b))
    b_min: float = float(np.min(b))
    b_max: float = float(np.max(b))
    b_pct_01: float = float(np.quantile(b, 0.01))
    b_pct_99: float = float(np.quantile(b, 0.99))

    # Report results
    print(f'\n{name} operator summary over row sum and row entry:')
    print(f"stat    : row       : entry      : abs(entry)")
    print(f'count   : {r_count:9d} : {a_count:9d}')
    print(f'mean    : {r_mean  :9.2e} : {a_mean  :9.2e} : {b_mean  :9.2e}')
    print(f'std     : {r_std   :9.2e} : {a_std   :9.2e} : {b_std   :9.2e}')
    print(f'min     : {r_min   :9.2e} : {a_min   :9.2e} : {b_min   :9.2e}')
    print(f'max     : {r_max   :9.2e} : {a_max   :9.2e} : {b_max   :9.2e}')
    print(f'pct 01  : {r_pct_01:9.2e} : {a_pct_01:9.2e} : {b_pct_01:9.2e}')
    print(f'pct 99  : {r_pct_99:9.2e} : {a_pct_99:9.2e} : {b_pct_99:9.2e}')

