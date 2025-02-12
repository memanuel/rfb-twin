# *************************************************************************************************
# Solve for the state of charge field at the mass transport limit for a given velocity field.
# Michael S. Emanuel
# 2024-10-09
# *************************************************************************************************

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, coo_array
import scipy.sparse.linalg as spla
from pathlib import Path
import argparse
from dataclasses import dataclass
import time
from typing import Optional

# Local imports
from utils import SimType, Geometry, FlowSim, ReactSim, load_flow_sim_impl, load_react_sim_impl, make_P_str
from flow_op import make_time_step_op, calc_dt

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
# Physical constants
F: float = 96485.3329   # Faraday's constant in Coulombs per mole
ne: int = 2             # number of electrons
c0: float = 20.0E-3     # Total AQDS concentration in moles per liter
diff: float = 4.0E-6    # The diffusivity of AQDS in cm^2 / sec

# *************************************************************************************************
def calc_dir_npy(sim_type: SimType, pressure: int, stage:int) -> Path:
    """The directory with numpy simulation data for a given pressure and stage"""
    # The top level directory
    dir1: str = 'npy'
    # The second level directory
    if sim_type == SimType.flow:
        dir2 = '01_flow'
    elif sim_type == SimType.react:
        dir2 = '02_mass_transport'
    # The pressure string
    dir3: str = make_P_str(P=pressure)
    # The stage
    dir4: str = f'stage{stage:02d}'
    # The full path
    return Path(f'{dir1}/{dir2}/{dir3}/{dir4}')

# *************************************************************************************************
def load_flow_sim(pressure: int, stage: int, level: int) -> FlowSim:
    """Load the flow simulation output for a given pressure, stage and level"""
    sim_type: SimType = SimType.flow
    dir_npy: Path = calc_dir_npy(sim_type=sim_type, pressure=pressure, stage=stage)
    return load_flow_sim_impl(dir_npy=dir_npy, level=level)

# *************************************************************************************************
def load_react_sim(pressure: int, stage: int, level: int) -> ReactSim:
    """Load the reaction simulation output for a given pressure, stage and level"""
    sim_type: SimType = SimType.react
    dir_npy: Path = calc_dir_npy(sim_type=sim_type, pressure=pressure, stage=stage)
    return load_react_sim_impl(dir_npy=dir_npy, level=level)

# *************************************************************************************************
def calc_dir_out(pressure: int, stage:int) -> Path:
    """The directory to save the results of the steady state solver"""
    # The top level directory
    dir1: str = 'npy'
    # The second level directory
    dir2 = 'mass_transport'
    # The pressure string
    dir3: str = make_P_str(P=pressure)
    # The stage
    dir4: str = f'soc{stage:02d}'
    # The full path
    return Path(f'{dir1}/{dir2}/{dir3}/{dir4}')

# *************************************************************************************************
@dataclass 
class SolverOpt:
    """Options for solve_steady"""
    # Tolerance for convergence
    tol: float
    # The time to flow across the channel
    tau: float
    # The time step
    dt: float
    # Max steps
    max_step: int
    # Interval to report progress and check for convergence
    prog_int: int
    # Interval to save intermediate results
    save_int: int
    # Pressure in Pa
    # pressure: int
    # Output directory where results are saved as .npy files
    dir_out: Path
    # Whether to apply clipping of SOC to [0, 1] after each time step
    do_clip: bool
    # Initial guess for SOC; defaults to 0
    s: Optional[NumpyFloatArray] = None
    # Step at start; used when restarting
    step: int = 0

# *************************************************************************************************
def solve_steady(T: csr_array, velocity: NumpyFloatArray, geom: Geometry, solv_opt: SolverOpt) -> NumpyFloatArray:
    """
    Solve for steady state by explicit time stepping
    INPUTS:
        T: Time step operator; sparse matrix in CSR format
        velocity: Velocity field; 3D array with shape (nx, ny, nz, 3)
        geom: Geometry object with grid information
        solv_opt: Solver options
    """
    # Unpack the solver options
    tol: float = solv_opt.tol
    tau: float = solv_opt.tau
    dt: float = solv_opt.dt
    max_step: int = solv_opt.max_step
    prog_int: int = solv_opt.prog_int
    save_int: int = solv_opt.save_int
    dir_out: Path = solv_opt.dir_out
    do_clip: bool = solv_opt.do_clip
    s: Optional[NumpyFloatArray] = solv_opt.s
    step: int = solv_opt.step

    # Number of cells
    N: int = geom.N
    # Shape of the grid
    shape: tuple[int, int, int] = (geom.nx, geom.ny, geom.nz)

    # Mask for cells of different types; flatten to 1d
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()
    is_wire: NumpyBoolArray = geom.is_wire.flatten()
    is_solid: NumpyBoolArray = geom.is_solid.flatten()
    is_outlet: NumpyBoolArray = geom.is_outlet.flatten()

    # SOC on even steps
    s0: NumpyFloatArray = np.zeros(N, dtype=np.float64) if s is None else s.flatten().copy()
    # Impose SOC boundary conditions
    s0[is_wire] = 1.0
    s0[is_solid] = 1.0
    # SOC on odd steps
    s1: NumpyFloatArray = s0.copy()
    # The last SOC
    soc_last: NumpyFloatArray
    # Change in SOC for progress and early termination
    chng: NumpyFloatArray = np.zeros(N, dtype=np.float64)
    # Factor to inflate one step RMS change to make it dimensionless
    rms_mult: float = tau / dt

    # Velocity x component
    u: NumpyFloatArray = velocity[:, :, :, 0].flatten()
    # Velocity at the outlet
    u_out: NumpyFloatArray = u[is_outlet]
    # Sum of velocity at the outlet; used for utilization
    u_out_sum: float = float(np.sum(u_out))
    # Flow rate in cubic centimeters per second
    h: float = geom.h * 1.0E-4
    Q_out: float = u_out_sum * h * h

    # Current at 100% utilization
    Q_Lps: float = Q_out * 1.0E-3
    I_max_A: float = Q_Lps * c0 * ne * F
    I_max_mA: float = I_max_A * 1.0E3

    # Create output directory if it doesn't exist
    dir_out.mkdir(parents=True, exist_ok=True)
    # Save the time step operator
    np.save(dir_out / 'T.npy', T)

    # Open text file to write convergence data
    path_conv: Path = dir_out / 'convergence.csv'
    fh = open(path_conv, 'a')

    # Status
    print(f'Running up to {max_step} time steps...')
    print(f'Tolerance for convergence: {tol:0.3e}')
    print(f'Flow rate at outlet: {Q_out:0.6e} cm^3/sec')
    print(f'Maximum current of 20 mM AQDS at 100% utilization: {I_max_mA:0.6e} mA')
    print('RMS change in SOC per time step:')
    print('step : change    :  mean    : outlet  : utilization')
    fh.write('step,change,mean,outlet,utilization\n')

    # Utility function - get last SOC
    def get_last_soc(i: int) -> NumpyFloatArray:
        return s0 if (i % 2) == 0 else s1

    # Last step taken
    i: int = 0

    # Start timer
    t0: float = float(time.time())

    # Take time steps
    for i in range(step+1, max_step+1):
        if (i % 2) == 1:
            s1[:] = T.dot(s0)
            if do_clip:
                np.clip(s1, 0.0, 1.0, out=s1)
        else:
            s0[:] = T.dot(s1)
            if do_clip:
                np.clip(s0, 0.0, 1.0, out=s0)

        # Save intermediate results
        if (save_int > 0) and (i % save_int) == 0:
            soc_last = get_last_soc(i)
            path: Path = dir_out / f'soc_{i:06d}'
            np.save(path, soc_last)

        # Report progress
        if (i % prog_int) == 0:
            # Calculate the change in SOC
            chng[:] = s1 - s0
            # Ignore non-liquid cells
            # chng[~is_fluid] = 0.0
            # Calculate RMS difference scaled up for time factor
            rms_chng: float = rms_mult * np.sqrt(np.mean(np.square(chng)))
            # Mean SOC on liquid cells
            soc_last = get_last_soc(i)
            s_mean: float = float(np.mean(soc_last[is_fluid]))
            # Mean SOC at outlet
            s_out: float = float(np.mean(soc_last[is_outlet]))
            # Utilization; this is the SOC weighted by the velocity at the outlet
            util: float = float(np.sum(soc_last[is_outlet] * u_out) / u_out_sum)
            # Status
            print(f'{i:4d} : {rms_chng:0.3e} : {s_mean:0.6f} : {s_out:0.6f} : {util:0.6f}', flush=True)
            # Write to convergence file
            fh.write(f'{i:d},{rms_chng:0.6e},{s_mean:0.12f},{s_out:0.12f},{util:0.12f}\n')
            # Have we converged early?
            if rms_chng < tol:
                current: float = I_max_A * util
                print(f'Converged at step {i} with RMS change {rms_chng:0.3e}', flush=True)
                print(f'Mean SOC at outlet = {s_out:0.12f}.')
                print(f'Utilization = {util:0.12f}.')
                print(f'Current = {current:0.12f} mA.')
                break

    # Close the convergence file
    fh.close()

    # Report elapsed time
    t1: float = float(time.time())
    elapsed: float = t1 - t0
    avg_step_time: float = elapsed / (i+1)
    print(f'Elapsed time        : {elapsed:0.0f} seconds')
    print(f'Average step time   : {avg_step_time:0.3f} seconds')

    # The final SOC; convert back to 3D
    soc_last = get_last_soc(i).reshape(shape)

    # Save the last steady state and return it
    path: Path = dir_out / f'soc_{i:06d}'
    print(f'Saved steady state SOC to {path}.')
    np.save(path, soc_last)
    return soc_last

# *************************************************************************************************
def solve_steady_gmres(F: csr_array, geom: Geometry, s0: NumpyFloatArray) -> NumpyFloatArray:
    """
    Solve for steady state by solving a linear system Ax = b with the GMRES algorithm.
    INPUTS:
        F: Flow operator (advection plus diffusion); sparse matrix in CSR format
        geom: Geometry object with grid information
        s0: Initial guess for SOC in original 3D geometry.
    """
    # Extract masks from geometry
    is_inlet: NumpyBoolArray = geom.is_inlet.flatten()
    is_wire: NumpyBoolArray = geom.is_wire.flatten()
    # Mask of live cells
    is_live: NumpyBoolArray = (~is_inlet) & (~is_wire)
    # Total number of cells
    N: int = geom.N
    # Number of live cells
    L: int = int(np.sum(is_live))

    # Convert the original flow operator to COO format
    F_coo: coo_array = F.tocoo()
    # Unpack elements of the AD operator
    data_old: NumpyFloatArray = F_coo.data
    row_old: NumpyInt32Array
    col_old: NumpyInt32Array
    row_old, col_old = F_coo.coords # type: ignore

    # Index to map from original row number to sorted row number
    # This has the property that p2r[r] = p where r is the original row number and p is the sorted row number
    p2r = np.argsort(~is_live, kind='stable')
    # Index to map from sorted row number to original row number
    # This has the property that r2p[r] = p where r is the original row number and p is the sorted row number
    r2p = np.argsort(p2r, kind='stable')

    # Apply the index to map from original row number to sorted row number
    # row_old and col_old are sequences or original row numbers; they are mapped to sequences of new row numbers
    row_aug: NumpyInt32Array = r2p[row_old]
    col_aug: NumpyInt32Array = r2p[col_old]

    # Data type for all arrays
    dtype: np.dtype = np.float64 # type: ignore

    # Create a new operator A in the sorted row order; augmented coordinates with all N entries
    # A here is an abstract operator that appears in the equation Ax = b; it is NOT advection.
    data_aug: NumpyFloatArray = data_old
    coords_aug: tuple[NumpyInt32Array, NumpyInt32Array] = (row_aug, col_aug)
    shape_aug: tuple[int, int] = (N, N)
    A_aug_coo: coo_array = coo_array((data_aug, coords_aug), shape=shape_aug, dtype=dtype, copy=False)
    A_aug_coo.sum_duplicates()
    A_aug_coo.eliminate_zeros()
    # A_aug: csr_array = A_aug_coo.tocsr(copy=True)

    # Filter operator down to just the L live rows / columns
    is_live_ij = (row_aug < L) & (col_aug < L)
    row = row_aug[is_live_ij]
    col = col_aug[is_live_ij]
    data = data_aug[is_live_ij]
    # Create the new operator in the sorted row order
    coords: tuple[NumpyInt32Array, NumpyInt32Array] = (row, col)
    shape: tuple[int, int] = (L, L)
    A_coo: coo_array = coo_array((data, coords), shape=shape, dtype=dtype, copy=False)
    A_coo.sum_duplicates()
    A_coo.eliminate_zeros()
    A: csr_array = A_coo.tocsr(copy=True)

    # Input to original operator with boundary conditions
    s_bc_old: NumpyFloatArray = np.zeros(N, dtype=np.float64)
    s_bc_old[is_inlet] = 0.0
    s_bc_old[is_wire] = 1.0
    # Constant term of original operator from boundary conditions
    b_old: NumpyFloatArray = F.dot(s_bc_old)
    # Constant term in sorted row order
    b_aug: NumpyFloatArray = b_old[p2r]
    b: NumpyFloatArray = -b_aug[0:L]

    # Solve the linear system
    rtol: float = 1.0E-8
    atol: float = 1.0E-6
    maxiter: int = 3
    x0: NumpyFloatArray = s0.flatten()[p2r[0:L]].copy()

    # Report initial error
    ds: NumpyFloatArray = A.dot(x0) - b
    err_abs = np.linalg.norm(ds)
    err_rms: float = float(np.sqrt(np.mean(np.square(ds))))
    print(f'Error with initial guess:')
    print(f'Absolute    : {err_abs:0.3e}')
    print(f'RMS         : {err_rms:0.3e}')

    print(f'\nSolving for steady state with {L} rows and {F.nnz} entries...')
    print(f'atol = {atol:0.2e}, rtol={rtol:0.2e}, maxiter = {maxiter:d}.')
    x: NumpyFloatArray
    info: int
    x, info = spla.gmres(A, b, x0=x0, atol=atol, rtol=rtol, maxiter=maxiter) # type: ignore

    # Report results
    ds: NumpyFloatArray = A.dot(x) - b
    err_abs = np.linalg.norm(ds)
    err_rms: float = float(np.sqrt(np.mean(np.square(ds))))
    if info == 0:
        print(f'Solver converged successfully.')
    else:
        print(f'Warning: solver did not converge.')
    print(f'Error after solution:')
    print(f'Absolute    : {err_abs:0.3e}')
    print(f'RMS         : {err_rms:0.3e}')

    # Reassemble s into a 3d matrix
    is_wire_aug = is_wire[p2r]
    s_aug: NumpyFloatArray = np.zeros(N, dtype=np.float64)
    s_aug[0:L] = x
    s_aug[is_wire_aug] = 1.0
    s: NumpyFloatArray = s_aug[r2p]
    s3d: NumpyFloatArray = s.reshape(geom.shape)

    # # Decompose solution
    # s0_live = s0.copy()
    # s0_live[~is_live] = 0.0
    # s0_bc = s_bc_old.copy()

    # # DEBUG
    # s1: NumpyFloatArray = s0[p2r].copy()
    # ds0: NumpyFloatArray = AD.dot(s0)
    # ds0b: NumpyFloatArray = AD.dot(s0_live) + AD.dot(s0_bc) # type: ignore
    # ds1: NumpyFloatArray = A_aug.dot(s1)
    # # Recovered ds0
    # ds0r: NumpyFloatArray = ds1[p2r]

    # # Test the operator decomposition live plus boundary on augmented operator
    # is_wire_aug = is_wire[p2r]
    # s2_live = s1.copy()
    # s2_live[L:] = 0.0
    # s2_bc = np.zeros(N, dtype=np.float64)
    # s2_bc[is_wire_aug] = 1.0
    # ds2: NumpyFloatArray = A_aug.dot(s2_live) + A_aug.dot(s2_bc) # type: ignore
    # ds2b: NumpyFloatArray = A_aug.dot(s2_live) - b_aug

    # # Test operator deompisition live plus boundary on live operator
    # s3 = s2_live[0:L].copy()    
    # ds3: NumpyFloatArray = A.dot(s3) - b

    # # DEBUG
    # print(f'Sizes:')
    # print(f'L               : {L:12d}')
    # print(f'N               : {N:12d}')
    # print(f'AD.nnz          : {AD.nnz:12d}')
    # print(f'A.nnz           : {A.nnz:12d}')
    # print(f'\nRMS error of various vectors:')
    # print(f' s0     : {np.sqrt(np.sum(np.square(s0)/N)):0.3e}')
    # print(f' s1     : {np.sqrt(np.sum(np.square(s1)/N)):0.3e}')
    # print(f'ds0     : {np.sqrt(np.sum(np.square(ds0)/N)):0.3e}')
    # print(f'ds0b    : {np.sqrt(np.sum(np.square(ds0b)/N)):0.3e}')
    # print(f'ds1     : {np.sqrt(np.sum(np.square(ds1)/N)):0.3e}')
    # print(f'ds2     : {np.sqrt(np.sum(np.square(ds2)/N)):0.3e}')
    # print(f'ds2b    : {np.sqrt(np.sum(np.square(ds2b)/N)):0.3e}')
    # print(f'ds3     : {np.sqrt(np.sum(np.square(ds3)/N)):0.3e}')
    # # print(f'b       : {np.sqrt(np.sum(np.square(b)/N)):0.3e}')
    # print(f'ds2-ds0r: {np.sqrt(np.mean(np.square(ds0-ds0r)/N)):0.3e}')

    return s3d
    
# *************************************************************************************************
def report_soc_sim(soc_sim: NumpyFloatArray, velocity: NumpyFloatArray, geom: Geometry, T: csr_array, dt: float):
    """Report the simulated state of charge attributes"""
    # Set soc to 1 on solid cells; simulation writes them as NaN for plotting
    soc_sim[geom.is_solid] = 1.0

    # Mean SOC by cell type
    s1_liquid: float = float(np.mean(soc_sim[geom.is_fluid]))
    s1_wire: float = float(np.mean(soc_sim[geom.is_wire]))
    s1_solid: float = float(np.mean(soc_sim[geom.is_solid]))

    # Mean SOC at inlet and outlet
    s1_in: float = float(np.mean(soc_sim[geom.is_inlet]))
    s1_out: float = float(np.mean(soc_sim[geom.is_outlet]))

    # Flatten the SOC vector
    s1: NumpyFloatArray = soc_sim.flatten()

    # Mean velocity in x component at inlet
    u_bar: float = float(np.mean(velocity[0, :, :, 0]))
    # Nominal time to flow across the domain
    tau: float = (geom.nx * geom.h) / u_bar

    # Update the SOC
    s2: NumpyFloatArray = T.dot(s1)
    # The change in SOC on the time step
    chng: NumpyFloatArray = s2 - s1
    # Ignore non-liquid cells
    chng[~geom.is_fluid.flatten()] = 0.0
    # Calculate RMS difference
    rms_chng: float = np.sqrt(np.mean(np.square(chng)))
    # Difference scaled for time step
    rms_s: float = rms_chng * tau / dt

    print('\nMean SOC in loaded simulation result:')
    print(f'liquid      = {s1_liquid:0.6}')
    print(f'wire        = {s1_wire:0.6f}')
    print(f'solid       = {s1_solid:0.6f}')
    print(f'inlet       = {s1_in:0.6f}')
    print(f'outlet      = {s1_out:0.6f}')
    print('\nRMS Change in one time step')
    print(f'step        = {rms_chng:0.3e}')
    print(f'per flow    = {rms_s:0.3e}')

# *************************************************************************************************
def main():
    """Console program"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve for the state of charge field at the mass transport limit.')
    parser.add_argument('-P', '--pressure', type=int, help='Pressure in Pa (int)')
    parser.add_argument('-S', '--stage', type=int, help='Stage of the simulation (int)')
    parser.add_argument('-M', '--max_step', type=int, default=0, help='Maximum number of steps')
    parser.add_argument('-R', '--restart', type=int, default=0, help='Step to restart from')
    parser.add_argument('-C', '--cfl', type=float, default=0.50, help='CFL parameter (float, default 0.5)')
    parser.add_argument('-T', '--tol', type=float, default=1.0E-4, 
                        help='Convergence tolerance (RMS change in SOC) per flow time (float, default 1E-4)')
    parser.add_argument('--clip', action="store_true", help='Clip SOC to [0, 1]; default false')
    pa = parser.parse_args()

    # Set parameters to choose the flow simulation
    pressure: int = pa.pressure
    stage: int = pa.stage
    level: int = stage - 1
    cfl: float = pa.cfl
    tol: float = pa.tol
    max_step_input: int = pa.max_step
    restart: int = pa.restart
    do_clip: bool = pa.clip

    # Load the flow simulation
    sim: FlowSim = load_flow_sim(pressure=pressure, stage=stage, level=level)

    # Unpack velocity array
    velocity: NumpyFloatArray = sim.velocity
    # Unpack the geometry
    geom: Geometry = sim.geom
    # Unpack grid spacing
    h: float = sim.geom.h

    # Calculate the time steps
    dt: float = calc_dt(velocity=velocity, h=h, diff=diff, cfl=cfl)
    # Mean velocity in x component at inlet
    u_bar: float = float(np.mean(velocity[0, :, :, 0]))
    # Nominal time to flow across the domain
    tau: float = (geom.nx * h) / u_bar

    # Summary of command line inputs
    print('\nCommand Line Inputs:')
    print(f'pressure    = {pressure:d} Pa')
    print(f'stage       = {stage:d}')
    print(f'level       = {level:d}')
    print(f'max_step    = {max_step_input:6d}')
    print(f'restart     = {restart:6d}')
    print(f'CFL         = {cfl:0.3f}')
    print(f'tol         = {tol:0.3e}')
    print(f'clip SOC    = {do_clip}')
    # Summarize the geometry
    geom.summarize()

    # Build the advection operator
    # A: csr_array = make_advection_op(velocity=velocity, geom=geom, dt=dt)
    # Build the diffusion operator
    # D: csr_array = make_diffusion_op(geom=geom, diff=diff, dt=dt)
    # Build the flow operator in the original row order
    # F: csr_array = A + D
    # F.sum_duplicates()
    # F.eliminate_zeros()
    # Solve with GMRES
    # s1: NumpyFloatArray = solve_steady_gmres(F=F, geom=geom, s0=soc_sim)

    # Build the time step operator
    print('Building time step operator...')
    verbose: int = 0
    T: csr_array = make_time_step_op(velocity=velocity, geom=geom, diff=diff, dt=dt, verbose=verbose)

    # Options for the steady state solver
    dir_out: Path = calc_dir_out(pressure=pressure, stage=stage)
    max_step_auto: int = int(np.clip(8.0 * tau / dt, 50000, 100000))
    max_step: int = max_step_input if max_step_auto > 0 else max_step_auto
    prog_int: int = 10
    save_int: int = 5000

    # Summarize the flow simulation
    print('\nSimulation parameters:')
    print(f'h           = {geom.h*1.0E4:0.3f} microns')
    print(f'dt          = {dt:0.3e} seconds')
    print(f'u_bar       = {u_bar:0.3e} cm/sec')
    print(f'tau         = {tau:0.3e} seconds')
    print(f'tau/dt      = {tau/dt:.0f} steps / flow')
    if max_step_input == 0:
        print(f'Using automatic max_step = {max_step_auto:06d}.')

    # Load previous steady state SOC if available
    s: Optional[NumpyFloatArray] = None
    step: int = 0
    if restart > 0:
        try:
            path_soc = dir_out / f'soc_{restart:06d}.npy'
            s = np.load(path_soc)
            print(f'Restarting from {path_soc} at step {step:d}.')
            step = restart
        except OSError:
            print(f'Unable to load previous state from {path_soc}.')
    if step == 0:
        print(f'Starting simulation from scratch at step {step:d}.')

    # Create solv_opt object
    solv_opt: SolverOpt = SolverOpt(
        tol=tol, tau=tau, dt=dt, max_step=max_step, prog_int=prog_int, 
        save_int=save_int, dir_out=dir_out, do_clip=do_clip, s=s, step=step)
    
    # Solve for steady state
    ss: NumpyFloatArray = solve_steady(T=T, velocity=velocity, geom=geom, solv_opt=solv_opt)
    # Save the steady state SOC
    path: Path = dir_out / 'soc_steady.npy'
    np.save(path, ss)

# *************************************************************************************************
if __name__ == '__main__':
    main()
