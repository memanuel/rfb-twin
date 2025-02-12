# *************************************************************************************************
# Solve for the state of charge field in the Nernst model using explicit time stepping.
# Michael S. Emanuel
# 2024-10-21
# *************************************************************************************************

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, identity, load_npz, save_npz
from scipy.special import expit
from pathlib import Path
from dataclasses import dataclass
import argparse
import time
import datetime
import socket
from typing import Optional

# Local imports
from utils import Geometry, FlowSim, make_P_str, make_V_str
from flow_op import calc_dt, make_time_step_op, adjust_op_wire, summarize_op
from mt_util import load_flow_sim

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
# Physical constants
from constants import R, T, F, V_T, ne, c0, diff, k0

# *************************************************************************************************
def calc_dir_out(pressure: int, voltage: int, stage:int) -> Path:
    """The directory to save the results of the steady state solver"""
    # The top level directory
    dir1: str = 'npy'
    # The second level directory
    dir2 = '03_nernst'
    # The pressure string
    dir3: str = make_P_str(P=pressure)
    # The voltage string
    dir4: str = make_V_str(V=voltage)
    # The stage
    dir5: str = f'stage{stage:02d}'
    # The full path
    return Path(f'{dir1}/{dir2}/{dir3}/{dir4}/{dir5}')

# *************************************************************************************************
@dataclass 
class SolverOpt:
    """Options for solve_nernst"""
    # Pressure in Pa
    pressure: int
    # Applied voltage in mV
    voltage: float
    # Kinetic rate constant in cm/s
    k0: float

    # Relaxation parameter
    omega: float
    # Tolerance for convergence
    tol: float
    # The time to flow across the channel
    tau: float
    # The time step in seconds
    dt: float
    # Max steps
    max_step: int
    # Interval to report progress and check for convergence
    prog_int: int
    # Interval to save intermediate results
    save_int: int

    # Output directory where results are saved as .npy files
    dir_out: Path
    # Initial guess for SOC; defaults to 0
    s0: Optional[NumpyFloatArray] = None
    # Initial guess for dimensionless overpotential; defaults to 0
    eta0: Optional[NumpyFloatArray] = None
    # Whether to load the last available step
    restart: bool = False
    # Step at start; used when restarting
    restart_step: int = 0

# *************************************************************************************************
def solve_nernst(T: csr_array, velocity: NumpyFloatArray, geom: Geometry, solv_opt: SolverOpt) \
        -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """
    Solve for steady state in the Nernst model by explicit time stepping
    INPUTS:
        T: Time step operator for advection and diffusion; sparse matrix in CSR format
           T does NOT include setting the SOC on wire cells depending on the overpotential
        velocity: Velocity field; 3D array with shape (nx, ny, nz, 3)
        geom: Geometry object with grid information
        solv_opt: Solver options
    RETURNS:
        soc_last: The steady state SOC field; shape (nx, ny, nz)
        overpot: The overpotential field; shape (nx, ny, nz) in millivolts, 0 on wire
    """
    # Unpack the solver options
    voltage: float = solv_opt.voltage
    k0: float = solv_opt.k0
    omega: float = solv_opt.omega
    tol: float = solv_opt.tol
    tau: float = solv_opt.tau
    dt: float = solv_opt.dt
    max_step: int = solv_opt.max_step
    prog_int: int = solv_opt.prog_int
    save_int: int = solv_opt.save_int
    dir_out: Path = solv_opt.dir_out
    s0_in: Optional[NumpyFloatArray] = solv_opt.s0
    eta0_in: Optional[NumpyFloatArray] = solv_opt.eta0
    restart: bool = solv_opt.restart
    step: int = solv_opt.restart_step

    # Dimensionless applied potential in terms of thermal voltage; both voltage and V_t are in millivolts
    V_app: float = (ne * voltage) / V_T

    # Number of cells
    N: int = geom.N
    # Shape of the grid
    shape: tuple[int, int, int] = (geom.nx, geom.ny, geom.nz)
    # Grid spacing in cm
    h: float = geom.h

    # Mask for cells of different types; flatten to 1d
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()
    is_wire: NumpyBoolArray = geom.is_wire.flatten()
    is_solid: NumpyBoolArray = geom.is_solid.flatten()
    is_outlet: NumpyBoolArray = geom.is_outlet.flatten()
    # The number of fluid cells
    n_fluid: int = int(np.sum(is_fluid))
    # The number of wire cells
    n_wire: int = int(np.sum(is_wire))
    # Area of wire cells
    area: NumpyFloatArray = geom.area.flatten()[is_wire]
    # Volume of wire cells
    volume: NumpyFloatArray = geom.volume.flatten()[is_wire]
    # Dimensionless quantity for solving for the instantaneous dimensionless overpotential eta
    alpha: NumpyFloatArray = volume / (2.0 * area * k0 * dt)
    # Complementary relaxation parameter
    omega_comp: float = 1.0 - omega

    # Subfunctions - load state variables
    def load_step(i: int) -> tuple[NumpyFloatArray, NumpyFloatArray]:
        """Load the state variables from a npz file"""
        path_npz: Path = dir_out / f'nernst_{i:06d}.npz'
        npz = np.load(path_npz)
        soc: NumpyFloatArray = npz['soc']
        eta: NumpyFloatArray = npz['eta']
        return soc, eta
    
    def load_last() -> tuple[NumpyFloatArray, NumpyFloatArray, int]:
        """Load the state variables from the last npz file"""
        glob_npz: str = 'nernst_*.npz'
        paths_npz: list[Path] = sorted(list(dir_out.glob(glob_npz)))
        if len(paths_npz) == 0:
            raise ValueError(f'No files found matching {dir_out / glob_npz}.')
        path_npz: Path = paths_npz[-1]
        print(f'Loading state from {path_npz}')
        npz = np.load(path_npz)
        soc: NumpyFloatArray = npz['soc']
        eta: NumpyFloatArray = npz['eta']
        step: int = int(path_npz.stem.split('_')[-1])
        return soc, eta, step

    def soc_eq(eta: NumpyFloatArray) -> NumpyFloatArray:
        """Calculate the equilibrium SOC given the overpotential"""
        return expit(V_app - eta)

    # Are we starting from the last available step?
    if restart:
        s0_in, eta0_in, step = load_last()
        print(f'Loading data from last step ({step:d}).')
    # If restart was specified but s0_in or eta0_in is missing, load the specified step
    elif (step > 0) and (s0_in is None or eta0_in is None):
        s0_load, eta0_load = load_step(step)
        s0_in = s0_load if s0_in is None else s0_in
        eta0_in = eta0_load if eta0_in is None else eta0_in
        print(f'Loading data from step {step:d}.')

    # SOC on even steps
    s0: NumpyFloatArray = np.zeros(N, dtype=np.float64) if s0_in is None else s0_in.flatten().copy()
    # Dimensionless overpotential on wire cells at even time steps; eta = eta_mV / (ne V_T)
    eta0: NumpyFloatArray = np.zeros(n_wire, dtype=np.float64) if eta0_in is None else eta0_in.flatten().copy()
    # Impose SOC boundary conditions
    s0[is_wire] = soc_eq(eta0)
    s0[is_solid] = 1.0
    # SOC on odd steps
    s1: NumpyFloatArray = s0.copy()
    # Dimensionless overpotential on wire cells at odd time steps
    eta1: NumpyFloatArray = eta0.copy()
    # Create output directory if it doesn't exist
    dir_out.mkdir(parents=True, exist_ok=True)
      
    def get_last_soc(i: int) -> NumpyFloatArray:
        """Get the last SOC on the wire"""
        return s0 if (i % 2) == 0 else s1

    def get_last_eta(i: int) -> NumpyFloatArray:
        """get the last overpotential on the wire (dimensionless)"""
        return eta0 if (i % 2) == 0 else eta1

    def save(i: int) -> None:
        """Save the state variables to a npz file"""
        path_npz: Path = dir_out / f'nernst_{i:06d}.npz'
        soc_last = get_last_soc(i)
        eta_last = get_last_eta(i)
        np.savez(path_npz, soc=soc_last, eta=eta_last, step=i)

    # The last SOC
    soc_last: NumpyFloatArray = get_last_soc(step+1)
    # The last overpotential, in dimensionless units of thermal voltage
    eta_last: NumpyFloatArray = np.zeros(n_wire, dtype=np.float64)
    # Negative of the change in SOC on the wire
    ds: NumpyFloatArray = np.zeros(n_wire, dtype=np.float64)
    # Geometric mean of s and 1-s on wire cells at even and odd time steps
    gms: NumpyFloatArray = np.sqrt(soc_last[is_wire] * (1.0 - soc_last[is_wire]))
    # Minimum value of gms; to avoid collapse of rate when s->0 or s->1
    gms_min: float = 1.0E-3
    # The new estimate of HALF the overpotential
    half_eta_new: NumpyFloatArray = np.zeros(n_wire, dtype=np.float64)

    # Change in SOC on the fluid cells
    chng: NumpyFloatArray = np.zeros(n_fluid, dtype=np.float64)
    # Factor to inflate one step RMS change to make it dimensionless
    rms_mult: float = tau / dt

    # Velocity x component
    u: NumpyFloatArray = velocity[:, :, :, 0].flatten()
    # Velocity at the outlet
    u_out: NumpyFloatArray = u[is_outlet]
    # Sum of velocity at the outlet; used for utilization
    u_out_sum: float = float(np.sum(u_out))
    # Flow rate in cubic centimeters per second
    Q_out: float = u_out_sum * h * h

    # Current at 100% utilization
    Q_Lps: float = Q_out * 1.0E-3           # Flow rate in liters per second
    I_max_A: float = Q_Lps * c0 * ne * F    # Maximum current in Amperes at 100% utilization
    I_max_mA: float = I_max_A * 1.0E3       # Maximum current in milliamperes at 100% utilization

    # Open text file to write convergence data
    path_conv: Path = dir_out / 'convergence.csv'
    with open(path_conv, 'a') as fh:
        # Status
        print(f'Running up to {max_step} time steps...')
        print(f'Tolerance for convergence: {tol:0.3e}')
        print(f'Flow rate at outlet: {Q_out:0.6e} cm^3/sec')
        print(f'Maximum current of 20 mM AQDS at 100% utilization: {I_max_mA:0.6e} mA')
        print('Time steps:')
        print('step : change    :  eta      : s_wire    :  s_fluid   :  s_outlet  : util_wire : util_out')
        # Write header row to convergence file if this is the first step (skip this on a restart)
        if step == 0:
            fh.write('step,change,eta,s_wire,s_fluid,s_outlet,util_wire,util_outlet\n')

        # Start timer
        t0: float = float(time.time())
        # Step counter; step i advances from s[i-1] to s[i]
        i: int
        # Take time steps
        for i in range(step+1, max_step+1):
            if (i % 2) == 1:            
                s1[:] = T.dot(s0)
                np.clip(s1, 0.0, 1.0, out=s1)
                ds = s0[is_wire] - s1[is_wire]
                np.maximum(np.sqrt(s0[is_wire] * (1.0 - s0[is_wire])), gms_min, out=gms)
                np.arcsinh((alpha * ds) / gms, out=half_eta_new)
                eta1[:] = omega_comp * eta0 + omega * (half_eta_new * 2.0)
                s1[is_wire] = soc_eq(eta1)
            else:
                s0[:] = T.dot(s1)
                np.clip(s0, 0.0, 1.0, out=s0)
                ds = s1[is_wire] - s0[is_wire]
                np.maximum(np.sqrt(s0[is_wire] * (1.0 - s0[is_wire])), gms_min, out=gms)
                np.arcsinh((alpha * ds) / gms, out=half_eta_new)
                eta0[:] = omega_comp * eta1 + omega * (half_eta_new * 2.0)
                s0[is_wire] = soc_eq(eta0)

            # Save intermediate results to npz
            if (save_int > 0) and (i % save_int) == 0:
                save(i=i)

            # Report progress
            if (i % prog_int) == 0:
                # Change in SOC on the whole domain
                chng[:] = s1[is_fluid] - s0[is_fluid]
                # Calculate RMS difference scaled up for time factor
                rms_chng: float = rms_mult * np.sqrt(np.mean(np.square(chng)))
                # Mean overpotential in millivolts
                eta_last = get_last_eta(i)
                eta_mean = np.mean(eta_last)
                op_mean = eta_mean * V_T / ne
                # Mean SOC on liquid and wire cells
                soc_last = get_last_soc(i)
                s_fluid: float = float(np.mean(soc_last[is_fluid]))
                s_wire: float = float(np.mean(soc_last[is_wire]))
                # Mean SOC at outlet
                s_out: float = float(np.mean(soc_last[is_outlet]))
                # Utilization on the wire; by summing source rate times volume on the cut cells
                util_wire: float = (2.0 * k0 / Q_out) * np.sum(area * gms * np.sinh(eta_last / 2.0))
                # Utilization at the outlet; this is the SOC weighted by the velocity at the outlet
                util_out: float = float(np.sum(soc_last[is_outlet] * u_out) / u_out_sum)
                # Current in milliamps on the wire
                current: float = I_max_mA * util_wire
                # DEBUG output
                # ds_dt = ds / dt
                # Status
                print(f'{i:4d} : {rms_chng:0.3e} : {eta_mean:0.6f} :  {s_wire:0.6f}  :  {s_fluid:0.6f}  :'
                      f'  {s_out:0.6f}  :  {util_wire:0.6f} :  {util_out:0.6f}', flush=True)
                # Write to convergence file
                fh.write(f'{i:d},{rms_chng:0.6e},{eta_mean:0.6f}{s_wire:0.12f},{s_fluid:0.12f},'
                         f'{s_out:0.12f},{util_wire:0.12f},{util_out:0.12f}\n')
                # Have we converged early?
                if rms_chng < tol:
                    print(f'Converged at step {i} with RMS change {rms_chng:0.3e}', flush=True)
                    break
    # Always save the final state
    save(i)

    # Report final statistics
    print(f'Mean eta             : {eta_mean:0.6f} (dimensionless)')
    print(f'Mean overpotential   : {op_mean:0.6f} (mV).')
    print(f'Mean SOC on wire     : {s_wire:0.12f}')
    print(f'Mean SOC in fluid    : {s_fluid:0.12f}')
    print(f'Mean SOC at outlet   : {s_out:0.12f}')
    print(f'Utilization (wire  ) : {util_wire:0.12f}.')
    print(f'Utilization (outlet) : {util_out:0.12f}.')
    print(f'Current (wire)       : {current:0.12f} mA.')
                
    # Report elapsed time
    t1: float = float(time.time())
    elapsed: float = t1 - t0
    avg_step_time: float = elapsed / (i+1)
    print(f'Elapsed time        : {elapsed:0.0f} seconds')
    print(f'Average step time   : {avg_step_time:0.3f} seconds')

    # The final SOC; convert back to 3D
    soc_last = get_last_soc(i).reshape(shape)
    # The final overpotential; save as 3D array in millivolts
    overpot: NumpyFloatArray = np.zeros(N)
    overpot[is_wire] = get_last_eta(i) * (V_T / ne)
    overpot = overpot.reshape(shape)
    return soc_last, overpot

# *************************************************************************************************
def main():
    """Console program"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve for the state of charge in the Nernst model.')
    parser.add_argument('-P', '--pressure', type=int, help='Pressure in Pa (int)')
    parser.add_argument('-V', '--voltage', type=int, help='Voltage in mV (int)')
    parser.add_argument('-S', '--stage', type=int, help='Stage of the simulation (int)')
    parser.add_argument('-C', '--cfl', type=float, default=0.50, help='CFL parameter (float, default 0.5)')
    parser.add_argument('-O', '--omega', type=float, default=1.0E-3, help='Relaxation parameter (float, default 1E-3)')
    parser.add_argument('-T', '--tol', type=float, default=1.0E-4, 
                        help='Convergence tolerance (RMS change in SOC) per flow time (float, default 1E-4)')
    parser.add_argument('-M', '--max_step', type=int, default=0, help='Maximum number of steps (int, default 0)')
    parser.add_argument('-R', '--restart', action='store_true',
                        help='Whether to restart from last step (bool, default False)')
    parser.add_argument('--restart_step', type=int, default=0, help='Specific step to restart from (int, default 0)')
    pa = parser.parse_args()

    # Set parameters to choose the flow simulation and set the applied voltage
    pressure: int = pa.pressure
    voltage: int = pa.voltage
    stage: int = pa.stage
    level: int = stage - 1
    cfl: float = pa.cfl
    omega: float = pa.omega
    tol: float = pa.tol
    max_step_input: int = pa.max_step
    restart: bool = pa.restart
    restart_step: int = pa.restart_step

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
    start_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname: str = socket.gethostname()
    print('nernst.py - Solve for the state of charge in the Nernst model.')
    print(f'Hostname  : {hostname:s}.')
    print(f'Start Time: {start_time:s}')
    print('\nCommand Line Inputs:')
    print(f'pressure        = {pressure:d} Pa')
    print(f'voltage         = {voltage:d} mV')
    print(f'stage           = {stage:d}')
    print(f'level           = {level:d}')
    print(f'max_step        = {max_step_input:6d}')
    print(f'restart         = {restart}')
    print(f'restart_step    = {restart_step:6d}')
    print(f'CFL             = {cfl:0.3f}')
    print(f'omega           = {omega:0.3e}')
    print(f'tol             = {tol:0.3e}')
    # Summarize the geometry
    geom.summarize()

    # Build the time step operator
    print('Building time step operator...')
    verbose: int = 0
    T: csr_array = make_time_step_op(velocity=velocity, geom=geom, diff=diff, dt=dt, verbose=verbose)
    # Add offsetting entries to wire cells
    T = adjust_op_wire(T, geom)
    # Report summary 
    I: csr_array = csr_array(identity(geom.N))
    F: csr_array = T - I
    summarize_op(F, geom, 'Flow operator with wire entries')

    # Additional options for the steady state solver
    dir_out: Path = calc_dir_out(pressure=pressure, voltage=voltage, stage=stage)
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

    # Create solv_opt object
    solv_opt: SolverOpt = SolverOpt(
        pressure=pressure, voltage=voltage, k0=k0, omega=omega,
        tol=tol, tau=tau, dt=dt, max_step=max_step, prog_int=prog_int, 
        save_int=save_int, dir_out=dir_out, restart=restart, restart_step=restart_step)
    
    # Solve for steady state
    soc: NumpyFloatArray 
    overpot: NumpyFloatArray
    soc, overpot = solve_nernst(T=T, velocity=velocity, geom=geom, solv_opt=solv_opt)

    # Save the steady state SOC and overpotential
    path_soc: Path = dir_out / 'soc_steady.npy'
    path_op: Path = dir_out / 'overpot_steady.npy'
    np.save(path_soc, soc)
    np.save(path_op, overpot)

# *************************************************************************************************
def test():
    """Run a test case for P=10, V=0, stage=2 starting from converged special model."""
    # Set parameters to choose the flow simulation and set the applied voltage
    pressure: int = 10
    voltage: int = 0
    stage: int = 2
    level: int = 1
    cfl: float = 0.5
    # omega: float = 1.0E-3
    omega: float = 0.0
    tol: float = 1.0E-3
    max_step_input: int = 10000
    # max_step_input: int = 100
    restart: bool = False
    restart_step: int = 0

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

    # Load time step operator
    path_T = Path('temp/T.npz')
    if not path_T.exists():
        # Build the time step operator
        print('Building time step operator...')
        verbose: int = 0
        T: csr_array = make_time_step_op(velocity=velocity, geom=geom, diff=diff, dt=dt, verbose=verbose)
        # Add offsetting entries to wire cells
        T = adjust_op_wire(T, geom)
        # Report summary 
        I: csr_array = csr_array(identity(geom.N))
        F: csr_array = T - I
        summarize_op(F, geom, 'Flow operator with wire entries')
        # Save time step operator
        save_npz(path_T, T)
    else:
        T: csr_array = load_npz(path_T)
        print(f'Loaded time step operator from file {path_T}.')

    # Additional options for the steady state solver
    dir_out: Path = calc_dir_out(pressure=pressure, voltage=voltage, stage=stage)
    max_step_auto: int = int(np.clip(8.0 * tau / dt, 50000, 100000))
    max_step: int = max_step_input if max_step_auto > 0 else max_step_auto
    prog_int: int = 10
    save_int: int = 5000

    # Load initial guess for SOC
    s0_in: NumpyFloatArray = np.load('07_react_special/P010/Vp000/n128/stage01/numpy_last/L0/soc.npy')
    # Load initial guess for overpotential
    overpot: NumpyFloatArray = np.load('07_react_special/P010/Vp000/n128/stage01/numpy_last/L0/overpot.npy')
    # eta0_all: NumpyFloatArray = (ne * overpot) / V_T
    eta0_in: NumpyFloatArray = (ne / V_T) * overpot[geom.is_wire]

    # Summarize the flow simulation
    print('\nSimulation parameters:')
    print(f'h           = {geom.h*1.0E4:0.3f} microns')
    print(f'dt          = {dt:0.3e} seconds')
    print(f'u_bar       = {u_bar:0.3e} cm/sec')
    print(f'tau         = {tau:0.3e} seconds')
    print(f'tau/dt      = {tau/dt:.0f} steps / flow')
    if max_step_input == 0:
        print(f'Using automatic max_step = {max_step_auto:06d}.')

    # Create solv_opt object
    solv_opt: SolverOpt = SolverOpt(
        pressure=pressure, voltage=voltage, k0=k0, omega=omega,
        tol=tol, tau=tau, dt=dt, max_step=max_step, prog_int=prog_int, 
        save_int=save_int, dir_out=dir_out, restart=restart, restart_step=restart_step,
        s0=s0_in, eta0=eta0_in)
    
    # Solve for steady state
    soc: NumpyFloatArray 
    overpot: NumpyFloatArray
    soc, overpot = solve_nernst(T=T, velocity=velocity, geom=geom, solv_opt=solv_opt)

    # Save the steady state SOC and overpotential
    path_soc: Path = dir_out / 'soc_steady.npy'
    path_op: Path = dir_out / 'overpot_steady.npy'
    np.save(path_soc, soc)
    np.save(path_op, overpot)

# *************************************************************************************************
if __name__ == '__main__':
    # main()
    test()
