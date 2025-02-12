import numpy as np
from math import gcd
from pathlib import Path
import os
import subprocess
from dataclasses import dataclass

# Local imports
from rfb_utils import SimulationType, ModelType, SimulationType2ShortName, ModelType2ShortName, sim_success

# **********************************************************************************************************************
# Global variables

# Dimension of this campaign
dim: int = 3

# Dictionary with geometry filename; key = geometry number, value = filename
geometry_tbl: dict[int, str] = {
    1: 'one_wire.rod',
    2: 'two_wire.rod',
    3: 'lattice.rod',
}

# Dictionary with subdirectory for geometry; key = geometry number, value = subdirectory
subdir_tbl: dict[int, str] = {
    1: '01_OneWire',
    2: '02_TwoWire',
    3: '03_Lattice',
}

# Table - whether to upsample from the previous level
upsample_tbl: dict[int, bool] = {
    1: True,
    2: True,
    3: True,
}

# Inlet pressure in pascals
pressure: float = 100.0

# Array of applied voltages in millivolts and volts
# voltage_min: int = 0
# voltage_max: int = 200
# votage_step: int = 25
# voltages_mv = np.arange(voltage_min, voltage_max+votage_step, votage_step, dtype=np.int32)
voltages_mv = np.array([0], dtype=np.int32)

# Dictionary: key = geometry number, value = number of coarse levels
eb2_geo_required_coarsening_tbl: dict [int, int] = {
     1: 0,
     2: 0,
     3: 0,
}

# Dictionary; key = geometry number, value = AMR build_coarse_by_coarsening flag
amr_build_coarse_by_coarsening_tbl: dict[int, int] = {
    1: 0,
    2: 0,
    3: 0,
}

# Debug flag
debug: bool = False

# Absolute directory of the twin directory
abs_dir_twin: Path = Path('/ssd/Harvard/rfb/twin')
# Directory for the geometry files
abs_dir_geometry: Path = Path(abs_dir_twin, 'geometry')

# Directories for various convergence studies
# dir_cnvg_flow: Path  = Path('02_cnvg_flow')
dir_cnvg_flow: Path  = Path('04_cnvg_flow_mesh')
# dir_cnvg_react: Path = Path('03_cnvg_react')
dir_cnvg_react: Path = Path('05_cnvg_react_mesh')

# Directories for various reference simluations
dir_flow: Path = Path('06_flow')
dir_react: Path = Path('07_react')
dir_react_bv: Path = Path('08_react_bv')

# Absolute directories for the flow and reaction convergence simulations
abs_dir_cnvg_flow: Path = Path(abs_dir_twin, dir_cnvg_flow)
abs_dir_cnvg_react: Path = Path(abs_dir_twin, dir_cnvg_flow)

# List of simulation run types for each voltage
sim_types: list[SimulationType] = [
    SimulationType.Flow, SimulationType.React]

# The geometry size (Lx, Ly, Lz) of the domain; in microns
Lx: int = 1280
Ly: int = 640
Lz: int = 160

# The low end of the domain for flow simulations; in microns
lo_x: int = 0
lo_y: int = 0
lo_z: int = 0

# EB2: threshold for a small cut cell to be tagged as regular (all liquid)
eb2_small_volfrac: float = 1.0E-6

# How many levels of refinement? max_level=5 means 6 total levels indexed [0, 1, 2, 3, 4, 5]
# max_level: int = 2
# max_grid_levels: int = max_level+1

# Set AMR blocking factor; AmReX recommends 8
amr_block_factor_dflt: int = 16

# Set AMR error buffer (number of buffer cells around each tagged cell)
amr_n_error_buf: int = 1
amr_n_error_buf_flow: int = amr_n_error_buf
amr_n_error_buf_react: int = amr_n_error_buf

# Set AMR ngrow - default is 4
amr_ngrow: int = 4

# Should we treat multiply cut cells as covered?
eb2_geo_cover_multiple_cuts: int = 0

# Set max number of coarsening levels for MLMG solver
eb2_geo_max_coarsening: int = 6

# The thermal voltage in mv at room temperature
thermal_voltage_mv: float = 25.693
# Number of electrons in this reaction
n_electron: int = 2
# Nominal voltage
E0_diff_mv: float = 200.0
# State of charge at equilibrium
soc_eq: float = np.exp(n_electron * -E0_diff_mv / thermal_voltage_mv)

# Two electroactive species:
# A: AQDS    (oxidized)
# B: H2-AQDS (reduced)
# Combined concentration of all AQDS species (A+B)
conc: int = 20

# The interval for printing status updates to the console - flow simulations
interval_status_flow: int = 1
# The interval for printing status updates to the console - reaction simulations, Special model
interval_status_react_sp: int = 100
# The interval for printing status updates to the console - reaction simulations, Nernst model
interval_status_react_ne: int = 10

# **********************************************************************************************************************
def calc_dir_cnvg_flow(G: int, nz: int) -> Path:
    """
    Calculate top level simulation directory for a flow convergence simulation.
    INPUTS:
    G:              Number of the geometry; one of [1, 2, 3]
    nz:             Number of grid cells in the z direction
    """
    return Path(dir_cnvg_flow, subdir_tbl[G], f'n{nz:03d}')

# **********************************************************************************************************************
def calc_dir_cnvg_react(G: int, nz: int) -> Path:
    """
    Calculate top level simulation directory for a reaction convergence simulation.
    INPUTS:
    G:              Number of the geometry; one of [1, 2, 3]
    nz:             Number of grid cells in the z direction
    """
    return Path(dir_cnvg_react, subdir_tbl[G], f'n{nz:03d}')

# **********************************************************************************************************************
def calc_dir_flow(nz: int) -> Path:
    """
    Calculate top level simulation directory for a reference flow simulation.
    INPUTS:
    nz:             Number of grid cells in the z direction
    """
    return Path(dir_flow, f'n{nz:03d}')

# **********************************************************************************************************************
def calc_dir_react_bv(nz: int) -> Path:
    """
    Calculate top level simulation directory for a reference Butler-Volmer reaction simulation.
    INPUTS:
    nz:             Number of grid cells in the z direction
    """
    return Path(dir_react, f'n{nz:03d}')

# **********************************************************************************************************************
@dataclass
class SimulationSpec:
    """Class defining a simulation specification"""
    # The type of simulation (Flow or React)
    sim_type: SimulationType
    # The geometry number
    G: int
    # The pressure in pascals
    P: float
    # The voltage in millivolts
    V: int
    # The total concentration of AQDS in millimolar
    conc: int
    # The state of charge at the inlet
    soc_in: float
    # The number of grid cells in the z direction at the coarsest level
    nz: int

# **********************************************************************************************************************
@dataclass
class SimulationScheduleEntry:
    """Class with one entry on a schedule for running a simulation in stages"""
    # The number of this stage of the simulation campaign, starting from 1
    stage_num: int
    # The simulation model to run in this stage
    model_type: ModelType
    # The number of levels of refinement to use
    grid_levels: int
    # The minimum number of steps for this stage
    min_step: int
    # The maximum number of steps for this stage
    max_step: int
    # The fixed stop time to use
    stop_time: float
    # Is this simulation being run to steady state?
    is_steady_state: bool
    # The CFL parameter to use
    cfl: float
    # Maximum consumption of reactant in one time step - initial setting
    max_consumption: float
    # Maximum consumption of reactant in one time step - global maximum during during adaptive tuning
    max_consumption_limit: float
    # Maximum overpotential
    overpot_max: float
    # Nernst omega
    nernst_omega: float
    # Maximum number of stagnant steps before simulation terminates early
    stagnant_steps_max: int
    # Tolerance (for flow or reaction solver)
    tol: float
    # Directory for amrex checkpoint
    amrex_input_dir: str
    # First level to be upsampled during restart
    upsample_from: int
    # Interval between status updates printed to console
    interval_status: int
    # Interval between AMReX checkpoints for the active simulation
    interval_amrex: int

    # Set default values
    def __post_init__(self):
        pass

# **********************************************************************************************************************
def make_inputs(sim_dir: Path, sim_spec: SimulationSpec, sched_entry: SimulationScheduleEntry, extra_macros: str):
    """
    Make the input3D files
    INPUTS:
    sim_dir:                Directory for this simulation
    sim_spec:               A simulation specification
    entry:                  A simulation schedule entry
    extra_macros:           String of extra macros passed to gpp
    """
    # Unpack the simulation specification
    sim_type: SimulationType = sim_spec.sim_type
    # The geometry number
    G: int = sim_spec.G
    # The pressure at the inlet
    pressure: float = sim_spec.P
    # The applied voltage in millivolts
    voltage_mv: int = sim_spec.V
    # The number of grid cells in the z direction
    grid_z: int = sim_spec.nz
    # The total concentration of AQDS in millimolar
    conc: int = sim_spec.conc
    # The state of charge at the inlet
    soc_in: float = sim_spec.soc_in

    # Unpack the schedule entry
    stage_num: int = sched_entry.stage_num
    model_type: ModelType = sched_entry.model_type
    grid_levels: int = sched_entry.grid_levels
    min_step: int = sched_entry.min_step
    max_step: int = sched_entry.max_step
    stop_time: float = sched_entry.stop_time
    is_steady_state: bool = sched_entry.is_steady_state
    cfl: float = sched_entry.cfl
    max_consumption: float = sched_entry.max_consumption
    max_consumption_limit: float = sched_entry.max_consumption_limit
    nernst_omega: float = sched_entry.nernst_omega
    overpot_max: float = sched_entry.overpot_max
    stagnant_steps_max: int = sched_entry.stagnant_steps_max
    tol: float = sched_entry.tol
    amrex_input_dir: str = sched_entry.amrex_input_dir
    upsample_from: int = sched_entry.upsample_from
    interval_amrex: int = sched_entry.interval_amrex
    interval_status: int = sched_entry.interval_status

    # The directory for this stage
    stage_dir: Path = Path(sim_dir, f'stage{stage_num:02d}')
    # Make the stage directory if necessary
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Short name of this simulation type
    sim_type_name: str = SimulationType2ShortName[sim_type]
    # Short name of this model type
    model_type_name: str = ModelType2ShortName[model_type]

    # This work occurs in the cfg directory
    os.chdir('cfg')

    # The name of the gpp input file to build the desired input file, e.g. make_flow.gpp, make_nernst.gpp, make_special.gpp
    fname_gpp: str = f'make_{model_type_name}.gpp'

    # The name of the input file for this stage, e.g. 01_flow/Q040/inputs.cfg
    fname_cfg = Path('..', stage_dir, f'inputs.cfg')

    # The voltage in volts
    voltage = voltage_mv * 1.0E-3

    # The two concentrations at the inlet
    conc_AQDS = conc * 1.0
    conc_A = conc_AQDS * (1.0 - soc_in)
    conc_B = conc_AQDS * soc_in
    # The concentrations for initial conditions; match inlet conditions
    conc_A_ic = conc_A
    conc_B_ic = conc_B

    # Set various flags
    # Is this a flow simulation?
    is_flow: bool = (sim_type == SimulationType.Flow)
    # Is this a reaction simulation?
    is_react: bool = (sim_type == SimulationType.React)
    # Flags for specific type of reaction simulations
    is_nernst: bool = (model_type == ModelType.Nernst)
    is_special: bool = (model_type == ModelType.Special)
    is_BV: bool = (model_type == ModelType.ButlerVolmer)
    # Is this a reaction with voltage?
    is_echem: bool = (is_nernst or is_special or is_BV)
    # Is this a cold start?  That means we haven't run any earlier stages of this type (flow or react)
    is_cold_start: bool = (stage_num == 1)

    # Geometry calculations
    # Dimensions of rectangle in meters
    Lx_m: float = Lx * 1.0E-6
    Ly_m: float = Ly * 1.0E-6
    Lz_m: float = Lz * 1.0E-6
    # Number of grid cells in each dimension at level 0
    grid_x: int = (Lx * grid_z) // Lz
    grid_y: int = (Ly * grid_z) // Lz

    # High end of rectangle in microns
    hi_x: int = lo_x + Lx
    hi_y: int = lo_y + Ly
    hi_z: int = lo_z + Lz

    # Low end of rectangle in meters
    lo_x_m: float = lo_x * 1.0E-6
    lo_y_m: float = lo_y * 1.0E-6
    lo_z_m: float = lo_z * 1.0E-6

    # High end of rectangle in meters
    hi_x_m: float = hi_x * 1.0E-6
    hi_y_m: float = hi_y * 1.0E-6
    hi_z_m: float = hi_z * 1.0E-6

    # Set grid size in microns
    grid_size: float = Lz / grid_z

    # Selected amr_n_error_buf for this type of simulation
    amr_n_error_buf: int = 0
    if is_flow:
        amr_n_error_buf = amr_n_error_buf_flow
    if is_react:
        amr_n_error_buf = amr_n_error_buf_react

    # The directory with the geometry file; as a relative path
    geometry_dir: Path = Path('../../../../geometry')
    # geometry_dir: Path = abs_dir_geometry.relative_to(abs_dir_cnvg_flow)
    # The name of the geometry file
    geometry_file: str = geometry_tbl[G]
    # The full path to the geometry file as string
    geometry_path: str = Path(geometry_dir, geometry_file).as_posix()

    # Set number of coarsening levels for MLMG solver; always at least grid_levels-1
    eb2_geo_required_coarsening: int = max(eb2_geo_required_coarsening_tbl[G], grid_levels-1)

    # Set build_coarse_by_coarsening flag
    amr_build_coarse_by_coarsening: int = amr_build_coarse_by_coarsening_tbl[G]

    # Set the blocking factor to be a divisor of the number of grid cells in the z direction
    amr_block_factor = gcd(grid_z, amr_block_factor_dflt)

    # gpp macros for geometry and grid configuration
    macros: str = ''
    macros += f' -DSIM_TYPE={sim_type_name} -DMODEL_TYPE={model_type_name}'
    macros += f' -DLO_X_M={lo_x_m:0.6f} -DLO_Y_M={lo_y_m:0.6f} -DLO_Z_M={lo_z_m:0.6f}'
    macros += f' -DHI_X_M={hi_x_m:0.6f} -DHI_Y_M={hi_y_m:0.6f} -DHI_Z_M={hi_z_m:0.6f}'
    macros += f' -DLX_M={Lx_m:0.6f} -DLY_M={Ly_m:0.6f} -DLZ_M={Lz_m:0.6f}'
    macros += f' -DGRID_X={grid_x:d} -DGRID_Y={grid_y:d} -DGRID_Z={grid_z:d}'
    macros += f' -DGRID_LEVELS={grid_levels:d}'
    macros += f' -DGRID_SIZE={grid_size:0.6f}'
    macros += f' -DAMR_N_ERROR_BUF={amr_n_error_buf:d}'
    macros += f' -DAMR_BLOCK_FACTOR={amr_block_factor:d}'
    macros += f' -DAMR_NGROW={amr_ngrow:d}'
    macros += f' -DAMR_BUILD_COARSE_BY_COARSENING={amr_build_coarse_by_coarsening:d}'
    macros += f' -DEB2_GEO_REQUIRED_COARSENING={eb2_geo_required_coarsening:d}'
    macros += f' -DEB2_GEO_MAX_COARSENING={eb2_geo_max_coarsening:d}'
    macros += f' -DEB2_GEO_COVER_MULTIPLE_CUTS={eb2_geo_cover_multiple_cuts:d}'
    macros += f' -DEB2_GEO_FILENAME="{geometry_path:s}"'
    macros += f' -DEB2_SMALL_VOLFRAC={eb2_small_volfrac:.3E}'

    # gpp macros for lattice, flow rate, concentrations, applied voltage
    macros += f' -DPRESSURE_INLET={pressure:0.6f}'
    macros += f' -DC_AQDS={conc_AQDS:0.1f} -DC_A={conc_A:0.16f} -DC_B={conc_B:0.16f}'
    macros += f' -DC_A_IC={conc_A_ic:0.16f} -DC_B_IC={conc_B_ic:0.16f}'
    if is_echem:
        macros += f' -DVOLTAGE_MV={voltage_mv:d} -DVOLTAGE={voltage:0.6f}'

    # The simulation stage
    macros += f' -DSTAGE={stage_num}'

    # Parameters depending on whether this is a restart
    reinit_all_conc: int
    reinit_some_conc: int
    reinit_epot_L: int
    reinit_epot_S: int
    if is_cold_start:
        # This is a "cold start" starting either from scratch (flow) or a different type of simulation
        restart_dir = amrex_input_dir if is_react else '""'
        reinit_all_conc = 1
        reinit_some_conc = 0
        reinit_epot_L = 1
        reinit_epot_S = 1
    else:
        # This is a "warm start" continuing an existing simulation
        restart_dir: str = amrex_input_dir
        # HACK this to 1 temporarily for cnvg_react_mesh only
        reinit_all_conc = 1
        reinit_some_conc = 0
        reinit_epot_L = 0
        reinit_epot_S = 0

    # restart the reaction counter on every stage
    react_zero_step_counter: int = 1

    # Set the numpy flags; numpy input directories are passed with the schedule inputs
    numpy_read_flag: int = 0
    numpy_write_flag: int = 1

    # Set number of initial iterations in flow simulation
    fii: int = 4 if is_flow else 0
    # only do this on a cold start of a flow simulation, when not reading from numpy
    # fii: int = 4 if (is_cold_start and is_flow and not numpy_read_flag) else 0
    macros += f' -DFLOW_INITAL_ITERATIONS={fii:d}'

    # Macros to configure the restart directory
    if is_flow:
        macros += f' -DRESTART_DIR_FLOW={restart_dir:s}'
    elif is_nernst:
        macros += f' -DRESTART_DIR_NERNST={restart_dir:s}'
        macros += f' -DREINIT_ALL_CONC_NERNST={reinit_all_conc:d}'
        macros += f' -DREINIT_SOME_CONC_NERNST={reinit_some_conc:d}'
    elif is_special:
        macros += f' -DRESTART_DIR_SPECIAL={restart_dir:s}'
        macros += f' -DREINIT_ALL_CONC_SPECIAL={reinit_all_conc:d}'
        macros += f' -DREINIT_SOME_CONC_SPECIAL={reinit_some_conc:d}'
    elif is_BV:
        macros += f' -DRESTART_DIR_BV={restart_dir:s}'
        macros += f' -DREINIT_ALL_CONC_BV={reinit_all_conc:d}'
        macros += f' -DREINIT_SOME_CONC_BV={reinit_some_conc:d}'
        macros += f' -DREINIT_EPOT_L={reinit_epot_L:d}'
        macros += f' -DREINIT_EPOT_S={reinit_epot_S:d}'

    # Macro for the first directory to upsample from
    macros += f' -DUPSAMPLE_FROM={upsample_from:d}'

    # Macros to configure numpy operations and restart
    macros += f' -DNUMPY_WRITE_FLAG={numpy_write_flag:d}'
    macros += f' -DNUMPY_READ_FLAG={numpy_read_flag:d}'
    macros += f' -DNUMPY_VERBOSE=0'

    # Macros to configure the flow CFL parameter and number of steps
    if is_flow:
        macros += f' -DFLOW_CFL={cfl:0.6f}'
        macros += f' -DFLOW_MIN_STEP={min_step}'
        macros += f' -DFLOW_MAX_STEP={max_step}'
        # Placeholders for reaction CFL and steps
        macros += f' -DREACT_CFL=0.000 -DREACT_INT=0 -DREACT_MIN_STEP=0 -DREACT_MAX_STEP=0'
    else:
        macros += f' -DREACT_MIN_STEP={min_step}'
        macros += f' -DREACT_MAX_STEP={max_step}'
        macros += f' -DREACT_CFL={cfl:0.6f}'
        # Placeholder values for the flow simulation when we run a reaction
        macros += f' -DFLOW_MAX_STEP=0 -DFLOW_MIN_STEP=0 -DFLOW_CFL=0.000 -DFLOW_INT=100'

    # Is this simulation to steady state?
    macros += f' -DSTEADY_STATE={is_steady_state:d}'

    # Set the stop time; will be used only if it has a positive value
    macros += f' -DSTOP_TIME={stop_time:0.9f}'

    # Interval for status updates to console
    macros += f' -DSTATUS_INT={interval_status:d}'

    # Macros to set the interval for amrex checkpoints
    if is_flow:
        macros += f' -DFLOW_INT={interval_amrex:d}'
        macros += f' -DREACT_INT=0'
    else:
        macros += f' -DREACT_INT={interval_amrex:d}'
        macros += f' -DFLOW_INT=0'

    # Should the reaction step counter be reset to zero?
    macros += f' -DREACT_ZERO_STEP_COUNTER={react_zero_step_counter}'

    # Optionally override the flow tolerance; default 1.0E-6 set in cfg/00_setup.gpp
    if is_flow:
        macros += f' -DFLOW_TOL={tol:6.3E}'

    # Set the concentration tolerance, max consumption and max overpotential for reaction models
    if is_react:
        macros += f' -DREACT_TOL_SOC={tol:8.2E}'
        macros += f' -DMAX_CONSUMPTION={max_consumption:12.10f}'
        macros += f' -DMAX_CONSUMPTION_LIMIT={max_consumption_limit:12.10f}'
        macros += f' -DOVERPOT_MAX={overpot_max:8.6f}'
        macros += f' -DSTAGNANT_STEPS_MAX={stagnant_steps_max:d}'

    # Special settings for Nernst model
    if is_nernst:
        macros += f' -DNERNST_OMEGA={nernst_omega:8.6f}'

    # Add in the extra macros and clean up the macro string
    macros += extra_macros
    macros = macros.strip()

    # Build the configuration file using gpp
    command_string = f'gpp {macros} -x {fname_gpp} > {fname_cfg}'
    if debug:
        print('********************************************************************************')
        print(command_string)
    subprocess.run(command_string, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

    # Get back to the main directory for this study
    os.chdir('..')

# **********************************************************************************************************************
def make_schedule_flow(max_levels: int, do_upsample: bool) -> list[SimulationScheduleEntry]:
    """
    Make a schedule for the flow simulations.
    INPUTS:
        max_levels:    The maximum number of grid levels
        do_upsample:        Should we upsample from the previous level?
    OUTPUTS:
        sched:  A list of SimulationScheduleEntry objects
    """
    # The model_type is always flow
    model_type: ModelType = ModelType.Flow
    # Minimum number of steps for each stage
    min_step: int = 0
    # Maximum steps for all the stages
    max_step: int = 64000
    # Stop time for each stage
    stop_time: float = 0.005
    # Is this to steady state?
    is_steady_state: bool = False
    # Use a moderately aggressive CFL for all the stages
    cfl: float = 0.50
    # Placeholder values for parameters only used in reaction stages
    max_consumption: float = 0.0
    max_consumption_limit: float = 0.0
    nernst_omega: float = 0.0
    overpot_max: float = 0.0
    # Maximum number of stagnant steps
    stagnant_steps_max: int = 1000

    # Set a tolerance for each stage
    # tol_tbl: dict[int, float] = {1: 5.0E-9, 2: 1.0E-8, 3: 1.0E-7, 4: 1.0E-6, 5: 1.0E-5}
    tol_dflt: float = 1.0E-9
    tol_tbl: dict[int, float] = {i+1: tol_dflt for i in range(max_levels)}

    # The status interval
    interval_status: int = interval_status_flow
    # The intervals bewteen checkpoints; don't use checkpoints, just save at end
    interval_amrex: int = max_step

    # Initialize the schedule with an empty list
    sched = list()

    # Iterate over the requested stages; stages start from 1, levels start from 0
    for i in range(1, max_levels+1):
        # Set the arguments for this stage
        stage_num: int = i
        grid_levels: int = i
        lev: int = stage_num - 1
        upsample_from: int = lev - 1 if do_upsample else 0
        tol: float = tol_tbl[stage_num]
        # The first entry is a cold start of the flow simulation with only one level of refinement
        # The remaining entries advance to the next level of refinement
        amrex_input_dir: str = f'../stage{i-1:02d}/chk_last' if (do_upsample and lev > 0) else '""'
        # Build this entry in the schedule
        sched.append(SimulationScheduleEntry(
            stage_num=stage_num, model_type=model_type, grid_levels=grid_levels,
            min_step=min_step, max_step=max_step, stop_time=stop_time, is_steady_state=is_steady_state,
            cfl=cfl, max_consumption=max_consumption, max_consumption_limit=max_consumption_limit,
            nernst_omega=nernst_omega, overpot_max=overpot_max,
            stagnant_steps_max=stagnant_steps_max, tol=tol,
            amrex_input_dir=amrex_input_dir, upsample_from=upsample_from,
            interval_status=interval_status, interval_amrex=interval_amrex))

    return sched

# **********************************************************************************************************************
def make_schedule_react(max_levels: int, flow_dir: Path) -> list[SimulationScheduleEntry]:
    """
    Make a schedule for the reaction simulations.
    INPUTS:
        max_levels:     The maximum number of grid levels
        flow_dir:       Path object with the directory of the matching flow simulation
    OUTPUTS:
        sched:          A list of SimulationScheduleEntry objects
    """
    # The model_type is always special
    model_type: ModelType = ModelType.Special
    # Minimum number of steps for each stage
    min_step: int = 1000
    # Maximum steps for all the stages
    max_step: int = 128000
    # Stop time for each stage
    stop_time: float = 1.16167017
    # Is this to steady state?
    is_steady_state: bool = False

    # Use a moderate CFL for all the stages
    cfl: float = 0.50
    # Max consumption in the special model; start with a conservative setting because we are
    # adaptively increasing max_consumption after successfully converged time steps
    max_consumption: float = 1.0 / 128.0
    # Global limit on maximum consumption of reactant in a cell during adaptive tuning of mrc parameter
    max_consumption_limit: float = 1.0 / 16.0

    # Set relaxation parameter for Nernst model
    nernst_omega: float = 1.0 / 64.0
    # Maximum overpotential in volts
    overpot_max: float = 0.100

    # Maximum number of stagnant steps
    stagnant_steps_max: int = 128000
    # Set tolerance for all the stages
    tol: float = 1.0E-6

    # The status interval
    interval_status: int = interval_status_flow
    # The intervals bewteen checkpoints
    interval_amrex: int = 1000

    # Initialize the schedule with an empty list
    sched = list()

    # Each entry simulates a level of refinement starting from its flow simulation checkpoint
    for i in range(1, max_levels+1):
        # Set the arguments for this stage
        stage_num: int = i
        grid_levels: int = i
        lev: int = stage_num - 1
        upsample_from: int = max(lev, 0)
        # The input directory for flow checkpoints
        amrex_input_dir: str = Path('..', '..', '..', '..', flow_dir, f'stage{i:02d}', 'chk_last').as_posix()
        # Build this entry in the schedule
        sched.append(SimulationScheduleEntry(
            stage_num=stage_num, model_type=model_type, grid_levels=grid_levels,
            min_step=min_step, max_step=max_step, stop_time=stop_time, is_steady_state=is_steady_state,
            cfl=cfl, max_consumption=max_consumption, max_consumption_limit=max_consumption_limit,
            nernst_omega=nernst_omega, overpot_max=overpot_max,
            stagnant_steps_max=stagnant_steps_max, tol=tol,
            amrex_input_dir=amrex_input_dir, upsample_from=upsample_from,
            interval_status=interval_status, interval_amrex=interval_amrex))

    return sched

# **********************************************************************************************************************
def make_inputs_cnvg_flow(G: int, nz: int, max_levels: int):
    """
    Build the inputs for the flow convergence simulations
    INPUTS:
    G:          The geometry number from [1, 2, 3]
    nz:         The number of grid cells in the z direction
    max_levels: The maximum number of grid levels
    """
    # The simulation specification
    sim_spec: SimulationSpec = SimulationSpec(
        sim_type=SimulationType.Flow,
        G=G,
        P=pressure,
        V=0,
        conc=conc,
        soc_in=soc_eq,
        nz=nz)

    # No extra macros
    extra_macros = ''

    # Do we use upsampling?
    # do_upsample: bool = upsample_tbl[G]
    do_upsample: bool = False

    # The schedule for flow simulations does not depend on L
    sched: list[SimulationScheduleEntry] = \
        make_schedule_flow(max_levels=max_levels, do_upsample=do_upsample)

    # Delete old input files
    sim_dir: Path = calc_dir_cnvg_flow(G=G, nz=nz)
    # Input files are e.g. 01_OneWire/stage01/input2D.cfg, Lattice1/stage02/input2D.cfg, etc.
    for p in sim_dir.glob(f'stage??/inputs.cfg'):
        p.unlink()

    # Iterate over stages in the flow schedule
    for sched_entry in sched:
        # Delegate to make_inputs
        make_inputs(sim_dir=sim_dir, sim_spec=sim_spec, sched_entry=sched_entry, extra_macros=extra_macros)

# **********************************************************************************************************************
def make_inputs_cnvg_react(G: int, nz: int, max_levels: int):
    """
    Build the inputs for the reaction convergence simulations
    INPUTS:
    G:          The geometry number from [1, 2, 3]
    nz:         The number of grid cells in the z direction
    max_levels: The maximum number of grid levels
    """
    # The voltage in this convergence simulation is always zero
    V: int = 0
    # The simulation specification
    sim_spec: SimulationSpec = SimulationSpec(
        sim_type=SimulationType.React,
        G=G,
        P=pressure,
        V=V,
        conc=conc,
        soc_in=soc_eq,
        nz=nz)

    # No extra macros
    extra_macros = ''

    # The corresponding flow directory
    flow_dir: Path = calc_dir_cnvg_flow(G=G, nz=nz)

    # The schedule for flow simulations does not depend on L
    sched: list[SimulationScheduleEntry] = make_schedule_react(flow_dir=flow_dir, max_levels=max_levels)

    # Delete old input files
    sim_dir: Path = calc_dir_cnvg_react(G=G, nz=nz)
    for p in sim_dir.glob(f'stage??/inputs.cfg'):
        p.unlink()

    # Iterate over stages in the reaction schedule
    for sched_entry in sched:
        # Delegate to make_inputs
        make_inputs(sim_dir=sim_dir, sim_spec=sim_spec, sched_entry=sched_entry, extra_macros=extra_macros)

# **********************************************************************************************************************
def make_inputs_react_bv(G: int, nz: int, max_levels: int):
    """
    Build the inputs for the reference Butler-Volmer reaction simulations
    INPUTS:
    G:          The geometry number from [1, 2, 3]
    nz:         The number of grid cells in the z direction
    max_levels: The maximum number of grid levels
    """
    # The voltage in this reference simulation is always zero
    V: int = 0
    # The simulation specification
    sim_spec: SimulationSpec = SimulationSpec(
        sim_type=SimulationType.React,
        G=G,
        P=pressure,
        V=V,
        conc=conc,
        soc_in=soc_eq,
        nz=nz)

    # No extra macros
    extra_macros = ''

    # The corresponding flow directory
    flow_dir: Path = calc_dir_flow(nz=nz)

    # The schedule for flow simulations does not depend on L
    sched: list[SimulationScheduleEntry] = make_schedule_react(flow_dir=flow_dir, max_levels=max_levels)

    # Delete old input files
    sim_dir: Path = calc_dir_react_bv(nz=nz)
    for p in sim_dir.glob(f'stage??/inputs.cfg'):
        p.unlink()

    # Iterate over stages in the reaction schedule
    for sched_entry in sched:
        # Delegate to make_inputs
        make_inputs(sim_dir=sim_dir, sim_spec=sim_spec, sched_entry=sched_entry, extra_macros=extra_macros)

# **********************************************************************************************************************
def main(Gs: list[int], nzs: list[int], max_levels: int, cnvg_flow: bool, cnvg_react: bool):
    """
    Build lattice and input files
    INPUTS:
        Gs: list[int] - list of geometry numbers
        nzs: list[int] - list of grid sizes in z
        cnvg_flow: bool - should we build flow convergence simulation inputs?
        cnvg_react: bool - should we build flow convergence simulation inputs?
    """
    # Iterate over all the geometry numbers and grid sizes
    for G in Gs:
        for nz in nzs:
            # Make the flow convergence simulations if requested
            if cnvg_flow:
                make_inputs_cnvg_flow(G=G, nz=nz, max_levels=max_levels)
            # Make the reaction convergence simulations if requested
            if cnvg_react:
                make_inputs_cnvg_react(G=G, nz=nz, max_levels=max_levels)

# **********************************************************************************************************************
if __name__ == '__main__':
    # Tuple with all the allowed geometry numbers
    # Gs: list[int] = [1, 2, 3]
    Gs: list[int] = [3]

    # Factors of 30 starting with 5
    # nzs_30: list[int] = [5, 6, 10, 15, 30]
    # Factors of 60 starting with 5
    # nzs_60: list[int] = [5, 6, 10, 12, 15, 20, 30, 60]
    # Factors of 120 starting with 5
    # nzs_120: list[int] = [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]

    # All the resolutions to use for grid in z; factors of 60 starting with 5
    # nzs: list[int] = [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60,]

    # Maximum number of levels of gridding
    # max_levels: int = 1

    # Flags for which simulations to build
    cnvg_flow: bool = False
    cnvg_react: bool = True

    # Invoke main function
    # main(Gs=Gs, nzs=nzs, max_levels=max_levels, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)
    # main(Gs=Gs, nzs=[8 ], max_levels=5, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)
    # main(Gs=Gs, nzs=[16], max_levels=4, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)
    # main(Gs=Gs, nzs=[32], max_levels=3, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)
    # main(Gs=Gs, nzs=[64], max_levels=2, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)
    # main(Gs=Gs, nzs=[128], max_levels=1, cnvg_flow=cnvg_flow, cnvg_react=cnvg_react)

    make_inputs_react_bv(G=3, nz=32, max_levels=3)