import numpy as np
import numpy.typing as npt
from scipy.special import expit
from pathlib import Path
from dataclasses import dataclass, field

from utils import CellType, Geometry, calc_geometry, geometry_add_types, geometry_add_measure, make_V_str

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
# Path to the flow simulations
top_dir_flow: Path = Path('npy/01_flow')
# Path to the mass transport utilization simulations
top_dir_mtu: Path = Path('npy/02_mass_transport')
# Path to the explicit Nernst simulation (simplified Butler-Volmer model)
top_dir_exp: Path = Path('npy/03_nernst')

# *************************************************************************************************
# Physical constants
from constants import R, T, F, V_T, ne, c0, diff, k0

# *************************************************************************************************
@dataclass
class ReactSimExplicit:
    """Data for an Explicit Nernst reaction simulation."""
    # Geometry of the simulation
    geom: Geometry
    # Inlet pressure in Pascals
    p_in: int
    # Applied voltage in mV
    V_app: int
    # Flow velocity; shape (Nx, Ny, Nz, 3)
    velocity: NumpyFloatArray
    # State of charge; shape (Nx, Ny, Nz)
    soc: NumpyFloatArray
    # Dimensionless overpotential; shape (Nx, Ny, Nz)
    eta: NumpyFloatArray

# *************************************************************************************************
def load_domain(dir_sim: Path) -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """Load the domain from file"""
    # Load the lower coordinates of the box and convert to microns
    lo_levs: NumpyFloatArray = np.load(dir_sim / 'domain_lo.npy')
    # Load the upper coordinates of the box
    hi_levs: NumpyFloatArray = np.load(dir_sim / 'domain_hi.npy')
    # Extract just the top level and convert from cm to microns
    cm2um: float = 1.0E4
    lo: NumpyFloatArray = lo_levs[0] * cm2um
    hi: NumpyFloatArray = hi_levs[0] * cm2um
    return lo, hi

# *************************************************************************************************
def load_flow_explicit(P: int, lev: int):
    """
    Load the flow data and geometry for an explicit simulation.
    INPUTS:
        P: Inlet pressure in Pascals
        lev: Level of the simulation
    Returns:
        geom: Geometry object
        velocity: Velocity field
    """
    # The stage of the simulation
    stage: int = lev + 1
    # Build path to the data
    dec_exp: int = int(np.round(np.log10(P)))
    dec_man: int = P // (10**dec_exp)
    P_str: str = f'P{dec_man:d}E{dec_exp:d}'
    stg: str = f'stage{stage:02d}'
    # The directory with the flow simulation
    dir_sim: Path = top_dir_flow / f'{P_str:s}/{stg:s}'
    # Load the domain
    lo, hi = load_domain(dir_sim=dir_sim)
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[lev]
    # Build the geometry
    geom: Geometry = calc_geometry(lo=lo, hi=hi, shape=grid_shape)

    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / f'L{lev:0d}'
    # Load the velocity field
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    # Load the geometry information
    cell_type: NumpyInt8Array = np.load(dir_lev / 'cell_type.npy')
    refinement: NumpyInt8Array = np.load(dir_lev / 'refinement.npy')
    area: NumpyFloatArray = np.load(dir_lev / 'area.npy')
    volume: NumpyFloatArray = np.load(dir_lev / 'volume.npy')
    # Add cell types to the geometry
    geometry_add_types(geom=geom, cell_type=cell_type, refinement=refinement)
    # Add the measures to the geometry
    geometry_add_measure(geom=geom, area=area, volume=volume)
    # Return the geometry and velocity field
    return geom, velocity

# *************************************************************************************************
def load_sim_mtu(P: int, lev: int):
    """
    Load the data for a mass transport utilization simulation.
    INPUTS:
        P: Inlet pressure in Pascals
        lev: Level of the simulation
    Returns:
        ReactSimExplicit
    """
    # The stage of the simulation
    stage: int = lev + 1
    # Build path to the data
    dec_exp: int = int(np.round(np.log10(P)))
    dec_man: int = P // (10**dec_exp)
    P_str: str = f'P{dec_man:d}E{dec_exp:d}'
    stg: str = f'soc{stage:02d}'
    
    # Load the geometry and velocity field
    geom: Geometry
    velocity: NumpyFloatArray
    geom, velocity = load_flow_explicit(P=P, lev=lev)

    # Load the reaction data for the mass transport model at the requested operating conditions and level
    dir_lev_rxn: Path = top_dir_mtu / f'{P_str:s}/{stg:s}'
    # Load the state of charge
    soc: NumpyFloatArray = np.load(dir_lev_rxn / 'soc_steady.npy')
    # The overpotential is zero for the mass transport utilization simulations
    eta: NumpyFloatArray = np.zeros_like(soc)

    # Create the dataclass
    V_app: int = 0
    return ReactSimExplicit(geom=geom, p_in=P, V_app=V_app, velocity=velocity, soc=soc, eta=eta)

# *************************************************************************************************
def load_sim_explicit(P: int, V: int, lev: int):
    """
    Load the data for an explicit Nernst simulation.
    INPUTS:
        P: Inlet pressure in Pascals
        V: Applied voltage in mV
        lev: Level of the simulation
    Returns:
        ReactSimExplicit
    """
    # The stage of the simulation
    stage: int = lev + 1
    # Build path to the data
    dec_exp: int = int(np.round(np.log10(P)))
    dec_man: int = P // (10**dec_exp)
    P_str: str = f'P{dec_man:d}E{dec_exp:d}'
    V_str: str = make_V_str(V)
    stg: str = f'stage{stage:02d}'
    
    # Load the geometry and velocity field
    geom: Geometry
    velocity: NumpyFloatArray
    geom, velocity = load_flow_explicit(P=P, lev=lev)

    # Load the reaction data for the explicit Nernst model at the requested operating conditions and level
    dir_lev_rxn: Path = top_dir_exp / f'{P_str:s}/{V_str:s}/{stg:s}'
    # Load the state of charge
    soc: NumpyFloatArray = np.load(dir_lev_rxn / 'soc_steady.npy')
    # Load the overpotential
    eta: NumpyFloatArray = np.load(dir_lev_rxn / 'overpot_steady.npy')

    # Create the dataclass
    return ReactSimExplicit(geom=geom, p_in=P, V_app=V, velocity=velocity, soc=soc, eta=eta)

# *************************************************************************************************
def calc_util(geom: Geometry, velocity: NumpyFloatArray, soc: NumpyFloatArray):
    """Calculate the utilization from the velocity and state of charge."""
    # Mask for the outlet
    mask: NumpyBoolArray = geom.is_outlet
    # Cross sectional area of outlet cells
    a_out: float = geom.h * geom.h
    # Velocity at the outlet - x component
    u: NumpyFloatArray = velocity[..., 0]
    u_out: NumpyFloatArray = u[mask]
    # State of charge at the outlet
    s_out: NumpyFloatArray = soc[mask]

    # Flow rate at the outlet
    Q: float = np.sum(a_out * u_out)
    # Outbound flow of reduced species
    Q_out: float = np.sum(a_out * u_out * s_out)
    # Implied utilization
    util: float = Q_out / Q
    return util

# *************************************************************************************************
def calc_seq(V: float):
    """
    Calculate equilibrium state of charge for a given applied voltage.
    INPUTS:
        V: Applied reducing voltage in mV
    RETURNS:
        seq: Equilibrium state of charge
    """
    # Dimensionless overpotential
    V_art: float = V / (ne * V_T)
    # Calculate the equilibrium state of charge
    seq: float = expit(V_art)
    return seq