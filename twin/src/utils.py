import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# *************************************************************************************************
class SimType(Enum):
    """Enumeration of the types of simulations"""
    # The simulation is a flow simulation
    flow = 1
    # The simulation is a mass reaction simulation
    react = 2

# *************************************************************************************************
class CellType(Enum):
    """Enumeration of the types of cells"""
    # The cell is a fluid cell
    fluid = 1
    # The cell is a boundary cell
    boundary = 2
    # The cell is a solid cell
    solid = 4

# *************************************************************************************************
class ReactionModel(Enum):
    """Enumeration of reaction models"""
    # Special model (a.k.a. Simplified Butler-Volmer)
    Special = 1
    # Nernst model
    Nernst = 2
    # Full Butler-Volmer model
    ButlerVolmer = 3
    # Explicit Nernst model
    ExplicitNernst = 4

# *************************************************************************************************
@dataclass 
class Geometry:
    """Class to hold the geometry of the simulation"""
    # Low end of the domain in cm; array of 3 floats
    lo: NumpyFloatArray
    # High end of the domain in cm; array of 3 floats
    hi: NumpyFloatArray
    # Grid spacing in cm
    h: float

    # Number of grid points in x direction
    nx: int
    # Number of grid points in y direction
    ny: int
    # Number of grid points in z direction
    nz: int

    # Shape of the grid
    shape: tuple[int,int,int] = field(init=False)
    # Size of the grid
    N: int = field(init=False)

    # Index of cell nodes in x direction; array of shape (nx,)
    ii: NumpyInt32Array
    # Index of cell nodes in y direction; array of shape (ny,)
    jj: NumpyInt32Array
    # Index of cell nodes in k direction; array of shape (nz,)
    kk: NumpyInt32Array

    # Row number of a cell with coordinates (i, j, k); shape (nx, ny, nz); chosen such that
    # rr.flatten() == np.arange(N)
    rr: NumpyInt32Array
    
    # Location of cell nodes in x direction; array of shape (nx+1,)
    xn: NumpyFloatArray
    # Location of cell nodes in y direction; array of shape (ny+1,)
    yn: NumpyFloatArray
    # Location of cell nodes in z direction; array of shape (nz+1,)
    zn: NumpyFloatArray

    # Location of cell centers in x direction; array of shape (nx,)
    xc: NumpyFloatArray
    # Location of cell centers in y direction; array of shape (ny,)
    yc: NumpyFloatArray
    # Location of cell centers in z direction; array of shape (nz,)
    zc: NumpyFloatArray

    # Meshgrid of cell centers; array of shape (nx, ny, nz, 3)
    center: NumpyFloatArray

    # The liquid volume of each cell; array of shape (nx, ny, nz)
    volume: NumpyFloatArray = field(init=False)
    # The surface area of each boundary cell; array of shape (nx, ny, nz)
    area: NumpyFloatArray = field(init=False)

    # The type of each cell; array of shape (nx, ny, nz)
    cell_type: NumpyInt8Array = field(init=False)
    # The refinement level of each cell; array of shape (nx, ny, nz)
    refinement: NumpyInt8Array = field(init=False)

    # These are optional fields that can be generated if needed
    # Mask of fluid cells
    is_fluid: NumpyBoolArray = field(init=False)
    # Mask of wire cells
    is_wire: NumpyBoolArray = field(init=False)
    # Mask of solid cells
    is_solid: NumpyBoolArray = field(init=False)

    # Mask of cells at the inlet
    is_inlet: NumpyBoolArray = field(init=False)
    # Mask of cells at the outlet
    is_outlet: NumpyBoolArray = field(init=False)
    # Mask of live cells
    is_live: NumpyBoolArray = field(init=False)

    def __post_init__(self):
        self.shape = (self.nx, self.ny, self.nz)
        self.N = self.nx * self.ny * self.nz
        self.volume = None
        self.area = None
        self.cell_type = None
        self.refinement = None
        self.is_fluid = None
        self.is_wire = None
        self.is_solid = None
        self.is_inlet = (self.ii == 0)
        self.is_outlet = (self.ii == self.nx-1)
        self.is_live = None

    def summarize(self) -> None:
        n_liquid: int = int(np.sum(self.is_fluid))
        n_wire: int = int(np.sum(self.is_wire))
        n_solid: int = int(np.sum(self.is_solid))
        n_total: int = n_liquid + n_wire + n_solid
        print('\nCell count by type')
        print(f'liquid  : {n_liquid:12d}')
        print(f'wire    : {n_wire:12d}')
        print(f'solid   : {n_solid:12d}')
        print(f'total   : {n_total:12d}')
        print(f'N       : {self.N:12d}')

# *************************************************************************************************
@dataclass
class FlowSim:
    """Data for a flow simulation."""
    # Geometry of the simulation
    geom: Geometry
    # Inlet pressure in Pascals
    p_in: int
    # Flow velocity; shape (Nx, Ny, Nz, 3)
    velocity: NumpyFloatArray
    # Pressure at cell centers; shape (Nx, Ny, Nz)
    pressure_cell: NumpyFloatArray
    # Pressure at cell corners; shape (Nx+1, Ny+1, Nz+1)
    # pressure_node: NumpyFloatArray
    # Flow speed; shape (Nx, Ny, Nz)
    speed: NumpyFloatArray

# *************************************************************************************************
@dataclass
class ReactSim:
    """Data for a reaction simulation."""
    # Geometry of the simulation
    geom: Geometry
    # Inlet pressure in Pascals
    p_in: int
    # Flow velocity; shape (Nx, Ny, Nz, 3)
    velocity: NumpyFloatArray
    # State of charge; shape (Nx, Ny, Nz)
    soc: NumpyFloatArray
    # Current; shape (Nx, Ny, Nz)
    current: NumpyFloatArray = None
    # Overpotential; shape (Nx, Ny, Nz)
    overpot: NumpyFloatArray = None
    # Potential in the liquid; shape (Nx, Ny, Nz)
    epot: NumpyFloatArray = None

# *************************************************************************************************
def calc_geometry(lo: NumpyFloatArray, hi: NumpyFloatArray, shape: tuple[int,int,int]) -> Geometry:
    """
    Build Geometry object with all the geometry information
    INPUTS:
        lo - lower coordinates of the domain
        hi - upper coordinates of the domain
        shape - shape of the spatial domain gridding (nx, ny, nz)
    """
    # Unpack the shape
    nx: int
    ny: int
    nz: int
    nx, ny, nz = shape[0:3]

    # Unpack domain low coordinates
    x_lo: float = lo[0]
    y_lo: float = lo[1]
    z_lo: float = lo[2]

    # Unpack domain high coordinates
    x_hi: float = hi[0]
    y_hi: float = hi[1]
    z_hi: float = hi[2]

    # The grid spacing
    h: float = (x_hi - x_lo) / nx

    # Meshgrid of cell index coordinates (i, j, k) matching cell centers
    ii: NumpyInt32Array = np.zeros(shape, dtype=np.int32)
    jj: NumpyInt32Array = np.zeros(shape, dtype=np.int32)
    kk: NumpyInt32Array = np.zeros(shape, dtype=np.int32)
    # Assign the coordinates
    ii[:] = np.arange(nx, dtype=np.int32).reshape(nx, 1 , 1 )
    jj[:] = np.arange(ny, dtype=np.int32).reshape(1 , ny, 1 )
    kk[:] = np.arange(nz, dtype=np.int32).reshape(1 , 1 , nz)

    # The row number of a cell with coordinates (i, j, k); shape (nx, ny, nz); chosen such that
    # rr.flatten() == np.arange(N)
    rr: NumpyInt32Array = ii*ny*nz + jj*nz + kk

    # Build grid node points in each direction; add 1 to the number of points
    xn: NumpyFloatArray = np.linspace(x_lo, x_hi, nx + 1)
    yn: NumpyFloatArray = np.linspace(y_lo, y_hi, ny + 1)
    zn: NumpyFloatArray = np.linspace(z_lo, z_hi, nz + 1)

    # Build grid center points in each direction
    xc: NumpyFloatArray = (xn[0:nx+0] + xn[1:nx+1]) / 2
    yc: NumpyFloatArray = (yn[0:ny+0] + yn[1:ny+1]) / 2
    zc: NumpyFloatArray = (zn[0:nz+0] + zn[1:nz+1]) / 2

    # Build meshgrid of cell centers
    xg: NumpyFloatArray     # shape (nx, ny, nz) with x coordinates
    yg: NumpyFloatArray     # shape (nx, ny, nz) with y coordinates
    zg: NumpyFloatArray     # shape (nx, ny, nz) with z coordinates
    xg, yg, zg = np.meshgrid(xc, yc, zc, indexing='ij')
    # Wrap the centers into an array of shape (nx, ny, nz, 3)
    center: NumpyFloatArray = np.stack((xg, yg, zg), axis=3)

    # Wrap the geometry into a Geometry object
    geom: Geometry = Geometry(lo=lo, hi=hi, h=h, nx=nx, ny=ny, nz=nz, ii=ii, jj=jj, kk=kk, rr=rr,
                              xn=xn, yn=yn, zn=zn, xc=xc, yc=yc, zc=zc, center=center)

    # Return tuple of the grid center points
    return geom

# *************************************************************************************************
def geometry_add_types(geom: Geometry, cell_type: NumpyInt8Array, refinement: NumpyInt8Array) -> None:
    """Update the geometry object with cell types in place"""
    # Set the cell_type
    geom.cell_type = cell_type
    # Set the refinement level
    geom.refinement = refinement

    # Mask of liquid cells
    geom.is_fluid = (cell_type == 1)
    # Mask of wire cells
    geom.is_wire = (cell_type == 2)
    # Mask of solid cells
    geom.is_solid = (cell_type == 4)
    # Mask of live cells
    geom.is_live = geom.is_fluid & (~geom.is_inlet)

# *************************************************************************************************
def geometry_add_measure(geom: Geometry, volume: NumpyFloatArray, area: NumpyFloatArray) -> None:
    """Update the geometry object with the cell volume and area in place"""
    # The cell volume
    geom.volume = volume
    # The cell area
    geom.area = area

# *************************************************************************************************
def load_flow_sim_impl(dir_npy: Path, level: int) -> FlowSim:
    """
    Load calculated data for a flow simulation from a path.
    INPUTS:
        dir_npy - the path to the numpy outputs, e.g. /twin/simulations/11_stokesflow/P1E0/stage01/numpy_last
        stage- the stage of the simulation
        level - the level of the simulation output data
    """
    # Load the domain data
    domain_lo: NumpyFloatArray = np.load(dir_npy / 'domain_lo.npy')
    domain_hi: NumpyFloatArray = np.load(dir_npy / 'domain_hi.npy')
    grid_shape: NumpyFloatArray = np.load(dir_npy / 'grid_shape.npy')

    # Low and high ends of domain
    lo: NumpyFloatArray = domain_lo[0]
    hi: NumpyFloatArray = domain_hi[0]

    # Unpack number of grid points at selected level
    nx: int = grid_shape[level][0]
    ny: int = grid_shape[level][1]
    nz: int = grid_shape[level][2]
    # The grid shape
    shape: tuple[int,int,int] = (nx, ny, nz)

    # Build the grid geometry
    geom: Geometry = calc_geometry(lo, hi, shape)

    # Directory for the selected level data
    dir_lev: Path = dir_npy / f'L{level:0d}'

    # Placeholder for pressure; to be filled in later
    p_in: int = 0

    # Load the data on the selected level
    cell_type: NumpyInt8Array = np.load(dir_lev / 'cell_type.npy')
    refinement: NumpyInt8Array = np.load(dir_lev / 'refinement.npy').astype(np.int8)
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    pressure_cell: NumpyFloatArray = np.load(dir_lev / 'pressure_cell.npy')
    volume: NumpyFloatArray = np.load(dir_lev / 'volume.npy')
    area: NumpyFloatArray = np.load(dir_lev / 'area.npy')

    # Add the cell types to the geometry
    geometry_add_types(geom=geom, cell_type=cell_type, refinement=refinement)
    # Add the cell volume and area to the geometry
    geometry_add_measure(geom=geom, volume=volume, area=area)

    # Calculate the speed from the velocity
    speed: NumpyFloatArray = np.linalg.norm(velocity, axis=-1)

    # Return the flow simulation data
    return FlowSim(geom=geom, p_in=p_in, velocity=velocity, speed=speed, pressure_cell=pressure_cell)

# *************************************************************************************************
def load_flow_sim_path(dir_sim: Path, stage: int, level: int) -> FlowSim:
    """
    Load calculated data for a flow simulation from a path with the simulation output.
    INPUTS:
        dir_sim - the path to the simulation, e.g. /twin/simulations/11_stokes_flow/P1E0
        stage- the stage of the simulation
        level - the level of the simulation output data
    """
    # Delegate to load_flow_sim_impl
    dir_npy: Path = dir_sim /f'stage{stage:02d}' / 'numpy_last'
    return load_flow_sim_impl(dir_npy=dir_npy, level=level)

# *************************************************************************************************
def load_react_sim_impl(dir_npy: Path, level: int) -> ReactSim:
    """
    Load calculated data for a reaction simulation from a path containing the numpy data.
    INPUTS:
        dir_npy - the path to the numpy outputs, e.g. /twin/simulations/12_mass_transport/P1E0/stage01/numpy_last
        level - the level of the simulation output data
    """
    # Load the domain data
    domain_lo: NumpyFloatArray = np.load(dir_npy / 'domain_lo.npy')
    domain_hi: NumpyFloatArray = np.load(dir_npy / 'domain_hi.npy')
    grid_shape: NumpyFloatArray = np.load(dir_npy / 'grid_shape.npy')

    # Low and high ends of domain
    lo: NumpyFloatArray = domain_lo[0]
    hi: NumpyFloatArray = domain_hi[0]

    # Unpack number of grid points at selected level
    nx: int = grid_shape[level][0]
    ny: int = grid_shape[level][1]
    nz: int = grid_shape[level][2]
    # The grid shape
    shape: tuple[int,int,int] = (nx, ny, nz)

    # Build the grid geometry
    geom: Geometry = calc_geometry(lo, hi, shape)

    # Directory for the selected level data
    dir_lev: Path = dir_npy / f'L{level:0d}'

    # Placeholder for pressure; to be filled in later
    p_in: int = 0

    # Load the data on the selected level
    cell_type: NumpyInt8Array = np.load(dir_lev / 'cell_type.npy')
    refinement: NumpyInt8Array = np.load(dir_lev / 'refinement.npy').astype(np.int8)
    volume: NumpyFloatArray = np.load(dir_lev / 'volume.npy')
    area: NumpyFloatArray = np.load(dir_lev / 'area.npy')
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    soc: NumpyFloatArray = np.load(dir_lev / 'soc.npy')
    current: NumpyFloatArray = np.load(dir_lev / 'current.npy')
    overpot: NumpyFloatArray = np.load(dir_lev / 'overpot.npy')

    # Add the cell types to the geometry
    geometry_add_types(geom=geom, cell_type=cell_type, refinement=refinement)
    # Add the cell volume and area to the geometry
    geometry_add_measure(geom=geom, volume=volume, area=area)

    # Return the flow simulation data
    # return ReactSim(geom=geom, p_in=p_in, cell_type=cell_type, refinement=refinement, volume=volume, area=area,
    #                velocity=velocity, soc=soc, current=current, overpot=overpot)
    return ReactSim(geom=geom, p_in=p_in, velocity=velocity, soc=soc, current=current, overpot=overpot)

# *************************************************************************************************
def load_react_sim_path(dir_sim: Path, stage: int, level: int) -> ReactSim:
    """
    Load calculated data for a reaction simulation from a path.
    INPUTS:
        dir_sim - the path to the simulation,  e.g. /twin/simulations/12_mass_transport/P1E0
        stage - the stage of the simulation
        level - the level of the simulation output data
    """
    # Delegate to load_react_sim_impl
    dir_npy: Path = dir_sim /f'stage{stage:02d}' / 'numpy_last'
    return load_react_sim_impl(dir_npy=dir_npy, level=level)

# *************************************************************************************************
def make_P_str(P: int) -> str:
    """Make a string for the pressure"""
    exp: int = int(np.floor(np.log10(float(P))))
    man: int = int(np.round(P / np.power(10.0, exp)))
    P_str: str = f'P{man:d}E{exp:d}'
    return P_str

# *************************************************************************************************
def make_V_str(V: int) -> str:
    """Make a string for the voltage in millivolts"""
    sgn: str = 'p' if V >= 0 else 'm'
    mag: int = int(abs(V))
    return f"V{sgn}{mag:03d}"
