import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from pathlib import Path

# Local imports
from utils import Geometry, calc_geometry
from cnvg_utils import SimOutput
from plot_utils import plot_scalar

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# *************************************************************************************************
# Path to the reference numpy simulation data
dir_sim_ref: Path = Path('06_flow/P100/n128/stage01/numpy_last')
# Path to the upsampled numpy simulation data (using mesh refinement)
dir_sim_mesh: Path = Path('06_flow/P100/n064/stage02/numpy_last')

# Path to save the plots
dir_plots: Path = Path('figs/05_upsample')
dir_plots.mkdir(parents=True, exist_ok=True)

# *************************************************************************************************
# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# *************************************************************************************************
def load_domain() -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """Load the domain from file"""
    # Load the lower coordinates of the box and convert to microns
    lo_levs: NumpyFloatArray = np.load(dir_sim_ref / 'domain_lo.npy')
    # Load the upper coordinates of the box
    hi_levs: NumpyFloatArray = np.load(dir_sim_ref / 'domain_hi.npy')
    # Extract just the top level and convert from cm to microns
    # cm2um: float = 1.0E4
    # lo: NumpyFloatArray = lo_levs[0] * cm2um
    # hi: NumpyFloatArray = hi_levs[0] * cm2um
    lo: NumpyFloatArray = lo_levs[0]
    hi: NumpyFloatArray = hi_levs[0]
    return lo, hi

# *************************************************************************************************
def load_sim_ref() -> SimOutput:
    """Load the numpy arrays for the reference simulation at n=128"""
    # The directory containing the simulation numpy output for this stage
    dir_sim: Path = dir_sim_ref
    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / 'L0'
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[0]
    # Load the numpy arrays for this level
    refinement: NumpyInt16Array = np.load(dir_lev / 'refinement.npy')
    cell_type: NumpyFloatArray = np.load(dir_lev / 'cell_type.npy')
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    pressure: NumpyFloatArray = np.load(dir_lev / 'pressure_cell.npy')
    soc = np.zeros(grid_shape)
    # Wrap the numpy arrays in a SimOutput object
    sim_out: SimOutput = SimOutput(grid_shape=grid_shape, refinement=refinement, cell_type=cell_type, 
                                   velocity=velocity, pressure=pressure, soc=soc)
    return sim_out

# *************************************************************************************************
def load_sim_ups(level: int) -> SimOutput:
    """Load the numpy arrays for the upsampled simulation at n=64 at the given level"""
    # The directory containing the simulation numpy output for this stage
    dir_sim: Path = dir_sim_mesh
    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / f'L{level:0d}'
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[level]
    # Load the numpy arrays for this level
    refinement: NumpyInt16Array = np.load(dir_lev / 'refinement.npy')
    cell_type: NumpyFloatArray = np.load(dir_lev / 'cell_type.npy')
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    pressure: NumpyFloatArray = np.load(dir_lev / 'pressure_cell.npy')
    soc = np.zeros(grid_shape)
    # Wrap the numpy arrays in a SimOutput object
    sim_out: SimOutput = SimOutput(grid_shape=grid_shape, refinement=refinement, cell_type=cell_type, 
                                   velocity=velocity, pressure=pressure, soc=soc)
    return sim_out

# *************************************************************************************************
def make_interpolator(vel: NumpyFloatArray, geom: Geometry) -> RegularGridInterpolator:
    """Create interpolators for the velocity field and unit vectors"""
    # Wrap the cell centers into a tuple as expected by RegularGridInterpolator
    points: tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray] = (geom.xc, geom.yc, geom.zc)

    # Select the interpolation method
    method: str = 'linear'
    # Fill value for the velocity field
    fill_value: NumpyFloatArray = np.zeros(3)
    # Don't raise an error when trying to interpolate outside of the domain
    bounds_error: bool = False

    # Create the interpolator for the velocity field
    vel_func: RegularGridInterpolator = \
        RegularGridInterpolator(points, vel, method=method, bounds_error=bounds_error, fill_value=fill_value)

    return vel_func

# *************************************************************************************************
def upsample_vel(sim_m0, sim_m1):
    """"""
    # Extract the velocity fields
    vel_m0: NumpyFloatArray = np.nan_to_num(sim_m0.velocity, nan=0.0)
    vel_m1: NumpyFloatArray = np.nan_to_num(sim_m1.velocity, nan=0.0)

    # Load the domain
    lo, hi = load_domain()

    # Shape of the domain on the coarse mesh refined level
    shape0: tuple[int, int, int] = vel_m0.shape[0:3]
    # Shape of the domain on the fine mesh refined level
    shape1: tuple[int, int, int] = vel_m1.shape[0:3]

    # Calculate geometry on coarse and fine levels
    geom0: Geometry = calc_geometry(lo=lo, hi=hi, shape=shape0)
    geom1: Geometry = calc_geometry(lo=lo, hi=hi, shape=shape1)

    # Center coordinates
    center1: NumpyFloatArray = geom1.center

    # Build interpolator for velocity on the coarse mesh refined level
    vel_func = make_interpolator(vel=vel_m0, geom=geom0)

    # Upsample the velocity field on the coarse mesh refined level
    vel_ups: NumpyFloatArray = vel_func(center1)
    # Initialize mesh refined velocity
    vel_m: NumpyFloatArray = vel_m1.copy()
    # Mask where upsampled velocity may be better
    mask: NumpyBoolArray = (sim_m1.refinement == 0)
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[0:-2, 1:-1, 1:-1]
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[2:  , 1:-1, 1:-1]
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[1:-1, 0:-2, 1:-1]
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[1:-1, 2:  , 1:-1]
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[1:-1, 1:-1, 0:-2]
    mask[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & mask[1:-1, 1:-1, 2:  ]
    # Overwrite original velocity with the upsampled one
    vel_m[mask] = vel_ups[mask]

    return vel_m

# *************************************************************************************************
def plot_speed(speed: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot the speed at a given z-slice
    INPUTS:
        speed: Array of speed data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = 0.0
    vmax: float = np.nanmax(speed[:,:,k]).astype(float)
    cmap: str = 'turbo'
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar(phi=speed, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_vel_err(vel_err: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot the speed at a given z-slice
    INPUTS:
        vel_err: Array of velocity error data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = 0.0
    vmax: float = np.nanmax(vel_err[:,:,k]).astype(float)
    cmap: str = 'jet'
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar(phi=vel_err, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_pressure(pressure: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style, path: Path) -> None:
    """
    Plot the pressure at a given z-slice
    INPUTS:
        pressure: Array of pressure data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = 0.0
    vmax: float = np.nanmax(pressure[:,:,k]).astype(float)
    cmap: str = 'turbo'
    cbar_fmt: str = '%0.02f'
    # Delegate to plot_scalar
    plot_scalar(phi=pressure, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def main():
    """Compare the flow fields of the reference and mesh refined simulations"""
    # Load the reference simulation
    sim_ref: SimOutput = load_sim_ref()

    # Load the mesh refined simulation
    sim_ups: SimOutput = load_sim_ups(level=1)

    # Extract the velocity fields
    vel_ref: NumpyFloatArray = np.nan_to_num(sim_ref.velocity, nan=0.0)
    vel_ups: NumpyFloatArray = np.nan_to_num(sim_ups.velocity, nan=0.0)

    # Load the domain
    lo, hi = load_domain()
    # Shape of the domain
    shape: tuple[int, int, int] = vel_ref.shape[0:3]
    # Calculate geometry
    geom: Geometry = calc_geometry(lo=lo, hi=hi, shape=shape)
    # Plot style
    fig_style: bool = True

    # Compute the velocity difference, with and without NaNs
    vel_diff_n: NumpyFloatArray = sim_ups.velocity - sim_ref.velocity
    vel_diff: NumpyFloatArray = vel_ups - vel_ref
    # L2 norm of the velocity difference
    vel_diff_L2: float = np.linalg.norm(vel_diff)
    # L2 norm of the reference velocity
    vel_ref_L2: float = np.linalg.norm(vel_ref)
    # Relative L2 norm of the velocity difference
    vel_diff_rel: float = vel_diff_L2 / vel_ref_L2
    print(f'L2 norm of velocity difference:\n{vel_diff_rel:6.2e}')
    # Mask of refined cells
    mask = (sim_ups.refinement > 0)
    vel_diff_L2_ref: float = np.linalg.norm(vel_diff[mask])
    vel_ref_L2_ref: float = np.linalg.norm(vel_ref[mask])
    vel_diff_rel_ref: float = vel_diff_L2_ref / vel_ref_L2_ref
    print(f'L2 norm of velocity difference on refined cells:\n{vel_diff_rel_ref:6.2e}')

    # Speed of both simulations; leave the NaNs so they plot in white
    speed_ref: NumpyFloatArray = np.linalg.norm(sim_ref.velocity, axis=3)
    speed_ups: NumpyFloatArray = np.linalg.norm(sim_ups.velocity, axis=3)
    # The velocity error
    vel_err_abs: NumpyFloatArray = np.linalg.norm(vel_diff_n, axis=3)
    vel_ref_rms: float = np.sqrt(np.nanmean(np.square(speed_ref)))
    vel_err: NumpyFloatArray = vel_err_abs / vel_ref_rms
    # Selected slices
    for k in np.arange(0, geom.nz, 8):
        # z-coordinate of this slice
        zk: float = geom.zn[k]
        # Titles
        title_spd_ref: str = f'Speed (cm/s) at z = {zk:.0f} $\\mu$m - Reference'
        title_spd_ups: str = f'Speed (cm/s) at z = {zk:.0f} $\\mu$m - Upsampled'
        tex_err: str = r"$\left({||u'_{ijk} - u^{r}_{ijk}||}/{||u^{r}||} \right)$ "
        title_err: str = f'Relative Error {tex_err} at z = {zk:.0f} $\\mu$m'
        title_prs: str = f'Pressure (Pa) at z = {zk:.0f} $\\mu$m'
        # Paths
        path_spd_ref: Path = dir_plots / f'speed_ref_{k:03d}.png'
        path_spd_ups: Path = dir_plots / f'speed_ups_{k:03d}.png'
        path_err: Path = dir_plots / f'error_rel_{k:03d}.png'
        path_prs: Path = dir_plots / f'pressure_{k:03d}.png'
        # Plot the speed of the reference and upsampled simulations; velocity error and pressure
        plot_speed(speed=speed_ref, geom=geom, k=k, title=title_spd_ref, fig_style=fig_style, path=path_spd_ref)
        plot_speed(speed=speed_ups, geom=geom, k=k, title=title_spd_ups, fig_style=fig_style, path=path_spd_ups)
        plot_vel_err(vel_err=vel_err, geom=geom, k=k, title=title_err, fig_style=fig_style, path=path_err)
        plot_pressure(pressure=sim_ref.pressure, geom=geom, k=k, title=title_prs, fig_style=fig_style, path=path_prs)

# *************************************************************************************************
if __name__ == '__main__':
    main()
