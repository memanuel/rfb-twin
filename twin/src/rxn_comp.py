import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pathlib import Path

# Local imports
from utils import ReactionModel, ReactSim, Geometry, calc_geometry, geometry_add_types, geometry_add_measure
from plot_utils import plot_scalar, plot_scalar_yz
from explicit_utils import ReactSimExplicit, load_sim_mtu, load_sim_explicit, calc_util, calc_seq

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# *************************************************************************************************
# Pressure shared by all the simulations
p_in: int = 10

# Path to the special reaction simulation (simplified Butler-Volmer model)
top_dir_sp: Path = Path('07_react_special/P010/Vp000')
# Path to the Nernst reaction simulation
top_dir_ne: Path = Path('08_react_nernst/P010/Vp000')
# Path to the the Butler-Volmer reaction simulation
top_dir_bv: Path = Path('09_react_bv/P010/Vp000')

# Table: key = reaction model, value = directory
top_dir_tbl: dict[ReactionModel, Path] = {
    ReactionModel.Special           : top_dir_sp,
    ReactionModel.Nernst            : top_dir_ne,
    ReactionModel.ButlerVolmer      : top_dir_bv,
}

# *************************************************************************************************
# Path to save the plots
dir_plots: Path = Path('figs/06_react_ref')
dir_plots.mkdir(parents=True, exist_ok=True)

# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# *************************************************************************************************
def load_domain(dir_sim: Path) -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """Load the domain from file"""
    # Load the lower coordinates of the box and convert to microns
    lo_levs: NumpyFloatArray = np.load(dir_sim / 'domain_lo.npy')
    # Load the upper coordinates of the box
    hi_levs: NumpyFloatArray = np.load(dir_sim / 'domain_hi.npy')
    # Extract just the top level and convert from cm to microns
    # cm2um: float = 1.0E4
    # lo: NumpyFloatArray = lo_levs[0] * cm2um
    # hi: NumpyFloatArray = hi_levs[0] * cm2um
    # Extract the corners
    lo: NumpyFloatArray = lo_levs[0]
    hi: NumpyFloatArray = hi_levs[0]
    return lo, hi

# *************************************************************************************************
def load_sim_impl(dir_sim: Path, lev: int) -> ReactSim:
    """Load the numpy arrays for the reference simulation at n=128"""
    # Load the domain
    lo, hi = load_domain(dir_sim=dir_sim)
    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / f'L{lev:0d}'
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[lev]
    # Build the geometry
    geom: Geometry = calc_geometry(lo=lo, hi=hi, shape=grid_shape)

    # Load additional geometry data
    cell_type: NumpyInt16Array = np.load(dir_lev / 'cell_type.npy')
    refinement: NumpyInt16Array = np.load(dir_lev / 'refinement.npy')
    area: NumpyFloatArray = np.load(dir_lev / 'area.npy')
    volume: NumpyFloatArray = np.load(dir_lev / 'volume.npy')
    # Add cell types and measure to the geometry
    geometry_add_types(geom=geom, cell_type=cell_type, refinement=refinement)    
    geometry_add_measure(geom=geom, area=area, volume=volume)

    # Load the numpy arrays for this level
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    soc: NumpyFloatArray = np.load(dir_lev / 'soc.npy')
    # Load the current if available
    if (dir_lev / 'current.npy').exists():
        current: NumpyFloatArray = np.load(dir_lev / 'current.npy')
    else:
        current = None
    # Load the overpot if available
    if (dir_lev / 'overpot.npy').exists():
        overpot: NumpyFloatArray = np.load(dir_lev / 'overpot.npy')
    else:
        overpot = None
    # Load the epotL if available
    if (dir_lev / 'epotL.npy').exists():
        epot: NumpyFloatArray = np.load(dir_lev / 'epotL.npy')
    else:
        epot = None
    # Wrap the numpy arrays in a ReactSim object
    sim: ReactSim = ReactSim(
        geom=geom, p_in=p_in, velocity=velocity, soc=soc, current=current, overpot=overpot, epot=epot)
    return sim

# *************************************************************************************************
def load_sim_ref(model: ReactionModel) -> ReactSim:
    """Load the numpy arrays for the reference simulation with n=128"""
    # Directory with the numpy arrays for the requested reaction model
    top_dir: Path = top_dir_tbl[model]
    # The stage and level
    lev: int = 0
    stage: int = lev + 1
    # Simulation directory
    dir_sim: Path = top_dir / 'n128' / f'stage{stage:02d}' / 'numpy_last'
    # Delegate to load_sim_impl
    sim: ReactSim = load_sim_impl(dir_sim=dir_sim, lev=lev)
    return sim

# *************************************************************************************************
def load_sim_ups(model: ReactionModel) -> ReactSim:
    """Load the numpy arrays for the upsampled reaction simulation with n=64"""
    # Directory with the numpy arrays for the requested reaction model
    top_dir: Path = top_dir_tbl[model]
    # The stage and level
    lev: int = 1
    stage: int = lev + 1
    # Simulation directory
    dir_sim: Path = top_dir / 'n064' / f'stage{stage:02d}' / 'numpy_last'
    # Delegate to load_sim_impl
    sim: ReactSim = load_sim_impl(dir_sim=dir_sim, lev=lev)
    return sim

# *************************************************************************************************
def plot_soc(soc: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot the state of charge at a given z-slice
    INPUTS:
        soc: Array of SOC data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = 0.0
    vmax: float = np.nanmax(soc[:,:,k]).astype(float)
    cmap: Colormap = plt.get_cmap('jet')
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar(phi=soc, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_soc_diff(soc_diff: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot the state of charge at a given z-slice
    INPUTS:
        speed: Array of speed data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = np.nanmin(soc_diff[:,:,k]).astype(float)
    vmax: float = np.nanmax(soc_diff[:,:,k]).astype(float)
    cmap: Colormap = plt.get_cmap('jet')
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar(phi=soc_diff, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_pot(pot: NumpyFloatArray, geom: Geometry, k: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot a potential at a given z-slice
    INPUTS:
        epot: Array of potential data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = np.nanmin(pot[:,:,k]).astype(float)
    vmax: float = np.nanmax(pot[:,:,k]).astype(float)
    cmap: Colormap = plt.get_cmap('jet')
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar(phi=pot, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_pot_yz(pot: NumpyFloatArray, geom: Geometry, i: int, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot a potential at a given x-slice
    INPUTS:
        epot: Array of potential data to plot; shape (nx, ny, nz)
        geom: Geometry object
        i: Index of x-slice to plot
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Set plot options
    vmin: float = np.nanmin(pot[i,:,:]).astype(float)
    vmax: float = np.nanmax(pot[i,:,:]).astype(float)
    cmap: Colormap = plt.get_cmap('jet')
    cbar_fmt: str = '%0.03f'
    # Delegate to plot_scalar
    plot_scalar_yz(phi=pot, geom=geom, i=i, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_ref_sim(fig_style: bool):
    """Plot SOC for the reference simulation in the Special model"""
    # Load the reference reaction simulation - special model
    sim: ReactSim = load_sim_ref(model=ReactionModel.Special)
    # The geometry
    geom: Geometry = sim.geom

    # Selected z-slice
    k: int = geom.nz // 2
    z: float = geom.zn[k]

    # Plot the SOC
    title: str = f'SOC at z = {z:.0f} $\\mu$m'
    path: Path = dir_plots / f'soc_{k:03d}.png'
    plot_soc(soc=sim.soc, geom=geom, k=k, title=title, fig_style=fig_style, path=path)

    # Load the reference reaction simulation - BV model
    sim = load_sim_ref(model=ReactionModel.ButlerVolmer)

    # Overwrite potential with NaN on solid cells
    sim.epot[sim.geom.is_solid] = np.nan

    # Plot the overpotential
    eta_tex: str = r'$\eta_{\textrm{act}}$'
    title = f'Overpotential {eta_tex} at z = {z:.0f} $\\mu$m (mV)'
    path = dir_plots / f'overpot_{k:03d}.png'
    plot_pot(pot=sim.overpot, geom=geom, k=k, title=title, fig_style=fig_style, path=path)

    # Plot the potential in the liquid
    title = f'Electric Potential $\\phi_L$ at z = {z:.0f} $\\mu$m (mV)'
    path = dir_plots / f'epot_{k:03d}.png'
    plot_pot(pot=sim.epot, geom=geom, k=k, title=title, fig_style=fig_style, path=path)

    # Selected x-slice
    i: int = geom.nx // 2
    x: float = geom.xn[i]

    # Plot the potential in the liquid in the YZ plane
    title = f'Electric Potential $\\phi_L$ at x = {x:.0f} $\\mu$m (mV)'
    path = dir_plots / f'epot_YZ_{i:03d}.png'
    plot_pot_yz(pot=sim.epot, geom=geom, i=i, title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_model_diff(model: ReactionModel, fig_style: bool):
    """Plot SOC for the reference simulation in the Special model"""
    # Load the reference reaction simulation
    sim_sp: ReactSim = load_sim_ref(model=ReactionModel.Special)
    # Load the selected reaction model
    sim: ReactSim = load_sim_ref(model=model)
    # The geometry
    geom: Geometry = sim.geom
    # Difference in SOC fields
    soc_diff: NumpyFloatArray = sim.soc - sim_sp.soc

    # Selected z-slice
    k: int = geom.nz // 2
    z: float = geom.zn[k]

    # Plot the SOC difference
    title: str = f'SOC Difference at z = {z:.0f} $\\mu$m'
    path: Path = dir_plots / f'soc_diff_{model.name}_{k:03d}.png'
    plot_soc_diff(soc_diff=soc_diff, geom=geom, k=k, title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def compare_upsampled():
    """Compare the reference reaction simulation to the upsampled simulation"""
    # Report the RMS difference
    print(f'\nRMS difference in SOC fields by model type:')
    # Iterate over all three reaction models
    for model in [ReactionModel.Special,  ReactionModel.Nernst, ReactionModel.ButlerVolmer]:
        # Load the reference reaction simulation
        sim_ref: ReactSim = load_sim_ref(model=model)
        # Load the upsampled reaction simulation
        sim_ups: ReactSim = load_sim_ups(model=model)
        # Calculate the RMS difference in SOC fields
        rms: float = np.sqrt(np.nanmean(np.square(sim_ups.soc - sim_ref.soc)))
        # Report the RMS difference
        print(f'{model.name:12}: {rms:8.2e}')

# *************************************************************************************************
def compare_explicit():
    """Compare the upsampled special reaction simulation to the explicit Nernst model"""
    # Load the upsampled Nernst reaction simulation
    sim_ups: ReactSim = load_sim_ups(model=ReactionModel.Nernst)

    # Operating conditions for the explicit simulations (MTU and Nernst)
    P: int = 10
    V: int = 0
    lev: int = 1

    # Equilibrium SOC at this potential
    seq: float = calc_seq(V=V)

    # Load the mass transport simulation
    sim_mtu: ReactSimExplicit = load_sim_mtu(P=p_in, lev=lev)

    # Load the Explicit Nernst reaction simulation
    sim_exp: ReactSimExplicit = load_sim_explicit(P=P, V=V, lev=lev)

    # Calculate the RMS difference in SOC fields
    rms: float = np.sqrt(np.nanmean(np.square(sim_ups.soc - sim_exp.soc)))
    # Report the RMS difference
    print(f'\nRMS difference in SOC fields - Explicit Nernst vs. Nernst:')
    print(f'{rms:8.2e}')

    # Calculate utilization of each model
    util_ups: float = calc_util(geom=sim_ups.geom, velocity=sim_ups.velocity, soc=sim_ups.soc)
    util_exp: float = calc_util(geom=sim_exp.geom, velocity=sim_exp.velocity, soc=sim_exp.soc)    
    util_mtu: float = calc_util(geom=sim_mtu.geom, velocity=sim_mtu.velocity, soc=sim_mtu.soc)
    # Limiting utilization at this voltage for inifinitely fast kinetics
    util_kin: float = seq * util_mtu

    # Report utilization of each model
    print(f'\nUtilization by model:')
    print(f'Standard Nernst model : {util_ups:0.6f}')
    print(f'Explicit Nernst model : {util_exp:0.6f}')
    print(f'Fast kinetic limit    : {util_kin:0.6f}\n')
    print(f'Mass transport limit  : {util_mtu:0.6f}')
    print(f'Equilibrium SOC       : {seq:0.6f}')

# *************************************************************************************************
def compare_models():
    """Compare the refernce reaction simulations in three different models"""
    # Load the reference reaction simulation for all three models
    sim_sp: ReactSim = load_sim_ref(model=ReactionModel.Special)
    sim_ne: ReactSim = load_sim_ref(model=ReactionModel.Nernst)
    sim_bv: ReactSim = load_sim_ref(model=ReactionModel.ButlerVolmer)
    sim_ex: ReactSim = load_sim_ref(model=ReactionModel.ExplicitNernst)

    # Calculate difference in SOC fields
    rms_ne: float = np.sqrt(np.nanmean(np.square(sim_ne.soc - sim_sp.soc)))
    rms_bv: float = np.sqrt(np.nanmean(np.square(sim_bv.soc - sim_sp.soc)))
    rms_ex: float = np.sqrt(np.nanmean(np.square(sim_ex.soc - sim_sp.soc)))

    # Report the RMS differences
    print(f'\nRMS difference in SOC fields vs. Special model:')
    print(f'Nernst model          : {rms_ne:8.2e}')
    print(f'Butler-Volmer model   : {rms_bv:8.2e}')
    print(f'Explicit Nernst model : {rms_ex:8.2e}')

    # Calculate the mean of phiL
    phi_mean: float = np.nanmean(sim_bv.epot)
    phi_std: float = np.nanstd(sim_bv.epot)
    phi_min: float = np.nanmin(sim_bv.epot)
    phi_max: float = np.nanmax(sim_bv.epot)

    # Report the summary of electric potential in the liquid
    print(f'\nSummary of electric potential in the liquid:')
    print(f'Mean: {phi_mean:0.3f} mV')
    print(f'Std : {phi_std :0.3f} mV')
    print(f'Min : {phi_min :0.3f} mV')
    print(f'Max : {phi_max :0.3f} mV')

# *************************************************************************************************
def main():
    """Compare the diffrent models and compare mesh refined vs. reference simulations"""

    # Use the figure style for the plots
    fig_style: bool = True

    # Plot the reference simulation
    plot_ref_sim(fig_style=fig_style)

    # Plot difference between special and BV model
    # plot_model_diff(model=ReactionModel.ButlerVolmer, fig_style=fig_style)

    # Compare the mesh refined simulations
    # compare_upsampled(fig_style=fig_style)

    # Compare different types of models
    # compare_models()

    # Compare the explicit Nernst model to the upsampled Nernst model
    # compare_explicit(fig_style=fig_style)

# *************************************************************************************************
if __name__ == '__main__':
    main()
