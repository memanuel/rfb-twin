import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from pathlib import Path
import warnings

# Local imports
from cnvg_utils import SimOutput, geom_name_tbl, subdir_tbl, plot_dir, fit_error_nonlinear

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]

# *************************************************************************************************
# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# Dictionary with key = geometry number and value = nz_ref for the geometry
nz_ref_tbl: dict[int, int] = {
    1: 120,
    2: 120,
    3: 120,
}

# Dictionary with key = (geometry number, level) and value = list of nzs for the geometry for steady flow
nzs_tbl: dict[tuple[int, int], list[int]] = {
    (1, 0): [6, 8, 10, 12, 15, 20, 24, 30, 40, 60,],
    (2, 0): [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60,],
    (3, 0): [5, 6, 8, 10, 12, 15, 20, 24, 40, 60],
}

# *************************************************************************************************
def load_react_sim(G: int, nz: int, stage: int, level: int) -> SimOutput:
    """
    Load numpy arrays from one stage of a completed simulation.
    INPUTS:
    G: Number keying the geometry
    nz: Number of grid points in the z-direction
    stage: Which stage to load; stage n has n-1 levels of refinement
    level: Which level to load; level 0 is the coarsest grid
    """
    # The top level directory name for this flow convergence study
    dir_top: str = '03_cnvg_react'
    # The name of the subdirectory for this geometry
    sub_dir: str = subdir_tbl[G]
    # The directory containing the simulation numpy output for this stage
    dir_sim: Path = Path(f'{dir_top}/{sub_dir}/n{nz:03d}/stage{stage:02d}/numpy_last')
    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / f'L{level:0d}'
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # The number of levels of mesh refinement
    refinement: NumpyInt16Array = np.load(dir_lev / 'refinement.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[level]
    # Load the numpy arrays for this level
    cell_type: NumpyFloatArray = np.load(dir_lev / 'cell_type.npy')
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    pressure: NumpyFloatArray = np.load(dir_lev / 'pressure_cell.npy')
    soc: NumpyFloatArray = np.load(dir_lev / 'soc.npy')
    # Wrap the numpy arrays in a SimOutput object
    sim_out: SimOutput = SimOutput(grid_shape=grid_shape, refinement=refinement, cell_type=cell_type, 
                                   velocity=velocity, pressure=pressure, soc=soc)
    return sim_out

# *************************************************************************************************
def soc_diff_rms(s_ref: SimOutput, s: SimOutput) -> float:
    """
    Compute the root mean square of the difference in state of charge fields between two simulations.
    Measure the difference on the coarse grid.
    INPUTS:
    s_ref: Reference simulation
    s: Comparison simulation
    RETURNS:
    rms: Root mean square of the difference in SOC fields (dimensionless)
    """
    # refinement ratio
    rr: int = s_ref.grid_shape[0] // s.grid_shape[0]
    # SOC on the coarse grid; convert NaNs to zeros
    s1 = np.nan_to_num(s.soc, nan=0.0)
    # SOC on the fine grid, averaged over the refinement ratio
    nx, ny, nz = s.grid_shape
    s2_fine = s_ref.soc.reshape((nx, rr, ny, rr, nz, rr))
    axis: tuple[int, ...] = (1, 3, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s2 = np.nan_to_num(np.nanmean(s2_fine, axis=axis), nan=0.0)
    # difference in velocity fields
    soc_diff = np.nan_to_num(s1 - s2, nan=0.0)
    # L2 norm of the difference
    rms = np.sqrt(np.mean(np.square(soc_diff)))
    return float(rms)

# *************************************************************************************************
def soc_error(G: int, nz_ref: int, nzs: list[int], stage: int, level: int) \
    -> NumpyFloatArray:
    """
    Compute the RMS of the difference in SOC fields between two simulations.
    INPUTS:
    G: Number keying the geometry
    nz_ref: Number of grid points in the z-direction for the reference simulation
    nzs: List of grid points in the z-direction for the comparison simulations
    stage: Which stage to load; stage n has n-1 levels of refinement
    level: Which level to load; level 0 is the coarsest grid
    RETURNS:
    err: Array of L2 norms of the difference in velocity fields
    """
    # Initialize empty array of differences
    sz: int = len(nzs)
    err: NumpyFloatArray = np.zeros(sz)    
    # Load the reference simulation; this is always stage01 (level 0) for now
    stage_ref: int = 1
    level_ref: int = 0
    s_ref: SimOutput = load_react_sim(G=G, nz=nz_ref, stage=stage_ref, level=level_ref)
    # Iterate over simulations to compare
    for i, nz in enumerate(nzs):
        # Load the comparison simulation
        s: SimOutput = load_react_sim(G=G, nz=nz, stage=stage, level=level)
        # Compute the RMS of the difference in relative velocity fields
        err[i] = soc_diff_rms(s_ref=s_ref, s=s)
    return err

# *************************************************************************************************
def plot_soc_error(nzs: list[int], err: NumpyFloatArray, nz_ref: int, G: int, level: int, 
                   fig_style: bool, verbose: bool) -> None:
    """
    Plot the L2 norm of the difference in velocity fields between two simulations.
    INPUTS:
    nzs:        List of grid points in the z-direction for the comparison simulations
    err:        Array of L2 norms of the difference in velocity fields
    nz_ref:     Number of grid points in the z-direction for the reference simulation
    G:          Number keying the geometry
    fig_style:  Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    verbose:    Print the errors to the console
    """
    # Name of this geometry
    geom_name: str = geom_name_tbl[G]
    
    # Report the velocity error if requested
    if verbose:
        print(f'Errors for reaction convergence - geometry {G:d}, level {level:0d}:')
        for i, nz in enumerate(nzs):
            diff: float = err[i]
            print(f'{nz:2d}: {diff:5.2e}')

    # Invert the grid points for the x-axis; h = 1/nz
    h: NumpyFloatArray = 1.0 / np.array(nzs)
    h_ref: float = 1.0 / nz_ref
    # Sample values of h for fitting
    hs = np.linspace(h[0], h[-1], 101)
    # Ratios
    rs: NumpyFloatArray = h_ref / hs

    # Calculate the nonlinear fit
    C: float
    p: float
    alpha: float
    rmse: float
    C, p, alpha, rmse = fit_error_nonlinear(h=h, err=err, h_ref=h_ref)
    err_fit = C * np.power(hs, p) * (1.0 - np.power(rs, p))

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Set plot title
    if not fig_style:
        ax.set_title(f'Reaction Convergence - {geom_name:s}', fontsize=20)
    else:
        ax.set_title(None)
    # Set axis labels
    fontsize: int = 20 if fig_style else 14
    ax.set_xlabel('Relative Step Size $h$ (dimensionless)', fontsize=fontsize)
    ax.set_ylabel('RMS of SOC Difference ', fontsize=fontsize)
    #  r'$\sqrt\left(\frac{1}{n} \sum{(s_i - s^R_i)^2}\right)$', 
    # Set log scale with base 2 and add legend
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # Plot the data and the linear fit
    markersize: float = 10.0 if fig_style else 6.0
    linewidth = 2.0 if fig_style else 1.0
    ax.plot(h, err, marker='o', markersize=markersize, color='red', linestyle='None', label='Data')    
    ax.plot(hs, err_fit, color='black', linewidth=linewidth, label='Fit')

    # Add grid lines and legend
    gridwidth: float = 1.0 if fig_style else 0.5
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=gridwidth)
    legendsize: int = 14 if fig_style else 10
    ax.legend(fontsize=legendsize, frameon=True, fancybox=True, facecolor='white', edgecolor='black')

    # Increase the font size of the tick labels
    labelsize: int = 16 if fig_style else 13
    ax.tick_params(axis='both', labelsize=labelsize)

    # Parameter values and tex string for series label
    tex: str = r'$\epsilon(h) = C h^p (1 - \alpha r^p)$'
    params: str = f'$C={C:0.2f}, p={p:0.3f}, \\alpha={alpha:0.3f}$'
    err_str: str = f'RMSE={rmse:0.4f}'
    label: str = f'{tex:s}\n({params:s}\n{err_str:s})'
    # Textbox with the fit parameters
    bbox: dict = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}
    fontsize = 15 if fig_style else 10
    y_min, y_max = ax.get_ylim()
    text_x: float = np.power(2.0, -3.85)
    text_y: float = y_min * np.power(y_max / y_min, 0.20)
    ax.text(text_x, text_y, label, fontsize=fontsize, 
            horizontalalignment='left', verticalalignment='top', bbox=bbox)

    # Tight layout for the figure because there is no title
    if fig_style:
        fig.tight_layout()

    # Save the figure
    fname: str = (plot_dir / f'react_cnvg_{geom_name}.png').as_posix()
    fig.savefig(fname, bbox_inches='tight')

    # Status message
    print(f'Fit: err(h) = C h^p (1 - r^p).')
    print(f'C = {C:0.3f}')
    print(f'p = {p:0.6f}')
    print(f'alpha = {alpha:0.6f}')
    print(f'RMSE = {rmse:0.6f}')

# *************************************************************************************************
def main():
    """Generate plots of the SOC error for the reaction convergence study"""

    # Type declarations
    G: int
    nz_ref: int
    nzs: list[int]

    # Shared parameters all the convergence studies
    level: int = 0
    fig_style: bool = True
    verbose: bool = True

    # Geometries to process
    Gs: list[int] = [3]
    
    # Iterate over geometries
    for G in Gs:
        # The stage number for this level
        stage: int = level + 1
        # The reference nz for this geometry (will always be 120)
        nz_ref: int = nz_ref_tbl[G]
        # Select the table of nz values for this convergence type, geometry and level
        nzs: list[int] = nzs_tbl[(G, level)]
        # Get the velocity error to the reference simulation for this convergence type and geometry
        err: NumpyFloatArray = soc_error(G=G, nz_ref=nz_ref, nzs=nzs, stage=stage, level=level)
        # Plot the velocity error
        plot_soc_error(nzs=nzs, err=err, nz_ref=nz_ref, G=G, level=level, fig_style=fig_style, verbose=verbose)

# *************************************************************************************************
if __name__ == '__main__':
    main()
