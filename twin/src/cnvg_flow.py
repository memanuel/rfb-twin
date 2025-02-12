import numpy as np
import numpy.typing as npt
from enum import Enum
from matplotlib import pyplot as plt
from pathlib import Path
import warnings

# Local imports
from cnvg_utils import SimOutput, geom_name_tbl, subdir_tbl, plot_dir, fit_error_nonlinear

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]

# *************************************************************************************************
# Types of flow convergence studies
class FlowConvergenceType(Enum):
    Steady = 1
    Fixed = 2

# Dictionary with key = type of flow convergence, value = directory name
flow_cnvg_dir_tbl: dict[FlowConvergenceType, str] = {
    FlowConvergenceType.Steady  : '01_cnvg_flow',
    FlowConvergenceType.Fixed   : '02_cnvg_flow_tfix',
}

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
nzs_tbl_steady: dict[tuple[int, int], list[int]] = {
    (1, 0): [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60],
    (1, 1): [5, 6, 10, 12, 15, 20, 30, 60],
    # (1, 2): [5, 6, 8, 10, 12, 15, 20, 24, 30, ],

    (2, 0): [5, 6, 10, 12, 15, 15, 20, 30, 40, 60],
    (2, 1): [5, 6, 10, 20, 60],
    # (2, 2): [5, 6, 10, 12],

    (3, 0): [5, 8, 10, 12, 15, 20, 24, 30, 40, 60,],
    (3, 1): [5, 6, 10, 12, 15, 20, 30],
    # (3, 2): [5, 8, 10, 15,],
}

# Dictionary with key = (geometry number, level) and value = list of nzs for the geometry for fixed time
nzs_tbl_fixed: dict[tuple[int, int], list[int]] = {
    (1, 0): [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60,],
    (1, 1): [6, 10, 12, 15, 20, 30, 60,],
    # (1, 2): [5, 6, 10, 15, 30, ],

    (2, 0): [5, 6, 10, 12, 15, 20, 30, 40, 60],
    (2, 1): [5, 6, 10, 20, 60,],

    (3, 0): [5, 8, 10, 12, 15, 20, 24, 30, 40, 60,],
    (3, 1): [5, 10, 12, 15, 20, 30],
}

# Dictionary with key = (FlowConvergenceType, geometry number, level) and value = list of nzs
nzs_tbl: dict[tuple[FlowConvergenceType, int, int], list[int]] = dict()
for (G, level), nzs in nzs_tbl_steady.items():
    nzs_tbl[(FlowConvergenceType.Steady, G, level)] = nzs
for (G, level), nzs in nzs_tbl_fixed.items():
    nzs_tbl[(FlowConvergenceType.Fixed, G, level)] = nzs

# *************************************************************************************************
def load_flow_sim(cnvg_type: FlowConvergenceType, G: int, nz: int, stage: int, level: int) -> SimOutput:
    """
    Load numpy arrays from one stage of a completed simulation.
    INPUTS:
    cnvg_type: Type of flow convergence study; 
    G: Number keying the geometry
    nz: Number of grid points in the z-direction
    stage: Which stage to load; stage n has n-1 levels of refinement
    level: Which level to load; level 0 is the coarsest grid
    """
    # The top level directory name for this flow convergence study
    dir_top: str = flow_cnvg_dir_tbl[cnvg_type]
    # The name of the subdirectory for this geometry
    sub_dir: str = subdir_tbl[G]
    # The directory containing the simulation numpy output for this stage
    # dir_sim: Path = Path(f'01_cnvg_flow/{sub_dir}/n{nz:03d}/numpy/stage{stage:02d}')
    dir_sim: Path = Path(f'{dir_top}/{sub_dir}/n{nz:03d}/stage{stage:02d}/numpy_last')
    # Directory with numpy arrays for the requested level
    dir_lev: Path = dir_sim / f'L{level:0d}'
    # Load the grid shapes for all the levels
    grid_shapes = np.load(dir_sim / 'grid_shape.npy')
    # Select the grid shape for this level
    grid_shape = grid_shapes[level]
    # The number of levels of mesh refinement
    refinement: NumpyInt16Array = np.load(dir_lev / 'refinement.npy')
    # Load the numpy arrays for this level
    cell_type: NumpyFloatArray = np.load(dir_lev / 'cell_type.npy')
    velocity: NumpyFloatArray = np.load(dir_lev / 'velocity.npy')
    pressure: NumpyFloatArray = np.load(dir_lev / 'pressure_cell.npy')
    soc = np.zeros(grid_shape)
    # Wrap the numpy arrays in a SimOutput object
    sim_out: SimOutput = SimOutput(grid_shape=grid_shape, refinement=refinement, cell_type=cell_type, 
                                   velocity=velocity, pressure=pressure, soc=soc)
    return sim_out

# *************************************************************************************************
def vel_diff_l2_coarse(s_ref: SimOutput, s: SimOutput) -> float:
    """
    Compute the relative L2 norm of the difference in velocity fields between two simulations.
    Measure the difference on the coarse grid.
    INPUTS:
    s_ref: Reference simulation
    s: Comparison simulation
    RETURNS:
    l2_norm: L2 norm of the difference in velocity fields; relative to the comparison simulation (dimensionless)
    """
    # refinement ratio
    rr: int = s_ref.grid_shape[0] // s.grid_shape[0]
    # velocity on the coarse grid; convert NaNs to zeros
    v1 = np.nan_to_num(s.velocity, nan=0.0)
    # DEBUG
    # print(f's_ref.grid_shape: {s_ref.grid_shape}')
    # print(f'    s.grid_shape: {s.grid_shape}')
    # print(f'rr: {rr}')
    # velocity on the fine grid, averaged over the refinement ratio
    nx, ny, nz = s.grid_shape
    v2_fine = s_ref.velocity.reshape((nx, rr, ny, rr, nz, rr, 3))
    axis: tuple[int, ...] = (1, 3, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        v2 = np.nan_to_num(np.nanmean(v2_fine, axis=axis), nan=0.0)
    # difference in velocity fields
    vel_diff = np.nan_to_num(v1 - v2, nan=0.0)
    # L2 norm of the difference
    l2_norm = np.linalg.norm(vel_diff) / np.linalg.norm(v2)
    return float(l2_norm)

# *************************************************************************************************
def vel_diff_l2_fine(s_ref: SimOutput, s: SimOutput) -> float:
    """
    Compute the relative L2 norm of the difference in velocity fields between two simulations.
    Measure the difference on the fine grid.
    INPUTS:
    s_ref: Reference simulation
    s: Comparison simulation
    RETURNS:
    l2_norm: L2 norm of the difference in velocity fields; relative to the comparison simulation (dimensionless)
    """
    # Velocity on the fine grid
    v2 = np.nan_to_num(s_ref.velocity, nan=0.0)
    # refinement ratio
    rr: int = s_ref.grid_shape[0] // s.grid_shape[0]
    # velocity on the coarse grid; convert NaNs to zeros
    v1_coarse = np.nan_to_num(s.velocity, nan=0.0)
    # Matrix to upsample the coarse grid to the fine grid in the Kronecker delta
    up_mat: NumpyFloatArray = np.ones((rr, rr, rr, 1))
    # Upsample v1 to the fine grid
    v1 = np.kron(v1_coarse, up_mat)
    # difference in velocity fields
    vel_diff = np.nan_to_num(v1 - v2, nan=0.0)
    # L2 norm of the difference
    l2_norm = np.linalg.norm(vel_diff) / np.linalg.norm(v2)
    return float(l2_norm)

# *************************************************************************************************
def vel_diff_l2(s_ref: SimOutput, s: SimOutput, use_fine: bool) -> float:
    """
    Compute the relative L2 norm of the difference in velocity fields between two simulations.
    Measure the difference on the fine grid.
    INPUTS:
    s_ref: Reference simulation
    s: Comparison simulation
    use_fine: Measure the difference on the fine grid? (True) or the coarse grid? (False)
    RETURNS:
    l2_norm: L2 norm of the difference in velocity fields; relative to the comparison simulation (dimensionless)
    """
    vd: float = vel_diff_l2_fine(s_ref=s_ref, s=s) if use_fine else vel_diff_l2_coarse(s_ref=s_ref, s=s)
    return vd

# *************************************************************************************************
def vel_error(cnvg_type: FlowConvergenceType, G: int, nz_ref: int, nzs: list[int], 
              stage: int, level: int, use_fine: bool) \
    -> NumpyFloatArray:
    """
    Compute the L2 norm of the difference in velocity fields between two simulations.
    INPUTS:
    cnvg_type: Type of flow convergence study
    G: Number keying the geometry
    nz_ref: Number of grid points in the z-direction for the reference simulation
    nzs: List of grid points in the z-direction for the comparison simulations
    stage: Which stage to load; stage n has n-1 levels of refinement
    level: Which level to load; level 0 is the coarsest grid
    use_fine: Measure the difference on the fine grid? (True) or the coarse grid? (False)
    RETURNS:
    err: Array of L2 norms of the difference in velocity fields
    """
    # Initialize empty array of differences
    sz: int = len(nzs)
    err: NumpyFloatArray = np.zeros(sz)    
    # Load the reference simulation; this is always stage01 (level 0) for now
    stage_ref: int = 1
    level_ref: int = 0
    s_ref: SimOutput = load_flow_sim(cnvg_type=cnvg_type, G=G, nz=nz_ref, stage=stage_ref, level=level_ref)
    # Iterate over simulations to compare
    for i, nz in enumerate(nzs):
        # Load the comparison simulation
        s: SimOutput = load_flow_sim(cnvg_type=cnvg_type, G=G, nz=nz, stage=stage, level=level)
        # Compute the L2 norm of the difference in relative velocity fields
        err[i] = vel_diff_l2(s_ref=s_ref, s=s, use_fine=use_fine)
    return err

# *************************************************************************************************
def plot_vel_error(nzs: list[int], err: NumpyFloatArray, nz_ref: int, G: int, level: int, verbose: bool) -> None:
    """
    Plot the L2 norm of the difference in velocity fields between two simulations.
    INPUTS:
    nzs:    List of grid points in the z-direction for the comparison simulations
    err:    Array of L2 norms of the difference in velocity fields
    nz_ref: Number of grid points in the z-direction for the reference simulation
    G:      Number keying the geometry
    verbose: Print the errors to the console
    """
    # Always build this plot using the fixed time convergence type
    cnvg_type: FlowConvergenceType = FlowConvergenceType.Fixed
    # Name of this convergence type
    cnvg_type_name: str = cnvg_type.name
    # Name of this geometry
    geom_name: str = geom_name_tbl[G]
    # Number of this plot
    # plot_num: int = 3*(cnvg_type.value - 1) + G
    plot_num: int = G
    
    # Report the velocity error if requested
    if verbose:
        print(f'Errors for flow convergence type {cnvg_type_name:s} and geometry {G:d}, level {level:0d}:')
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
    err_fit = C * np.power(hs, p) * (1.0 - alpha * np.power(rs, p))

    # Parameter values and tex string for series label
    tex: str = r'$\epsilon(h) = C h^p (1 - \alpha r^p)$'
    params: str = f'$C={C:0.1f}, p={p:0.3f}, \\alpha={alpha:0.3f}$'
    err_str: str = f'RMSE={rmse:0.4f}'
    label: str = f'{tex:s}\n({params:s}\n{err_str:s})'

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the data and the linear fit
    ax.plot(h, err, marker='o', color='red', linestyle='None', label='Data')    
    ax.plot(hs, err_fit, color='black', linewidth=2, label=label)
    # Set log scale with base 2 and add legend
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.legend(frameon=True, fancybox=True)
    # Set title and axis labels
    # ax.set_title(f'Flow Convergence - {geom_name:s}, Level {level:0d}', fontsize=20)
    ax.set_title(f'Flow Convergence - {geom_name:s}', fontsize=20)
    ax.set_xlabel('Relative Step Size $h$ (dimensionless)', fontsize=14)
    ax.set_ylabel('$L^2$ Norm of Velocity Difference ' + 
                  r'$\left(\frac{||u - u_{\tiny\textrm{r}}||}{||u_{\tiny\textrm{r}}||} \right)$', 
                  fontsize=14)

    # Save the figure
    # fname: str = (plot_dir / f'{plot_num:02d}_flow_cnvg_{cnvg_type_name.lower():s}_L{level:0d}.png').as_posix()
    fname: str = (plot_dir / f'{plot_num:02d}_flow_cnvg_L{level:0d}.png').as_posix()
    fig.savefig(fname)

    # Status message
    print(f'Fit: err(h) = C h^p (1 - r^p). {params:s}. {err_str:s}.')
    print(f'C = {C:0.3f}.')
    print(f'p = {p:0.6f}.')
    print(f'alpha = {alpha:0.6f}.')

# *************************************************************************************************
def plot_vel_errors(nzs3: list[list[int]], errs3: list[NumpyFloatArray], 
                    nz_ref: int, level: int, fig_style: bool) -> None:
    """
    Plot the L2 norm of the difference in velocity fields between two simulations, 
    for all three geometries on a single plot.
    INPUTS:
    nzs3:   List of nzs for each geometry. nzs[g] is list of grid points 
            in the z-direction for the comparison simulations
    errs3:  List of err fo each geometry.
            errs[g] is array of L2 norms of the difference in velocity fields on grid g.
    nz_ref: Number of grid points in the z-direction for the reference simulation; shared by all three.
    level:  Level of mesh refinement; shared by all three.
    fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    # Invert the grid points for the x-axis; h = 1/nz
    h3: list[NumpyFloatArray] = [1.0 / np.array(nzs) for nzs in nzs3]
    h_ref: float = 1.0 / nz_ref
    # Sample values of h for fitting
    hs3: list[NumpyFloatArray] = [np.linspace(h[0], h[-1], 101) for h in h3]
    # Ratios
    rs3: list[NumpyFloatArray] = [h_ref / hs for hs in hs3]

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Colors for the three geometries
    colors: list[str] = ['blue', 'green', 'red',]

    # Process each geometry in specified order
    for g in [2, 0, 1]:
        # Unpack the data for this geometry
        h: NumpyFloatArray = h3[g]
        hs: NumpyFloatArray = hs3[g]
        rs: NumpyFloatArray = rs3[g]
        err: NumpyFloatArray = errs3[g]

        # Calculate the nonlinear fit
        C: float
        p: float
        alpha: float
        rmse: float
        C, p, alpha, rmse = fit_error_nonlinear(h=h, err=err, h_ref=h_ref)
        err_fit = C * np.power(hs, p) * (1.0 - alpha * np.power(rs, p))

        # Geometry number
        G: int = g + 1
        # Name of this geometry
        geom_name: str = geom_name_tbl[G]
        # Color for this geometry
        color: str = colors[g]

        # Parameter values and tex string for series label
        label_data: str = f'Data - {geom_name:s}'
        label_fit: str = f'Fit - {geom_name:s}'

        # Plot the data and the linear fit
        markersize: float = 10.0 if fig_style else 6.0
        linewidth = 2.0 if fig_style else 1.0
        ax.plot(h, err, marker='o', markersize=markersize, color=color, linestyle='None', label=label_data)
        ax.plot(hs, err_fit, color=color, linewidth=linewidth, alpha=0.5, label=label_fit)
        # Set log scale with base 2 and add legend
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        # Set plot title
        if not fig_style:
            ax.set_title(f'Flow Convergence', fontsize=20)
        else:
            ax.set_title(None)
        # Set axis labels
        fontsize: int = 20 if fig_style else 14
        ax.set_xlabel('Relative Step Size $h$ (dimensionless)', fontsize=fontsize)
        ax.set_ylabel('$L^2$ Norm of Velocity Difference ' + 
                    r'$\left(\frac{||u - u_{\tiny\textrm{r}}||}{||u_{\tiny\textrm{r}}||} \right)$', 
                    fontsize=fontsize)
        # Increase the font size of the tick labels
        labelsize: int = 16 if fig_style else 13
        ax.tick_params(axis='both', labelsize=labelsize)

    # Add grid lines and legend
    gridwidth: float = 1.0 if fig_style else 0.5
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=gridwidth)
    legendsize: int = 14 if fig_style else 10
    ax.legend(fontsize=legendsize, frameon=True, fancybox=True, facecolor='white', edgecolor='black')

    # Tight layout for the figure because there is no title
    if fig_style:
        fig.tight_layout()

    # Save the figure
    fname: str = f'flow_cnvg.png' if fig_style else f'flow_cnvg_L{level:0d}.png'
    path: Path = (plot_dir / fname).as_posix()
    fig.savefig(path, bbox_inches='tight')

# *************************************************************************************************
def main():
    """Generate plots of the velocity error for the flow convergence study"""

    # Type declarations
    G: int
    nz_ref: int
    nzs: list[int]

    # Shared parameters for the stage and level
    cnvg_type: FlowConvergenceType = FlowConvergenceType.Fixed
    use_fine: bool = False
    verbose: bool = True

    # Geometries to process
    Gs: list[int] = []

    # Iterate over geometries
    for G in Gs:
        # Iterate over levels
        for level in [0, ]:
            # The stage number for this level
            stage: int = level + 1
            # The reference nz for this geometry (will always be 120)
            nz_ref: int = nz_ref_tbl[G]
            # Select the table of nz values for this convergence type, geometry and level
            nzs: list[int] = nzs_tbl[(cnvg_type, G, level)]
            # Get the velocity error to the reference simulation for this convergence type and geometry
            err: NumpyFloatArray = vel_error(
                cnvg_type=cnvg_type, G=G, nz_ref=nz_ref, nzs=nzs, stage=stage, level=level, use_fine=use_fine)
            # Plot the velocity error
            plot_vel_error(nzs=nzs, err=err, nz_ref=nz_ref, G=G, level=level, verbose=verbose)
    
    # Set plot options
    Gs_all: list[int] = [1, 2, 3]
    level: int = 0
    stage: int = level + 1
    nz_ref: int = 120
    nzs3: list[list[int]] = []
    errs3: list[NumpyFloatArray] = []
    fig_style: bool = True

    # Report the velocity error if requested
    if verbose:
        print(f'Errors for flow convergence type {cnvg_type.name:s}, level {level:0d}:')

    # Iterate over geometries
    for G in Gs_all:
            # Select the table of nz values for this convergence type, geometry and level
            nzs: list[int] = nzs_tbl[(cnvg_type, G, level)]
            # Get the velocity error to the reference simulation for this convergence type and geometry
            err: NumpyFloatArray = vel_error(
                cnvg_type=cnvg_type, G=G, nz_ref=nz_ref, nzs=nzs, stage=stage, level=level, use_fine=use_fine)
            # Append the nz values and errors to the lists
            nzs3.append(nzs)
            errs3.append(err)
            # Report the velocity error if requested
            if verbose:
                print(f'Errors for geometry {G:d}:')
                for i, nz in enumerate(nzs):
                    diff: float = err[i]
                    print(f'{nz:2d}: {diff:5.2e}')

    # Plot the velocity errors for all three geometries on a single plot
    plot_vel_errors(nzs3=nzs3, errs3=errs3, nz_ref=nz_ref, level=level, fig_style=fig_style)

# *************************************************************************************************
if __name__ == '__main__':
    main()
