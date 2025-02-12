import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from pathlib import Path

# Local imports
from make_inputs import subdir_tbl
from cnvg_utils import SimOutput, geom_name_tbl, plot_dir, fit_error_mesh

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]

# *************************************************************************************************
# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# *************************************************************************************************
def load_sim(G: int, nz: int, stage: int, level: int) -> SimOutput:
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
    dir_top: str = '05_cnvg_react_mesh'
    # The name of the subdirectory for this geometry
    sub_dir: str = subdir_tbl[G]
    # The directory containing the simulation numpy output for this stage
    dir_sim: Path = Path(f'{dir_top}/{sub_dir}/n{nz:03d}/stage{stage:02d}/numpy_last')
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
    soc: NumpyFloatArray = np.load(dir_lev / 'soc.npy')
    # Wrap the numpy arrays in a SimOutput object
    sim_out: SimOutput = SimOutput(grid_shape=grid_shape, refinement=refinement, cell_type=cell_type, 
                                   velocity=velocity, pressure=pressure, soc=soc)
    return sim_out

# *************************************************************************************************
def soc_diff_rms(s_ref: SimOutput, s: SimOutput) -> float:
    """
    Compute the root mean square difference in state of charge fields between two simulations.
    Measure the difference on the fine grid.
    INPUTS:
    s_ref: Reference simulation
    s: Comparison simulation
    RETURNS:
    rms: RMS difference of the difference in state of charge fields
    """
    # SOC on the fine grid
    s2 = np.nan_to_num(s_ref.soc, nan=0.0)
    # refinement ratio
    rr: int = s_ref.grid_shape[0] // s.grid_shape[0]
    # SOC on the coarse grid; convert NaNs to zeros
    s1_coarse = np.nan_to_num(s.soc, nan=0.0)
    # Matrix to upsample the coarse grid to the fine grid in the Kronecker delta
    up_mat: NumpyFloatArray = np.ones((rr, rr, rr))
    # Upsample s1 to the fine grid
    s1 = np.kron(s1_coarse, up_mat)
    # difference in velocity fields
    s_diff = np.nan_to_num(s1 - s2, nan=0.0)
    # RMS of the difference
    rms: float = np.sqrt(np.mean(np.square(s_diff)))
    return rms

# *************************************************************************************************
def soc_error(G: int, nz_ref: int, nzs_comp: list[int]) \
    -> tuple[NumpyInt16Array, NumpyInt16Array, NumpyFloatArray]:
    """
    Compute the root mean square of the difference SOC between a collection of simulations and a reference.
    INPUTS:
    G:          Number keying the geometry
    nz_ref:     Number of grid points in the z-direction for the reference simulation
    nzs_comp:   List of grid points in the z-direction for the comparison simulations
    stage:      Which stage to load; stage n has n-1 levels of refinement
    RETURNS:
    nz:         Array with the number of grid points in the z-direction for the comparison simulation
    level:      Array with the mesh refinement level for the comparison simulation
    err:        Array of RMS differences in SOC
    """
    # Initialize empty list of nz
    nz_l: list[int] = []
    # Initialize empty list of levels
    level_l: list[int] = []
    # Initialize empty list of errors
    err_l: list[float] = []

    # Load the reference simulation; this is always stage01 (level 0) for now
    stage_ref: int = 1
    level_ref: int = 0
    s_ref: SimOutput = load_sim(G=G, nz=nz_ref, stage=stage_ref, level=level_ref)

    # Iterate over simulations to compare
    nz_i: int
    for nz_i in nzs_comp:
        # The expected number of mesh refinement levels
        level_max: int = int(np.log2(nz_ref / nz_i))
        # Iterate over mesh refinement levels
        for level_i in range(level_max+1):
            # The stage number for this level
            stage_i: int = level_i + 1
            # Load the comparison simulation
            try:
                s: SimOutput = load_sim(G=G, nz=nz_i, stage=stage_i, level=level_i)
            except FileNotFoundError:
                # Skip this simulation if it does not exist
                # msg = f'File not found: G={G:d}, n{nz_i:03d}, stage{stage_i:02d}, L{level_i:0d}'
                # warnings.warn(msg)
                continue
            # Compute the L2 norm of the difference in relative velocity fields
            err_i: float = soc_diff_rms(s_ref=s_ref, s=s)
            # Append the results to the lists
            nz_l.append(nz_i)
            level_l.append(level_i)
            err_l.append(err_i)
    # Convert to Numpy arrays
    nz: NumpyInt16Array = np.array(nz_l, dtype=np.int16)
    level: NumpyInt16Array = np.array(level_l, dtype=np.int16)
    err: NumpyFloatArray = np.array(err_l, dtype=np.float64)    
    # Return the arrays
    return (nz, level, err)

# *************************************************************************************************
def plot_soc_error(nz: NumpyInt16Array, level: NumpyInt16Array, err: NumpyFloatArray, 
                   nz_ref: int, G: int, verbose: bool) -> None:
    """
    Plot the RMS of the difference in SOC between two simulations.
    INPUTS:
    nz:         Array with the number of grid points in the z-direction for the comparison simulation
    level:      Array with the mesh refinement level for the comparison simulation
    err:        Array of L2 norms of the difference in velocity fields
    nz_ref:     Number of grid points in the z-direction for the reference simulation
    G:          Number keying the geometry
    verbose:    Print the errors to the console
    """
    # Name of this geometry
    geom_name: str = geom_name_tbl[G]
    # Number of rows in the table of nz values
    i_max: int = nz.shape[0]
    
    # Report the velocity error if requested
    if verbose:
        print(f'Errors for reaction convergence with mesh refinement and geometry {G:d}:')
        print(f"{'nz':6s} : {'level':6s} : {'err':8s}")
        for i in range(i_max):
            nz_i: int = nz[i]
            level_i: int = level[i]
            err_i: float = err[i]
            print(f'{nz_i:6d} : {level_i:6d} : {err_i:8.2e}')

    # Invert the grid points for the x-axis; h = 1/nz
    h: NumpyFloatArray = 1.0 / nz
    h_ref: float = 1.0 / nz_ref

    # Range of h values
    h_min: float = np.min(h).astype(float)
    h_max: float = np.max(h).astype(float)
    # Corresponding powers of 2
    log2h_min: float = np.log2(h_min)
    log2h_max: float = np.log2(h_max)
    # Range of error values
    err_min: float = np.min(err).astype(float)
    err_max: float = np.max(err).astype(float)

    # Sample values of h for fitting - coarse
    hh_c_L0 = np.logspace(log2h_min+0, log2h_max, 129, base=2.0)
    hh_c_L1 = np.logspace(log2h_min+0, log2h_max, 129, base=2.0)
    hh_c_L2 = np.logspace(log2h_min+1, log2h_max, 129, base=2.0)
    hh_c_L3 = np.logspace(log2h_min+2, log2h_max, 129, base=2.0)
    # Special handling for the finest level
    dx = 1.0 / 16.0
    hh_c_L4 = np.logspace(log2h_min+3-dx, log2h_max+dx, 17, base=2.0)

    # Sample values of h for fitting - fine at various refinement levels
    hh_f_L0 = hh_c_L0 / 1.0
    hh_f_L1 = hh_c_L1 / 2.0
    hh_f_L2 = hh_c_L2 / 4.0
    hh_f_L3 = hh_c_L3 / 8.0
    hh_f_L4 = hh_c_L4 / 16.0

    # Refinement ratio rr = h_ref / h_c on the coarse grid
    rr_c_L0 = h_ref / hh_c_L0
    rr_c_L1 = h_ref / hh_c_L1
    rr_c_L2 = h_ref / hh_c_L2
    rr_c_L3 = h_ref / hh_c_L3
    rr_c_L4 = h_ref / hh_c_L4

    # Refinement ratio rr = h_ref / h_f on the fine grid
    # rr_f_L0 = h_ref / hh_f_L0
    # rr_f_L1 = h_ref / hh_f_L1
    # rr_f_L2 = h_ref / hh_f_L2
    # rr_f_L3 = h_ref / hh_f_L3
    # rr_f_L4 = h_ref / hh_f_L4
    # DEBUG
    rr_f_L0 = rr_c_L0
    rr_f_L1 = rr_c_L1
    rr_f_L2 = rr_c_L2
    rr_f_L3 = rr_c_L3
    rr_f_L4 = rr_c_L4

    # Masks for each level of refinement
    mask_L0: NumpyInt16Array = (level == 0)
    mask_L1: NumpyInt16Array = (level == 1)
    mask_L2: NumpyInt16Array = (level == 2)
    mask_L3: NumpyInt16Array = (level == 3)
    mask_L4: NumpyInt16Array = (level == 4)

    # Calculate the nonlinear fit
    C_c: float
    C_f: float
    p: float
    alpha: float
    rmse: float
    C_c, C_f, p, alpha, rmse = fit_error_mesh(h=h, level=level, err=err, h_ref=h_ref)

    # Report the fit parameters
    if verbose:
        print('Fit: err(h_c, h_f) =\n\t'
              'C_c * h_c^p + C_f * h_f^p * (1 - alpha (h_ref/h_c)^p).')
        print(f'C_c = {C_c:6.2f}, C_f = {C_f:6.2f}.')
        print(f'p   = {p:0.3f}, alpha = {alpha:0.3f}.')
        print(f'RMSE = {rmse:0.4f}.')

    # Calculate the error for the fit at each level of refinement
    err_fit_L0 = \
        C_c * np.power(hh_c_L0, p) * (1.0 - alpha * np.power(rr_c_L0, p)) + \
        C_f * np.power(hh_f_L0, p) * (1.0 - alpha * np.power(rr_f_L0, p))
    err_fit_L1 = \
        C_c * np.power(hh_c_L1, p) * (1.0 - alpha * np.power(rr_c_L1, p)) + \
        C_f * np.power(hh_f_L1, p) * (1.0 - alpha * np.power(rr_f_L1, p))
    err_fit_L2 = \
        C_c * np.power(hh_c_L2, p) * (1.0 - alpha * np.power(rr_c_L2, p)) + \
        C_f * np.power(hh_f_L2, p) * (1.0 - alpha * np.power(rr_f_L2, p))
    err_fit_L3 = \
        C_c * np.power(hh_c_L3, p) * (1.0 - alpha * np.power(rr_c_L3, p)) + \
        C_f * np.power(hh_f_L3, p) * (1.0 - alpha * np.power(rr_f_L3, p))
    err_fit_L4 = \
        C_c * np.power(hh_c_L4, p) * (1.0 - alpha * np.power(rr_c_L4, p)) + \
        C_f * np.power(hh_f_L4, p) * (1.0 - alpha * np.power(rr_f_L4, p))

    # Parameter values and tex string for series label
    tex: str = \
        r'$\varepsilon = \left\lbrace C_c {h_c}^p + C_f {h_f}^p \right\rbrace ' + \
        r'\left\lbrace 1 - \alpha (h_{ref} / h_c)^p \right\rbrace$'
    params: str = f'$C_c={C_c:0.2f}, C_f={C_f:0.2f}, p={p:0.3f}, \\alpha={alpha:0.3f}$'
    err_str: str = f'RMSE={rmse:0.4f}'
    label: str = f'{tex:s}\n{params:s}\n{err_str:s}'

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data for each level of refinement
    line_L0 = ax.plot(h[mask_L0], err[mask_L0], marker='o', linestyle='None', label='L0')
    line_L1 = ax.plot(h[mask_L1], err[mask_L1], marker='o', linestyle='None', label='L1')
    line_L2 = ax.plot(h[mask_L2], err[mask_L2], marker='o', linestyle='None', label='L2')
    line_L3 = ax.plot(h[mask_L3], err[mask_L3], marker='o', linestyle='None', label='L3')
    line_L4 = ax.plot(h[mask_L4], err[mask_L4], marker='o', linestyle='None', label='L4')

    # Plot the fit for each level of refinement
    ax.plot(hh_c_L0, err_fit_L0, linestyle='--', linewidth=0.5, label=None, color=line_L0[0].get_color())
    ax.plot(hh_c_L1, err_fit_L1, linestyle='--', linewidth=0.5, label=None, color=line_L1[0].get_color())
    ax.plot(hh_c_L2, err_fit_L2, linestyle='--', linewidth=0.5, label=None, color=line_L2[0].get_color())
    ax.plot(hh_c_L3, err_fit_L3, linestyle='--', linewidth=0.5, label=None, color=line_L3[0].get_color())
    ax.plot(hh_c_L4, err_fit_L4, linestyle='--', linewidth=0.5, label=None, color=line_L4[0].get_color())

    # Set log scale with base 2 and add legend
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.legend()
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='-', linewidth=0.5)
    # Combined range of errors including both data and fits
    err_min_combined = min(err_min, np.min(np.concatenate([err_fit_L0, err_fit_L1, err_fit_L2, err_fit_L3, err_fit_L4])))
    err_max_combined = max(err_max, np.max(np.concatenate([err_fit_L0, err_fit_L1, err_fit_L2, err_fit_L3, err_fit_L4])))
    # Set limits on the y axis
    y_min: float = np.power(2.0, np.floor(np.log2(err_min_combined)))
    y_max: float = np.power(2.0, np.ceil(np.log2(err_max_combined)))
    # DEBUG
    # print(f'Min error by level:')
    # print(f'Data   : {err_min:0.2e}')
    # print(f'Fit L0 : {np.min(err_fit_L0):0.2e}')
    # print(f'Fit L1 : {np.min(err_fit_L1):0.2e}')
    # print(f'Fit L2 : {np.min(err_fit_L2):0.2e}')
    # print(f'Fit L3 : {np.min(err_fit_L3):0.2e}')
    # print(f'Fit L4 : {np.min(err_fit_L4):0.2e}')
    ax.set_ylim(y_min, y_max)

    # Set title and axis labels
    ax.set_title(f'Reaction Convergence with Refinement - {geom_name:s}', fontsize=20)
    ax.set_xlabel('Relative Step Size $h$ on Coarse Grid (dimensionless)', fontsize=14)
    ax.set_ylabel('Root Mean Square of SOC Difference ' + 
                  r'$\left\lbrace \frac{1}{N} \cdot \sum \left(s - s_{\tiny\textrm{r}}\right)^2 \right\rbrace^{1/2}$', 
                  fontsize=14)

    # Textbox with the fit parameters
    bbox: dict = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}
    text_x: float = np.power(2.0, -5.5)
    text_y: float = y_max * np.power(y_min / y_max, 0.05)
    ax.text(text_x, text_y, label, fontsize=10, 
            horizontalalignment='left', verticalalignment='top', bbox=bbox)

    # Number of this plot
    plot_num: int = 9 + G
    # Save the figure
    fname: str = (plot_dir / f'{plot_num:02d}_react_cnvg_mesh.png').as_posix()
    fig.savefig(fname)

# *************************************************************************************************
def main():
    """Generate plots of the SOC error for the reaction convergence study"""

    # Number of grid points in the z-direction for the reference simulation
    nz_ref: int = 128
    # Number of grid points in the z-direction for the comparison simulations
    nzs_comp: list[int] = [8, 16, 32, 64]
    # Geometries to process
    Gs: list[int] = [1, 2, 3]
    # Gs: list[int] = [3]
    # Verbosity flag
    verbose: bool = True
    
    # Iterate over geometries
    G: int
    for G in Gs:
        # Get the velocity error to the reference simulation for this convergence type and geometry
        nz: NumpyInt16Array
        level: NumpyInt16Array
        err: NumpyFloatArray  
        nz, level, err = soc_error(G=G, nz_ref=nz_ref, nzs_comp=nzs_comp)
        # Plot the SOC error
        plot_soc_error(nz=nz, err=err, nz_ref=nz_ref, G=G, level=level, verbose=verbose)

# *************************************************************************************************
if __name__ == '__main__':
    main()
