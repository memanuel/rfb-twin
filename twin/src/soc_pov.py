import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Local imports
from utils import Geometry, calc_geometry, geometry_add_types
from streamline_pov import include_streamline

# *************************************************************************************************
# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# Path to the numpy simulation data
path_sim: Path = Path('07_react_special/P010/Vp000/n128/stage01/numpy_last')

# Path to load integrated streamlines
path_stream: Path = Path('npy/11_streamline')

# Path to save the pov files
path_out: Path = Path('povray')

# Colormap for the state of charge
cmap = mpl.colormaps['jet']

# *************************************************************************************************
def load_geometry():
    # Load the lower coordinates of the box and convert to microns
    lo_levs: NumpyFloatArray = np.load(path_sim / 'domain_lo.npy')
    # Load the upper coordinates of the box
    hi_levs: NumpyFloatArray = np.load(path_sim / 'domain_hi.npy')
    # Load the grid shape on each level
    grid_shapes: NumpyInt16Array = np.load(path_sim / 'grid_shape.npy')
    # Extract just the top level and convert from cm to microns
    cm2um: float = 1.0E4
    lo: NumpyFloatArray = lo_levs[0] * cm2um
    hi: NumpyFloatArray = hi_levs[0] * cm2um
    # Shape at the top level
    grid_shape = grid_shapes[0]

    # Load the cell type and refinement
    path_lev: Path = path_sim / 'L0'
    cell_type: NumpyInt16Array = np.load(path_lev / 'cell_type.npy')
    refinement: NumpyInt16Array = np.load(path_lev / 'refinement.npy')

    # Calculate geometry
    geom: Geometry = calc_geometry(lo=lo, hi=hi, shape=grid_shape)
    geometry_add_types(geom=geom, cell_type=cell_type, refinement=refinement)
    return geom

# *************************************************************************************************
def load_soc() -> NumpyFloatArray:
    """Load state of charge field from file"""
    # Load the SOC field
    soc = np.load(path_sim / 'L0' / 'soc.npy')
    return soc

# *************************************************************************************************
def load_overpot() -> NumpyFloatArray:
    """Load overpotential from file"""
    overpot = np.load(path_sim / 'L0' / 'overpot.npy')
    return overpot

# *************************************************************************************************
def load_curr_dens() -> NumpyFloatArray:
    """Load current density"""
    current = np.load(path_sim / 'L0' / 'current.npy')
    area = np.load(path_sim / 'L0' / 'area.npy')
    curr_dens = current / area
    return curr_dens

# *************************************************************************************************
def make_soc_interpolator(soc: NumpyFloatArray, geom: Geometry) \
        -> RegularGridInterpolator:
    """Create interpolator for state of charge"""
    # Wrap the cell centers into a tuple as expected by RegularGridInterpolator
    points: tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray] = (geom.xc, geom.yc, geom.zc)

    # Select the interpolation method
    method: str = 'linear'
    # Don't raise an error when trying to interpolate outside of the domain
    bounds_error: bool = False

    # Fill value for the soc
    fill_value: float = np.nan

    # Create the interpolator for the state of charge
    soc_func: RegularGridInterpolator = \
        RegularGridInterpolator(points, soc, method=method, bounds_error=bounds_error, fill_value=fill_value)

    return soc_func

# *************************************************************************************************
def write_soc_grid(soc: NumpyFloatArray, geom: Geometry, stride: int) -> None:
    """
    Write out a POV-ray file with the spheres for the SOC on a regular grid.
    INPUTS:
        soc - array with the state of charge
        geom - Geometry object with the geometry information
        stride - spacing in index space; write out every s-th along each dimension
    """
    # Status message
    print(f'write_soc_grid: stride = {stride}')
    # Get min and max values of the SOC
    vmin: float = 0.0
    smax: float = float(np.nanmax(soc))
    vmax: float = float(np.ceil(smax*20)/20)
    # Report vmax for scalebar
    print(f'SOC vmax = {vmax:0.6f}.')

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # The current sphere as a string
    pov: str

    # Unpack the shape
    # Iterator for indices in x, y, z
    ii = list(range(stride//2, geom.nx, stride))
    jj = list(range(stride//2, geom.ny, stride))
    kk = list(range(stride//2, geom.nz, stride))

    # Name of the output file
    fname_pov: str = f"soc_grid_obj_s{stride:02d}.pov"

    # Write the streamlines to the file
    with open(path_out / fname_pov, "w") as fh:
        # Iterate over the points
        for i in tqdm(ii):
            x = geom.xc[i]
            for j in jj:
                y = geom.yc[j]
                for k in kk:
                    # Skip this point if it's solid
                    if geom.is_solid[i,j,k]:
                        continue
                    # Get the coordinates and SOC value
                    z = geom.zc[k]
                    s: float = soc[i, j, k]
                    # Calculate the color from the SOC
                    r, g, b = rgb_func(s)
                    # Build the POV string and write this line to the file
                    center: str = f"<{x:0.6f}, {y:0.6f}, {z:0.6f}>"
                    pigment: str = f"pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}> }}"
                    finish: str = "finish {reflection 0.0 specular 0.0 emission 1.0}"
                    texture: str = f"texture {{ {pigment:s} {finish:s} }}"
                    # rs is a symbolic radius that is set in the POV-ray rendering file
                    pov = f"sphere {{ {center:s}, rs {texture:s} }}\n"
                    fh.write(pov)

# *************************************************************************************************
def write_overpot(overpot: NumpyFloatArray, geom: Geometry) -> None:
    """
    Write out a POV-ray file with the overpotential on the cut cells.
    INPUTS:
        overpot - array with the state of overpot
        geom - Geometry object with the geometry information
    """
    # Status message
    print(f'write_overpot:')
    # Get min and max values of the overpotential
    vmin: float = 0.0
    smax: float = float(np.nanmax(overpot))
    vmax: float = float(np.ceil(smax*20)/20)
    # Report vmax for scalebar
    print(f'Overpotential vmax = {vmax:0.6f} mV.')

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # The current sphere as a string
    pov: str

    # Unpack the shape
    # Iterator for indices in x, y, z
    ii = list(range(geom.nx))
    jj = list(range(geom.ny))
    kk = list(range(geom.nz))

    # Name of the output file
    fname_pov: str = f"overpot_obj.pov"

    # Write the streamlines to the file
    with open(path_out / fname_pov, "w") as fh:
        # Iterate over the points
        for i in tqdm(ii):
            x = geom.xc[i]
            for j in jj:
                y = geom.yc[j]
                for k in kk:
                    # Get the coordinates and overpot value
                    z = geom.zc[k]
                    s: float = overpot[i, j, k]
                    # Skip if the SOC is not a number
                    if np.isnan(s):
                        continue
                    # Calculate the color from the overpotential
                    r, g, b = rgb_func(s)
                    # Build the POV string and write this line to the file
                    center: str = f"<{x:0.6f}, {y:0.6f}, {z:0.6f}>"
                    pigment: str = f"pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}> }}"
                    finish: str = "finish {reflection 0.0 specular 0.0 emission 1.0}"
                    texture: str = f"texture {{ {pigment:s} {finish:s} }}"
                    # rs is a symbolic radius that is set in the POV-ray rendering file
                    pov = f"sphere {{ {center:s}, rs {texture:s} }}\n"
                    fh.write(pov)

# *************************************************************************************************
def write_curr_dens(curr_dens: NumpyFloatArray, geom: Geometry) -> None:
    """
    Write out a POV-ray file with the current density on the cut cells.
    INPUTS:
        curr_dens - array with the current density
        geom - Geometry object with the geometry information
    """
    # Status message
    print(f'write_curr_dens:')
    # Get min and max values of the overpotential
    vmin: float = 0.0
    smax: float = float(np.nanmax(curr_dens))
    vmax: float = float(np.ceil(smax*20)/20)
    # Report vmax for scalebar
    print(f'Current density = {vmax:0.6f} mA / cm^2.')

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # The current sphere as a string
    pov: str

    # Unpack the shape
    # Iterator for indices in x, y, z
    ii = list(range(geom.nx))
    jj = list(range(geom.ny))
    kk = list(range(geom.nz))

    # Name of the output file
    fname_pov: str = f"curr_dens_obj.pov"

    # Write the streamlines to the file
    with open(path_out / fname_pov, "w") as fh:
        # Iterate over the points
        for i in tqdm(ii):
            x = geom.xc[i]
            for j in jj:
                y = geom.yc[j]
                for k in kk:
                    # Get the coordinates and overpot value
                    z = geom.zc[k]
                    s: float = curr_dens[i, j, k]
                    # Skip if the current density is not a number
                    if np.isnan(s):
                        continue
                    # Calculate the color from the overpotential
                    r, g, b = rgb_func(s)
                    # Build the POV string and write this line to the file
                    center: str = f"<{x:0.6f}, {y:0.6f}, {z:0.6f}>"
                    pigment: str = f"pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}> }}"
                    finish: str = "finish {reflection 0.0 specular 0.0 emission 1.0}"
                    texture: str = f"texture {{ {pigment:s} {finish:s} }}"
                    # rs is a symbolic radius that is set in the POV-ray rendering file
                    pov = f"sphere {{ {center:s}, rs {texture:s} }}\n"
                    fh.write(pov)

# *************************************************************************************************
def line2str(X: NumpyFloatArray, K: NumpyInt16Array, soc_func: RegularGridInterpolator,
             i: int, s: int, rgb_func) -> str:
    """
    Write out k/s spheres and (k-1)/s cylinders along the i-th streamline 
    and return a string with the povray code.
    INPUTS:
        X - array with the position along the streamlines
        K - array with the number of points in each streamline
        soc_func - interpolator for the state of charge
        i - index of the streamline
        s - spacing in index space; cylinders connect points s apart along the streamline
        rgb_func - function to calculate the color of an object based on the SOC
    """
    # The number of points on this streamline; sample every s points
    k: np.int16 = K[i] // s
    # The streamline as a numpy array; sample every s points on the input X array
    x: NumpyFloatArray = X[i][::s]
    # The state of charge along the sampled streamline
    soc: NumpyFloatArray = soc_func(x)

    # The POV-Ray string fragment
    pov: str = ""
    # Finish string shared by all spheres and cylinders
    finish : str = "finish {reflection 0.0 specular 0.0 emission 1.0}"

    # Iterate over the points in the streamline and write out spheres at each point
    for j in range(k):
        # Calculate the color of the sphere based on the SOC at this point
        r: float
        g: float
        b: float
        r, g, b = rgb_func(soc[j])
        # Write out a sphere at this point
        q: NumpyFloatArray = x[j]
        center: str = f"<{q[0]:0.6f}, {q[1]:0.6f}, {q[2]:0.6f}>"
        texture: str = f"texture {{pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}>}} {finish:s} }}"
        pov += f"sphere {{ {center:s}, rs {texture:s} }}\n"

    # Iterate over the points in the streamline and write out cylinders connecting them
    for j in range(k-1):
        # The starting point of this segment as a numpy array and POV-Ray string fragment
        q0: NumpyFloatArray = x[j]
        p0: str = f"<{q0[0]:0.6f}, {q0[1]:0.6f}, {q0[2]:0.6f}>"
        # The ending point of this segment as a numpy array and POV-Ray string fragment
        q1: NumpyFloatArray = x[j+1]
        p1: str = f"<{q1[0]:0.6f}, {q1[1]:0.6f}, {q1[2]:0.6f}>"
        # The average SOC along the segment
        soc_mid: float = (soc[j] + soc[j+1]) / 2.0
        # Calculate the color of the cylinder based on the average SOC
        r: float
        g: float
        b: float
        r, g, b = rgb_func(soc_mid)
        # Write out the cylinder
        texture: str = f"texture {{pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}>}} {finish:s} }}"
        pov += f"cylinder {{ {p0:s}, {p1:s}, rs {texture:s} }}\n"

    return pov

# *************************************************************************************************
def write_soc_stream(X: NumpyFloatArray, K: NumpyInt16Array, soc_func: RegularGridInterpolator,
                   s_origin: int, s_line: int) -> None:
    """
    Write out the file with the spheres and cylinders along the streamlines
    INPUTS:
        X - array with the position along the streamlines
        V - array with the velocity along the streamlines
        T - array with the time points along the streamlines
        K - array with the number of points in each streamline
        s_origin - stride for sampling the streamlines in the YZ plane
        s_line - stride for sampling points along each streamline
    """
    # Status message
    print(f'write_soc_grid: s_origin = {s_origin}, s_line={s_line}')

    # Number of streamlines
    m_line: int = X.shape[0]

    # Get min and max values of the SOC
    vmin: float = 0.0
    smax: float = float(np.nanmax(soc_func(X)))
    vmax: float = float(np.ceil(smax*20)/20)

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)

    # Name of the output file
    fname_pov: str = f"soc_stream_obj_s{s_origin:02d}.pov"
    # Write the streamlines to the file
    with open(path_out / fname_pov, "w") as fh:
        # Iterate over the streamlines
        for i in tqdm(range(m_line)):
            # Skip empty streamlines
            if K[i] == 0:
                continue
            # Starting point of this streamline
            y: float = X[i, 0, 1]
            z: float = X[i, 0, 2]
            # Should we write the streamline? Pick streamline origins with a stride of s_origin
            if include_streamline(y=y, z=z, stride=s_origin):
                fh.write(line2str(X=X, K=K, soc_func=soc_func, i=i, s=s_line, rgb_func=rgb_func))

# *************************************************************************************************
def summarize_field(phi: NumpyFloatArray, name: str) -> None:
    """Print short summary of a scalar field phi"""
    # Calculate Summary statistics
    phi_min: float = np.nanmin(phi)
    phi_max: float = np.nanmax(phi)
    phi_mean: float = np.nanmean(phi)
    phi_std: float = np.nanstd(phi)
    # Report the summary
    print(f'Summary statistics: {name}')
    print(f'Min : {phi_min:0.6f}')
    print(f'Max : {phi_max:0.6f}')
    print(f'Mean: {phi_mean:0.6f}')
    print(f'Std : {phi_std:0.6f}')

# *************************************************************************************************
def main(build_soc: bool, build_overpot: bool, build_curr_dens: bool) -> None:
    """Build the POV-ray files for the state of charge, overpotential, and current density."""
    # Load the geometry
    geom: Geometry = load_geometry()
    # Load the state of charge
    soc: NumpyFloatArray = load_soc()
    # Load the overpotential
    overpot: NumpyFloatArray = load_overpot()
    # Load the current density
    curr_dens: NumpyFloatArray = load_curr_dens()

    # Summary statistics for each field
    summarize_field(phi=soc, name='SOC')
    summarize_field(phi=overpot, name='Overpotential')
    summarize_field(phi=curr_dens, name='Current Density')

    # Quit early if not building any POV-ray object files
    if not (build_soc or build_overpot or build_curr_dens):
        return

    # Load the numpy arrays with the resampled streamlines
    X: NumpyFloatArray = np.load(path_stream / "Xr.npy")
    K: NumpyInt16Array = np.load(path_stream / "Kr.npy")

    # Build the SOC interpolator
    soc_func: RegularGridInterpolator = make_soc_interpolator(soc=soc, geom=geom)

    # Write the pov file for SOC sampled on a grid
    # stride: int
    # for stride in [16, 8, 4, 2, 1]:
    #     write_soc_grid(soc=soc, geom=geom, stride=stride)

    # Write the pov file for SOC sampled along the streamlines
    if build_soc:
        s_line: int = 1
        for s_origin in [16, 8, 4, 2, 1]:
            write_soc_stream(X=X, K=K, soc_func=soc_func, s_origin=s_origin, s_line=s_line)

    # Write the overpotential
    if build_overpot:
        write_overpot(overpot=overpot, geom=geom)

    # Write the current density
    if build_curr_dens:
        write_curr_dens(curr_dens=curr_dens, geom=geom)

# *************************************************************************************************
if __name__ == "__main__":
    # Actions to take
    build_soc: bool = False
    build_overpot: bool = False
    build_curr_dens: bool = False
    # Delegate the plotting to main()
    main(build_soc=build_soc, build_overpot=build_overpot, build_curr_dens=build_curr_dens)
