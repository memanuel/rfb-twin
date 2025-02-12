import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from pathlib import Path
from tqdm import tqdm

# Local imports
from utils import Geometry, calc_geometry

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# *************************************************************************************************
# Path to the numpy simulation data
path_sim: Path = Path('06_flow/P100/n128/stage01/numpy_last')

# Path to save output numpy files
path_out: Path = Path('npy/11_streamline')

# *************************************************************************************************
def load_domain() -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """Load the domain from file"""
    # Load the lower coordinates of the box and convert to microns
    lo_levs: NumpyFloatArray = np.load(path_sim / 'domain_lo.npy')
    # Load the upper coordinates of the box
    hi_levs: NumpyFloatArray = np.load(path_sim / 'domain_hi.npy')
    # Extract just the top level and convert from cm to microns
    cm2um: float = 1.0E4
    lo: NumpyFloatArray = lo_levs[0] * cm2um
    hi: NumpyFloatArray = hi_levs[0] * cm2um
    return lo, hi

# *************************************************************************************************
def load_vel() -> NumpyFloatArray:
    """Load velocity field from file"""
    # Load the velocity field
    vel = np.load(path_sim / 'L0' / 'velocity.npy')
    # Replace any NaN values with zeros
    vel = np.nan_to_num(vel, nan=0.0)
    return vel

# *************************************************************************************************
def make_interpolators(vel: NumpyFloatArray, geom: Geometry, vel_min: float) \
        -> tuple[RegularGridInterpolator, RegularGridInterpolator, RegularGridInterpolator]:
    """Create interpolators for the velocity field and unit vectors"""
    # Wrap the cell centers into a tuple as expected by RegularGridInterpolator
    points: tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray] = (geom.xc, geom.yc, geom.zc)

    # Select the interpolation method
    method: str = 'linear'
    # Don't raise an error when trying to interpolate outside of the domain
    bounds_error: bool = False

    # The speed of the velocity field
    spd: NumpyFloatArray = np.sqrt(np.sum(np.square(vel), axis=-1))
    # Mask where speed is too small
    mask_slow: NumpyBoolArray = (spd < vel_min)

    # Global maximum speed for scalebar
    spd_max: float = np.max(spd).astype(float)
    # Report this to the console - need it for scalebar
    print(f'Global max speed: {spd_max:0.6f}')

    # Set fill values for numerical stability to handle two cases:
    # (1) When the speed is too small,
    # (2) When out of bounds.
    # In both cases, we assume the velocity is (vel_min, 0, 0), the direction is (1, 0, 0), and the speed is vel_min.

    # Fill value for the velocity field
    fill_value_vel: NumpyFloatArray = np.array([vel_min, 0.0, 0.0])
    # Fill value for the direction
    fill_value_dir: NumpyFloatArray = np.array([1.0, 0.0, 0.0])
    # Fill value for the speed
    fill_value_spd: float = vel_min

    # Adjust the speed where it's too slow
    spd[mask_slow] = fill_value_spd
    # Adjust the the velocity field where the speed is too small
    vel[mask_slow] = fill_value_vel    
    # Normalize the velocity field to get the unit vectors; call this dir because it's a direction of travel
    # Note we don't need special handling for slow speeds here because the speed and velocity are already set
    dir: NumpyFloatArray = vel / spd.reshape(geom.nx, geom.ny, geom.nz, 1)

    # Create the interpolator for the velocity field
    vel_func: RegularGridInterpolator = \
        RegularGridInterpolator(points, vel, method=method, bounds_error=bounds_error, fill_value=fill_value_vel)

    # Create the interpolator for the unit vector
    dir_func: RegularGridInterpolator = \
        RegularGridInterpolator(points, dir, method=method, bounds_error=bounds_error, fill_value=fill_value_dir)

    # Create the interpolator for the speed
    spd_func: RegularGridInterpolator = \
        RegularGridInterpolator(points, spd, method=method, bounds_error=bounds_error, fill_value=fill_value_spd)

    return vel_func, dir_func, spd_func

# *************************************************************************************************
def integrate_streamlines(
        vel_func: RegularGridInterpolator, dir_func: RegularGridInterpolator, spd_func: RegularGridInterpolator,
        geom: Geometry, x0: NumpyFloatArray, ds: float, n_max: int) \
    -> tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray, NumpyFloatArray, NumpyInt16Array]:
    """
    Integrate one streamline starting at x0 with time step h
    INPUTS:
        vel_func: RegularGridInterpolator - interpolator for the velocity field
        dir_func: RegularGridInterpolator - interpolator for the unit vectors (direction)
        spd_func: RegularGridInterpolator - interpolator for the speed
        geom: Geometry - geometry of the simulation
        x0: NumpyFloatArray - starting position of the streamline; array of shape (n_pts, 3)
        ds: float - distance step size in microns
        n_max: int - maximum number of points on one streamline
    RETURNS:
        X: NumpyFloatArray - positions of the streamline; array of shape (n_pts, n_max, 3)
        V: NumpyFloatArray - velocities of the streamline; array of shape (n_pts, n_max, 3)
        L: NumpyFloatArray - arc length of each streamline; array of shape (n_pts, n_max)
        T: NumpyInt16Array - time spent on each streamline; array of shape (n_pts)
        K: NumpyInt16Array - number of points on each streamline; array of shape (n_pts)
    """
    # The number of streamlines to integrate
    m_line: int = x0.shape[0]
    # Lower bound on interpolation values; bottom left CELL CENTER of the domain
    lo: NumpyFloatArray = geom.center[0, 0, 0, :]
    # Upper bound on interpolation values; top right CELL CENTER of the domain
    hi: NumpyFloatArray = geom.center[-1, -1, -1, :]

    # The output positions along the streamline; shape (m_line, n_max, 3)
    X: NumpyFloatArray = np.zeros((m_line, n_max, 3), dtype=np.float64)
    # The output velocities along the streamline; shape (m_line, n_max, 3)
    V: NumpyFloatArray = np.zeros((m_line, n_max, 3), dtype=np.float64)
    # The time spent on each streamline; shape (m_line, n_max)
    T: NumpyFloatArray = np.zeros((m_line, n_max), dtype=np.float64)
    # The arc length of each streamline; shape (m_line, n_max)
    L: NumpyFloatArray = np.zeros((m_line, n_max), dtype=np.float64)
    # The number of points on each streamline; shape (m_line, )
    K: NumpyInt16Array = np.zeros(m_line, dtype=np.int16)

    # Constant prefactor used to evaluate wk in Ralston's Method
    cw: float = 2.0 / 3.0 * ds
    # Constant prefactor used to evaluate xk in Ralston's Method
    cx: float = 1.0 / 4.0 * ds
    # Constant coefficients for contributions to dt
    ct1: float = 0.25 * ds
    ct2: float = 0.75 * ds

    # The position at  the start of the time step; shape (m_line, 3)
    xk: NumpyFloatArray = x0
    # The direction at the start of the time step; shape (m_line, 3)
    uk: NumpyFloatArray = dir_func(xk)
    # The position at the intermediate point 2/3 of the way on the time step; shape (m_line, 3)
    xki: NumpyFloatArray = x0 + cw * uk
    # The direction at the intermediate point 2/3 of the way on the time step; shape (m_line, 3)
    uki: NumpyFloatArray = dir_func(xki)
    # The velocity at the start of the time step; shape (m_line, 3)
    vk: NumpyFloatArray = vel_func(xk)
    # The speed at the start of the time step; shape (m_line)
    sk: NumpyFloatArray = spd_func(xk)
    # The speed at the intermediate point 2/3 of the way on the time step; shape (m_line)
    ski: NumpyFloatArray = spd_func(xki)

    # Change in position on this step
    dx: NumpyFloatArray = np.zeros((m_line, 3), dtype=np.float64)
    # The cumulative arc length along each streamline in the current time step; shape (m_line)
    lk: NumpyFloatArray = np.zeros(m_line, dtype=np.float64)
    # The cumulative time spent along each streamline in the current time step; shape (m_line)
    tk: NumpyFloatArray = np.zeros(m_line, dtype=np.float64)
    # The length of the current time step; shape (m_line)
    dt: NumpyFloatArray = np.zeros(m_line, dtype=np.float64)
    # Mask indicating if we are still in the domain
    mask: NumpyBoolArray = np.all(lo <= xk, axis=1) & np.all(xk <= hi, axis=1)
    # Mask indicating if we are on the domain at the end of the time step
    mask_end: NumpyBoolArray = np.zeros(m_line, dtype=np.bool_)
    
    # Iterate over points in the streamline; quit early if we leave the domain
    iter = tqdm(list(range(n_max)), desc='Integrating streamlines', leave=False)
    for k in iter:
        # Save the current position, velocity, time and point number of the streamline
        X[:, k] = xk
        V[:, k] = vk
        L[:, k] = lk
        T[:, k] = tk
        K[mask] = k

        # Change in position on this step
        dx[mask] = cx * (uk[mask] + 3.0 * uki[mask])

        # Calculate the new position; only update points that are still in the domain
        xk[mask] += dx[mask]

        # Which points are in the domain at the end of the time step?
        mask_end = np.all(lo <= xk, axis=1) & np.all(xk <= hi, axis=1)
        # Update mask for points that are still in the domain
        mask = np.logical_and(mask, mask_end)

        # Calculate the new direction on points still in the domain on this step
        uk[mask] = dir_func(xk[mask])
        # The new intermediate position
        xki[mask] = xk[mask] + cw * uk[mask]
        # The new intermediate direction
        uki[mask] = dir_func(xki[mask])

        # Update the velocity at the start of this time point; zero out points that left the domain
        vk[mask] = vel_func(xk[mask])
        vk[~mask] = 0.0

        # Update the speed at the start and intermediate points
        sk[mask] = spd_func(xk[mask])
        ski[mask] = spd_func(xki[mask])

        # Update the arc length of the streamline
        lk[mask] += np.sqrt(np.sum(np.square(dx[mask]), axis=-1))

        # The amount of time spent in this step
        dt[mask] = (ct1 / sk[mask]) + (ct2 / ski[mask])
        # Update the time spent on each streamline
        tk[mask] += dt[mask]

        # If all points have left the domain, break the loop
        if np.all(~mask):
            print(f'Quitting early on step {k:d} - all points out of domain.')
            break

    return X, V, L, T, K

# *************************************************************************************************
def summarize_run(T: NumpyFloatArray, K: NumpyInt16Array, ds: float) -> None:
    """Summarize the results of the streamline integration to the console"""
    # Summary of number of points in each streamline, k    
    k_mean: float = np.mean(K).astype(float)
    k_min: int = np.min(K).astype(int)
    k_max: int = np.max(K).astype(int)
    k_99: int = np.quantile(K, 0.99).astype(int)

    # Average speed of each streamline
    mask: NumpyBoolArray = (K > 0)
    # Length of valid streamlines
    stream_length: NumpyFloatArray = ds * K[mask]
    # Time at the end of each streamline
    t_end: NumpyFloatArray = np.max(T, axis=1)
    # Time spent on valid streamlines
    stream_time: NumpyFloatArray = t_end[mask]
    # Average speed on valid streamlines
    sm: NumpyFloatArray = stream_length / stream_time
    # Summary of the average speed
    sm_min: float  = np.min(sm).astype(float)
    sm_max: float  = np.max(sm).astype(float)
    sm_mean: float = np.mean(sm).astype(float)

    print('Number of points on streamlines:')
    print(f'Mean: {k_mean:0.3f}')
    print(f'Min : {k_min:d}')
    print(f'Max : {k_max:d}')
    print(f'99% : {k_99:d}')
    print('Average speed on streamlines:')
    print(f'Mean: {sm_mean:9.6f}')
    print(f'Min : {sm_min:9.2e}')
    print(f'Max : {sm_max:9.6f}')
    # print('Global max speed (for scalebar):')

# *************************************************************************************************
def resample_streamlines(Xi: NumpyFloatArray, Vi: NumpyFloatArray, Ti: NumpyFloatArray, Li: NumpyFloatArray, 
                         Ki: NumpyInt16Array, ds: float) \
        -> tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray, NumpyFloatArray, NumpyInt16Array]:
    """
    Resample the streamlines with even spacing in arc length
    INPUTS:
        Xi: NumpyFloatArray - position on integrated streamlines; array of shape (n_pts, n_max_i, 3)
        Vi: NumpyFloatArray - velocity on integrated streamlines; array of shape (n_pts, n_max_i, 3)
        Ti: NumpyFloatArray - time spent on integrated streamlines; array of shape (n_pts, n_max_i)
        Li: NumpyFloatArray - arc length on integrated streamlines; array of shape (n_pts, n_max_i)
        Ki: NumpyInt16Array - number of points on each integrated streamline; shape (n_pts)
        ds: float - distance step size for the resampled streamlines, in microns
    RETURNS:
        Xr: NumpyFloatArray - position on resampled streamlines; array of shape (n_pts, n_max, 3)
        Vr: NumpyFloatArray - velocity on resampled streamlines; array of shape (n_pts, n_max, 3)
        Tr: NumpyFloatArray - time spent on resampled streamlines; array of shape (n_pts, n_max_i)
        Lr: NumpyFloatArray - arc length on resampled streamlines; array of shape (n_pts, n_max)
        Kr: NumpyInt16Array - number of points on each resampled streamline; shape (n_pts)
    """
    # Number of streamlines; this is the same for integrated and resampled streamlines
    m_line: int = Xi.shape[0]
    # Longest streamline
    l_max: float = float(np.max(Li))
    # Number of points on the longest resampled streamline
    n_max: int = int(np.floor(l_max / ds + 1))

    # Initialize the resampled arrays
    Xr: NumpyFloatArray = np.zeros((m_line, n_max, 3), dtype=np.float64)
    Vr: NumpyFloatArray = np.zeros((m_line, n_max, 3), dtype=np.float64)
    Tr: NumpyFloatArray = np.zeros((m_line, n_max), dtype=np.float64)
    Lr: NumpyFloatArray = np.zeros((m_line, n_max), dtype=np.float64)
    Kr: NumpyInt16Array = np.zeros(m_line, dtype=np.int16)

    # Iterate over the streamlines, resampling each one in turn
    iter = tqdm(list(range(m_line)), desc='Resampling streamlines', leave=False)
    for j in iter:
        # Number of points on the current integrated streamline
        ki: int = Ki[j]
        # Skip streamlines with no points
        if ki == 0:
            continue
        # Arc length of the current integrated streamline
        li: NumpyFloatArray = Li[j, 0:ki]
        # Position of the current integrated streamline
        xi: NumpyFloatArray = Xi[j, 0:ki, :]
        # Velocity of the current integrated streamline
        vi: NumpyFloatArray = Vi[j, 0:ki, :]
        # Time spent on the current integrated streamline
        ti: NumpyFloatArray = Ti[j, 0:ki]
        # Number of points on the resampled streamline
        kr: int = int(np.floor(li[-1] / ds + 1))
        # Arc length of the resampled streamline
        lr: NumpyFloatArray = np.arange(0, kr, dtype=np.float64) * ds
        # Mask of strictly increasing arc length; for pathological cases where the arc length doesn't increase
        mask: NumpyBoolArray = (np.diff(Li[j, 0:ki+1]) > 0.0)
        # Cubic spline interpolator for position
        x_func: CubicSpline = CubicSpline(li[mask], xi[mask])
        # Cubic spline interpolator for velocity
        v_func: CubicSpline = CubicSpline(li[mask], vi[mask])
        # Cubic spline interpolator for time
        t_func: CubicSpline = CubicSpline(li[mask], ti[mask])
        # Save the resampled streamline position
        Xr[j, 0:kr] = x_func(lr)
        # Save the resampled streamline velocity
        Vr[j, 0:kr] = v_func(lr)
        # Save the resampled streamline time
        Tr[j, 0:kr] = t_func(lr)
        # Save the resampled arc length; this should be the same as [0.0, ds, ... , (kr-1)*ds]
        Lr[j, 0:kr] = lr
        # Save the number of points on the resampled streamline
        Kr[j] = kr

    return Xr, Vr, Tr, Lr, Kr

# *************************************************************************************************
def integrate_all(stride: int, ds: float) -> None:
    """
    Integrate all the streamlines
    INPUTS:
        stride - stride length for streamline origins
        n_max - maximum number of points on one streamline
        ds - distance step size for the integrated streamlines, in microns
    """
    # Load the velocity field
    vel: NumpyFloatArray = load_vel()

    # Load the domain
    lo, hi = load_domain()

    # Shape of the domain - get from vel
    shape: tuple[int, int, int] = vel.shape[0:3]

    # Calculate geometry
    geom: Geometry = calc_geometry(lo=lo, hi=hi, shape=shape)

    # Maximum number of points on one streamline
    diag: float = np.sqrt(np.sum(np.square(hi - lo)))
    n_max: int = int(np.ceil(2.0 * diag / ds))

    # Set minimum velocity along the x-axis for interpolation
    speed: NumpyFloatArray = np.sqrt(np.nansum(np.square(vel), axis=-1))
    speed_is_pos: NumpyBoolArray = (speed > 0.0)
    min_pos_speed: float = np.min(speed[speed_is_pos]).astype(float)
    vel_min: float = min_pos_speed * 1.0E-2

    # Interpolator for the velocity field
    vel_func: RegularGridInterpolator 
    # Interpolator for the unit vector
    dir_func: RegularGridInterpolator
    # Interpolator for the speed
    spd_func: RegularGridInterpolator
    # Create the interpolators
    vel_func, dir_func, spd_func = make_interpolators(vel=vel, geom=geom, vel_min=vel_min)

    # Unpack the cell centers; only take every stride-th point
    yc: NumpyFloatArray = geom.yc[stride//2::stride]
    zc: NumpyFloatArray = geom.zc[stride//2::stride]
    # Number of streamlines to integrate
    m_line: int = yc.shape[0] * zc.shape[0]

    # Always start streamlines at the left edge of the domain (first cell center to be precise)
    x0_x: NumpyFloatArray = np.full(m_line, geom.xc[0], dtype=np.float64)
    # Meshgrid in the YZ plane for streamline origins
    x0_y, x0_z = np.meshgrid(yc, zc, indexing='ij')
    # Flatten the meshgrids
    x0_y = x0_y.flatten()
    x0_z = x0_z.flatten()

    # Starting points of streamlines
    x0: NumpyFloatArray = np.stack((x0_x, x0_y, x0_z), axis=1)

    # Integrate the streamlines
    Xi: NumpyFloatArray
    Vi: NumpyFloatArray
    Li: NumpyFloatArray
    Ti: NumpyFloatArray
    Ki: NumpyInt16Array
    Xi, Vi, Li, Ti, Ki = integrate_streamlines(
        vel_func=vel_func, dir_func=dir_func, spd_func=spd_func, geom=geom, x0=x0, ds=ds, n_max=n_max)

    # Summarize the results on the console
    summarize_run(T=Ti, K=Ki, ds=ds)

    # Save the results of the integrated streamlines to files
    np.save(path_out / 'Xi.npy', Xi)
    np.save(path_out / 'Vi.npy', Vi)
    np.save(path_out / 'Ti.npy', Ti)
    np.save(path_out / 'Li.npy', Li)
    np.save(path_out / 'Ki.npy', Ki)

# *************************************************************************************************
def resample_all(ds: float) -> None:
    """
    Resample all the streamlines
    INPUTS:
        ds - Distance step size for the resampled streamlines , in microns. 
             Not necessarily the same as for the integrated streamlines.
    """
    # Load the integrated streamlines
    Xi: NumpyFloatArray = np.load(path_out / 'Xi.npy')
    Vi: NumpyFloatArray = np.load(path_out / 'Vi.npy')
    Ti: NumpyFloatArray = np.load(path_out / 'Ti.npy')
    Li: NumpyFloatArray = np.load(path_out / 'Li.npy')
    Ki: NumpyInt16Array = np.load(path_out / 'Ki.npy')
    
    # Resample the streamlines
    Xr: NumpyFloatArray
    Vr: NumpyFloatArray
    Tr: NumpyFloatArray
    Lr: NumpyFloatArray
    Kr: NumpyInt16Array
    Xr, Vr, Tr, Lr, Kr = resample_streamlines(Xi=Xi, Vi=Vi, Ti=Ti, Li=Li, Ki=Ki, ds=ds)

    # Save the results of the resampled streamlines to files
    np.save(path_out / 'Xr.npy', Xr)
    np.save(path_out / 'Vr.npy', Vr)
    np.save(path_out / 'Tr.npy', Tr)
    np.save(path_out / 'Lr.npy', Lr)
    np.save(path_out / 'Kr.npy', Kr)

# *************************************************************************************************
def main(do_integrate: bool, do_resample: bool) -> None:
    """
    Console program
    INPUTS:
        do_integrate - Should we integrate the streamlines?
        do_resample - Should we resample the streamlines
    """
    # Stride for streamline origins
    stride: int = 2

    # Distance step size for integration, in microns
    ds_i: float = 0.25

    # Distance step size for resampling, in microns
    ds_r: float = 1.00

    # Report actions to be taken and parameters
    print('Actions to be taken and parameter values:')
    if do_integrate:
        print(f'Integrate streamlines')
        print(f'Streamline Origin Stride : {stride:d}')
        print(f'Integration step size    : {ds_i:0.2f} microns\n')
    if do_resample:
        print(f'Resample streamlines')
        print(f'Resample step size       : {ds_r:0.2f} microns\n')

    # Integrate all the streamlines if applicable
    if do_integrate:
        integrate_all(stride=stride, ds=ds_i)

    # Resample all the streamlines if applicable
    if do_resample:
        resample_all(ds=ds_r)

# *************************************************************************************************
if __name__ == "__main__":
    # Should we integrate the streamlines?
    do_integrate: bool = False
    # Should we resample the streamlines?
    do_resample: bool = False
    # Run the main program
    main(do_integrate=do_integrate, do_resample=do_resample)
