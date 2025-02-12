import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# *************************************************************************************************
# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# Path to load numpy files with integrated streamlines
path_in: Path = Path('npy/11_streamline')

# Path to save the pov files
path_out: Path = Path('povray')
# Grid step size in the underlying numerical simulation
# grid_h: float = 1.25
# Spacing between streamline origin points in the YZ plane
line_spacing: float = 2.5

# *************************************************************************************************
def line2str(X: NumpyFloatArray, V: NumpyFloatArray, K: NumpyInt16Array, T: NumpyFloatArray, 
             i: int, s: int, rfb_func) -> str:
    """
    Write out (k-1)/s cylinders along the i-th streamline and return a string with the povray code.
    INPUTS:
        X - array with the position along the streamlines
        V - array with the velocity along the streamlines
        K - array with the number of points in each streamline
        T - array with the time points along the streamlines
        i - index of the streamline
        s - spacing in index space; cylinders connect points s apart along the streamline
        rfb_func - function to calculate the color of an object based on a speed
    """
    # The number of points on this streamline; sample every s points
    k: np.int16 = K[i] // s
    # The streamline as a numpy array; sample every s points on the input X array
    x: NumpyFloatArray = X[i][::s]
    # The instantaneous velocity along the streamline; sample every s points on the input V array
    v: NumpyFloatArray = V[i][::s]
    # The time points along the streamline; sample every s points on the input T array
    t: NumpyFloatArray = T[i][::s]
    
    # The instantaneous speed along the sampled streamline; calculate from the velocity array
    speed: NumpyFloatArray = np.linalg.norm(v, axis=1)

    # The POV-Ray string fragment
    pov: str = ""
    # Finish string shared by all spheres and cylinders
    finish : str = "finish {reflection 0.0 specular 0.0 emission 1.0}"

    # Iterate over the points in the streamline and write out spheres at each point
    for j in range(k):
        # Calculate the color of the sphere based on the velocity at this point
        r: float
        g: float
        b: float
        r, g, b = rfb_func(speed[j])
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

        # Calculate the length of the segment and the time spent on it
        dq: float = float(np.linalg.norm(q1 - q0))
        dt: float = float(t[j+1] - t[j])
        # Calculate the average speed along the segment
        speed_avg: float = dq / dt

        # Calculate the color of the cylinder based on the average speed
        r: float
        g: float
        b: float
        r, g, b = rfb_func(speed_avg)

        # Write out the cylinder
        texture: str = f"texture {{pigment {{ color <{r:0.6f}, {g:0.6f}, {b:0.6f}>}} {finish:s} }}"
        pov += f"cylinder {{ {p0:s}, {p1:s}, rs {texture:s} }}\n"

    return pov

# *************************************************************************************************
def include_streamline(y: float, z: float, stride: int) -> bool:
    """
    Should we include this streamline?
    INPUTS:
        y - y-coordinate of the streamline
        z - z-coordinate of the streamline
        stride - desired stride for sampling the streamlines
    """
    # Distance from center to corner of a grid cell
    cc: float = line_spacing / 2.0
    # Index of the streamline in the YZ plane on the numerical grid (line_spacing apart)
    i: int = int(round((y-cc) / line_spacing))
    j: int = int(round((z-cc) / line_spacing))
    # Is the streamline on the grid of selected streamlines, i.e. a multiple of stride in both i and j?
    rem = stride // 2
    return ((i % stride == rem) and (j % stride == rem))

# *************************************************************************************************
def write_pov_file(X: NumpyFloatArray, V: NumpyFloatArray, T: NumpyFloatArray, K: NumpyInt16Array, 
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
    # Number of streamlines
    m_line: int = X.shape[0]

    # Set lower limit of color map to zero speed
    v_min: float = 0.0
    # Set upper limit of color map to maximum speed in the dataset
    v_max: float = np.max(np.linalg.norm(V, axis=2))

    # Build a colormap
    cmap = mpl.colormaps['jet']
    norm = plt.Normalize(vmin=v_min, vmax=v_max)

    def rfb_func(speed: float) -> tuple[float, float, float]:
        r, g, b = cmap(norm(speed))[0:3]
        return (r, g, b,)

    # Name of the output file
    fname_pov: str = f"stream_obj_s{s_origin:02d}.pov"
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
                fh.write(line2str(X=X, V=V, K=K, T=T, i=i, s=s_line, rfb_func=rfb_func))

# *************************************************************************************************
def main():
    """Generate the povray files with the spheres and cylinders along the streamlines"""
    # Load the numpy arrays; use the resampled streamlines
    X: NumpyFloatArray = np.load(path_in / "Xr.npy")
    V: NumpyFloatArray = np.load(path_in / "Vr.npy")
    T: NumpyFloatArray = np.load(path_in / "Tr.npy")
    K: NumpyInt16Array = np.load(path_in / "Kr.npy")

    # Stride for sampling the streamlines in the YZ plane
    s_origin: int = 4
    # Stride for sampling points along each streamline
    s_line: int = 1
    # Extract maximum speed
    spd: NumpyFloatArray = np.linalg.norm(V, axis=2)
    spd_max: float = np.max(spd).astype(float)

    # Report parameters
    print(f"Generating povray files with streamlines.")
    print(f'Number of streamlines: {X.shape[0]:d}')
    print(f"Stride for sampling the streamlines in the YZ plane: {s_origin:d}")
    print(f"Stride for sampling points along each streamline: {s_line:d}")
    print(f"Global maximum speed: {spd_max:0.3f} cm/sec.")

    # Write the povray file
    write_pov_file(X=X, V=V, T=T, K=K, s_origin=s_origin, s_line=s_line)

# *************************************************************************************************
if __name__ == "__main__":
    main()
