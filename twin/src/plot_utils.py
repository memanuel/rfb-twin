import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Local imports
from utils import Geometry, FlowSim, ReactSim

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]

# *************************************************************************************************
# Set Matplotlib parameters
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

# *************************************************************************************************
def plot_scalar(phi: NumpyFloatArray, geom: Geometry, k: int, vmin: float, vmax: float, cmap: Colormap,
                cbar_fmt: str, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot a scalar field at a given z-slice
    INPUTS:
        phi:  Array of data to plot; shape (nx, ny, nz)
        geom: Geometry object
        k: Index of z-slice to plot
        vmin: Minimum value for the color map
        vmax: Maximum value for the color map
        cmap: Colormap object
        cbar_fmt: Number format for the color bar
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Extract the data for the selected z-slice
    s: NumpyFloatArray = phi[:,:,k]

    # Create the figure
    fig = plt.figure(figsize=(10.0, 5.0))
    # Place axes in the figure
    height = 0.80
    width = 0.875
    width_cbar = 0.05
    width_tot = width + width_cbar
    left = (1.0 - width_tot) * 0.80
    bottom = (1.0 - height) * 0.50
    rect = (left, bottom, width, height)
    ax = fig.add_axes(rect=rect)

    # Build the 2D colormesh plot
    mesh = ax.pcolormesh(s.T, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add the colorbar on its own custom axes
    divider = make_axes_locatable(ax)
    cbar_size_pct = 5.0
    cbar_size_str = f'{cbar_size_pct:0.3f}%'
    pad = 0.10
    format = cbar_fmt
    cax = divider.append_axes("right", size=cbar_size_str, pad=pad)
    fig.add_axes(cax)
    cbar = fig.colorbar(mappable=mesh, ax=ax, cax=cax, format=format, pad=pad)
    # Set font size in colorbar automatically
    cbar_entry = f'{np.nanmean(s):{cbar_fmt.replace('%','')}}'
    labelsize = 10 if len(cbar_entry) < 5 else 9
    cbar.ax.tick_params(labelsize=labelsize)

    # Plot title
    if not fig_style:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title(None)
    # Axis labels
    fontsize: int = 20 if fig_style else 16
    ax.set_xlabel('x ($\\mu$m)', fontsize=fontsize)
    ax.set_ylabel('y ($\\mu$m)', fontsize=fontsize)

    # The tick positions in index space
    xticks = np.linspace(0, geom.nx, 5, dtype=np.int32)
    yticks = np.linspace(0, geom.ny, 5, dtype=np.int32)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Convert locations from cm to microns
    cm2um: float = 1.0e4
    lo: NumpyFloatArray = geom.lo * cm2um
    hi: NumpyFloatArray = geom.hi * cm2um

    # The tick labels
    labelsize: int = 16 if fig_style else 13
    xticklabels = np.linspace(lo[0], hi[0], 5, dtype=np.int32)
    yticklabels = np.linspace(lo[1], hi[1], 5, dtype=np.int32)
    ax.set_xticklabels(xticklabels, fontsize=labelsize)
    ax.set_yticklabels(yticklabels, fontsize=labelsize)

    # Save the figure and close it
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# *************************************************************************************************
def plot_scalar_yz(phi: NumpyFloatArray, geom: Geometry, i: int, vmin: float, vmax: float, cmap: Colormap,
                cbar_fmt: str, title: str, fig_style: bool, path: Path) -> None:
    """
    Plot a scalar field at a given x-slice
    INPUTS:
        phi:  Array of data to plot; shape (nx, ny, nz)
        geom: Geometry object
        i: Index of x-slice to plot
        vmin: Minimum value for the color map
        vmax: Maximum value for the color map
        cmap: Colormap object
        cbar_fmt: Number format for the color bar
        title: Title of the plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Extract the data for the selected z-slice
    s: NumpyFloatArray = phi[i,:,:]

    # Create the figure
    fig = plt.figure(figsize=(10.0, 2.5))
    # Place axes in the figure
    height = 0.80
    width = 0.875
    width_cbar = 0.05
    width_tot = width + width_cbar
    left = (1.0 - width_tot) * 0.80
    bottom = (1.0 - height) * 0.50
    rect = (left, bottom, width, height)
    ax = fig.add_axes(rect=rect)

    # Build the 2D colormesh plot
    mesh = ax.pcolormesh(s.T, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add the colorbar on its own custom axes
    divider = make_axes_locatable(ax)
    cbar_size_pct = 5.0
    cbar_size_str = f'{cbar_size_pct:0.3f}%'
    pad = 0.10
    format = cbar_fmt
    cax = divider.append_axes("right", size=cbar_size_str, pad=pad)
    fig.add_axes(cax)
    cbar = fig.colorbar(mappable=mesh, ax=ax, cax=cax, format=format, pad=pad)
    # Set font size in colorbar automatically
    cbar_entry = f'{np.nanmean(s):{cbar_fmt.replace('%','')}}'
    labelsize = 10 if len(cbar_entry) < 5 else 9
    cbar.ax.tick_params(labelsize=labelsize)

    # Plot title
    if not fig_style:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title(None)
    # Axis labels
    fontsize: int = 20 if fig_style else 16
    ax.set_xlabel('y ($\\mu$m)', fontsize=fontsize)
    ax.set_ylabel('z ($\\mu$m)', fontsize=fontsize)

    # The tick positions in index space
    xticks = np.linspace(0, geom.ny, 5, dtype=np.int32)
    yticks = np.linspace(0, geom.nz, 5, dtype=np.int32)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Convert locations from cm to microns
    cm2um: float = 1.0e4
    lo: NumpyFloatArray = geom.lo * cm2um
    hi: NumpyFloatArray = geom.hi * cm2um

    # The tick labels
    labelsize: int = 16 if fig_style else 13
    xticklabels = np.linspace(lo[1], hi[1], 5, dtype=np.int32)
    yticklabels = np.linspace(lo[2], hi[2], 5, dtype=np.int32)
    ax.set_xticklabels(xticklabels, fontsize=labelsize)
    ax.set_yticklabels(yticklabels, fontsize=labelsize)

    # Save the figure and close it
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# *************************************************************************************************
def plot_speed_impl(speed: NumpyFloatArray, geom: Geometry, P: int, k: int, fig_style: bool, path: Path) -> None:
    """
    Plot speed of one flow simulation
    INPUTS:
        speed: Array of speed values; shape (nx, ny, nz)
        geom: Geometry object
        P: Pressure at the inlet
        k: Index of z-slice to plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Extract the speed for the selected z-slice
    s: NumpyFloatArray = speed[:,:,k]
    # Corresponding z-coordinate
    z: float = geom.zn[k]

    # Create the figure
    fig = plt.figure(figsize=(10.0, 5.0))
    # Place axes in the figure
    height = 0.80
    width = 0.80
    width_cbar = 0.05
    width_tot = width + width_cbar
    left = (1.0 - width_tot) / 2.0
    bottom = (1.0 - height) / 2.0
    rect = (left, bottom, width, height)
    ax = fig.add_axes(rect=rect)

    # Build the 2D colormesh plot
    cmap: Colormap = plt.get_cmap('jet')
    vmin: float = 0.0
    vmax: float = np.nanmax(s).astype(float)
    mesh = ax.pcolormesh(s.T, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add the colorbar on its own custom axes
    divider = make_axes_locatable(ax)
    cbar_size_pct = 5.0
    cbar_size_str = f'{cbar_size_pct:0.3f}%'
    pad = 0.10
    format = '%0.03f'
    cax = divider.append_axes("right", size=cbar_size_str, pad=pad)
    fig.add_axes(cax)
    fig.colorbar(mappable=mesh, ax=ax, cax=cax, format=format, pad=pad)

    # Plot title
    if not fig_style:
        ax.set_title(f'Flow Speed (cm/s) - P = {P} Pa, z = {z:.0f} $\\mu$m', fontsize=20)
    else:
        ax.set_title(None)
    # Axis labels
    fontsize: int = 20 if fig_style else 16
    ax.set_xlabel('x ($\\mu$m)', fontsize=fontsize)
    ax.set_ylabel('y ($\\mu$m)', fontsize=fontsize)

    # The tick positions in index space
    xticks = np.linspace(0, geom.nx, 5, dtype=np.int32)
    yticks = np.linspace(0, geom.ny, 5, dtype=np.int32)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Convert locations from cm to microns
    cm2um: float = 1.0e4
    lo: NumpyFloatArray = geom.lo * cm2um
    hi: NumpyFloatArray = geom.hi * cm2um

    # The tick labels
    labelsize: int = 16 if fig_style else 13
    xticklabels = np.linspace(lo[0], hi[0], 5, dtype=np.int32)
    yticklabels = np.linspace(lo[1], hi[1], 5, dtype=np.int32)
    ax.set_xticklabels(xticklabels, fontsize=labelsize)
    ax.set_yticklabels(yticklabels, fontsize=labelsize)

    # Save the figure and close it
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# *************************************************************************************************
def plot_speed(sim: FlowSim, k: int, fig_style: bool, path: Path) -> None:
    """
    Plot speed of one flow simulation
    INPUTS:
        sim: Flow simulation data
        k: Index of z-slice to plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Unpack the geometry
    geom: Geometry = sim.geom
    # Unpack the inlet pressure
    P: int = sim.p_in
    # Extract the speed array
    speed: NumpyFloatArray = sim.speed
    # Delegate to the implementation
    plot_speed_impl(speed=speed, geom=geom, P=P, k=k, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_pressure(sim: FlowSim, k: int, fig_style: bool, path: Path) -> None:
    """
    Plot pressure of one flow simulation
    INPUTS:
        sim: Flow simulation data
        k: Index of z-slice to plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Unpack the geometry
    geom: Geometry = sim.geom
    # Unpack the inlet pressure
    P: int = sim.p_in
    # Extract the pressure array (cell centered)
    p_cell: NumpyFloatArray = sim.pressure_cell
    # Set min and max for the color map
    vmin: float = 0.0
    vmax: float = np.nanmax(p_cell).astype(float)
    cmap: Colormap = plt.get_cmap('jet')
    cbar_fmt: str = '%0.01f'
    # Title
    title: str = f'Pressure (Pa) - P = {P} Pa, z = {geom.zn[k]:.0f} $\\mu$m'
    # Delegate to the implementation
    plot_scalar(phi=p_cell, geom=geom, k=k, vmin=vmin, vmax=vmax, cmap=cmap, cbar_fmt=cbar_fmt, 
                title=title, fig_style=fig_style, path=path)

# *************************************************************************************************
def plot_soc_impl(soc: NumpyFloatArray, geom: Geometry, P: int, k: int, fig_style: bool, path: Path) -> None:
    """
    Plot state of charge of one reaction simulation from a numpy array of SOC values
    INPUTS:
        soc: Array of state of charge values; shape (nx, ny, nz)
        geom: Geometry object
        P: Pressure at the inlet
        k: Index of z-slice to plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Extract the SOC for the selected z-slice
    s: NumpyFloatArray = soc[:,:,k]
    # Corresponding z-coordinate
    z: float = geom.zn[k]

    # Create the figure
    fig = plt.figure(figsize=(10.0, 5.0))
    # Place axes in the figure
    height = 0.80
    width = 0.80
    width_cbar = 0.05
    width_tot = width + width_cbar
    left = (1.0 - width_tot) / 2.0
    bottom = (1.0 - height) / 2.0
    rect = (left, bottom, width, height)
    ax = fig.add_axes(rect=rect)

    # Build the 2D colormesh plot
    cmap: Colormap = plt.get_cmap('jet')
    vmin: float = 0.0
    # vmax: float = 1.0
    vmax: float = np.nanmax(s).astype(float)
    mesh = ax.pcolormesh(s.T, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add the colorbar on its own custom axes
    divider = make_axes_locatable(ax)
    cbar_size_pct = 5.0
    cbar_size_str = f'{cbar_size_pct:0.3f}%'
    pad = 0.10
    format = '%0.03f'
    cax = divider.append_axes("right", size=cbar_size_str, pad=pad)
    fig.add_axes(cax)
    fig.colorbar(mappable=mesh, ax=ax, cax=cax, format=format, pad=pad)

    # Title
    if not fig_style:
        ax.set_title(f'State of Charge - P = {P} Pa, z = {z:.0f} $\\mu$m', fontsize=14)
    else:
        ax.set_title(None)
    # Axis labels
    fontsize: int = 20 if fig_style else 16
    ax.set_xlabel('x ($\\mu$m)', fontsize=fontsize)
    ax.set_ylabel('y ($\\mu$m)', fontsize=fontsize)

    # The tick positions in index space
    xticks = np.linspace(0, geom.nx, 5, dtype=np.int32)
    yticks = np.linspace(0, geom.ny, 5, dtype=np.int32)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Convert locations from cm to microns
    cm2um: float = 1.0e4
    lo: NumpyFloatArray = geom.lo * cm2um
    hi: NumpyFloatArray = geom.hi * cm2um

    # The tick labels
    labelsize: int = 16 if fig_style else 13
    xticklabels = np.linspace(lo[0], hi[0], 5, dtype=np.int32)
    yticklabels = np.linspace(lo[1], hi[1], 5, dtype=np.int32)
    ax.set_xticklabels(xticklabels, fontsize=labelsize)
    ax.set_yticklabels(yticklabels, fontsize=labelsize)

    # Save the figure and close it
    plt.savefig(path)
    plt.close()

# *************************************************************************************************
def plot_soc(sim: ReactSim, k: int, fig_style: bool, path: Path) -> None:
    """
    Plot state of charge of one reaction simulation
    INPUTS:
        sim: Flow simulation data
        k: Index of z-slice to plot
        fig_style: Use style appropriate for a figure, e.g. no title, larger fonts, etc.
        path: Where to save the plot
    """
    # Unpack the geometry
    geom: Geometry = sim.geom
    # Unpack the pressure
    P: int = sim.p_in
    # Extract the SOC
    soc: NumpyFloatArray = sim.soc
    # Delegate to the implementation
    plot_soc_impl(soc=soc, geom=geom, P=P, k=k, fig_style=fig_style, path=path)
