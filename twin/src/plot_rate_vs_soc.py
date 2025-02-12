import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from pathlib import Path

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# *************************************************************************************************
# Directory for output figures
dir_fig: Path = Path('figs', '10_misc')

# Set Matplotlib parameters
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14

# *************************************************************************************************
def plot_rate(Vs: NumpyFloatArray, colors: list[str], fig_style: bool) -> None:
    """
    Build the plot for the nondimensionalized rate vs. state of charge.
    INPUTS:
        Vs: NumpyFloatArray - The sample values of the nondimensionalized applied voltage
            V = -n_e V_app / V_t
        colors: list[str] - The colors for each curve.
        fig_style:  Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    # Number of sample voltages
    k: int = Vs.shape[0]
    # Number of sample values for the state of charge
    m: int = 256
    # Sample values for the state of charge
    s = np.arange(0.5 / m, 1.0, 1.0 / m)
    # The overpotential
    eta = np.zeros((k, m))
    # The reaction rate
    rate = np.zeros((k, m))
    # Calculate the overpotential and rate for each sample voltage
    for i, V in enumerate(Vs):
        eta[i, :] = V - np.log(s / (1.0 - s))
        rate[i, :] = 2.0 * np.sqrt(s * (1.0 - s)) * np.sinh(0.5 * eta[i])
    # Calculate equilbium SOC for each voltage
    soc_eq = 1.0 / (1.0 + np.exp(-Vs))

    # Create a figure for rate vs. state of charge
    fig, ax1 = plt.subplots()
    # Create a twin axis for overpotential
    ax2 = ax1.twinx()
    # Set the title
    if not fig_style:
        ax1.set_title(r'Rate and Overpotential vs. SOC', fontsize=20)
    # Set axis labels
    fontsize: int = 17 if fig_style else 14
    ax1.set_xlabel(r'State of Charge $s$', fontsize=fontsize)
    ax1.set_ylabel(r'Reaction Rate $\tilde{R} = 2 \sqrt{s(1-s)} \sinh(\tilde{\eta}/2)$', fontsize=fontsize)
    ax2.set_ylabel(r'Overpotential $\tilde{\eta} = \tilde{V} - \log(s/(1-s))$', fontsize=fontsize)
    # Set limits
    ax1.set_xlim([0.0, 1.0])
    ax2.set_xlim([0.0, 1.0])
    # Increase the font size of the tick labels
    labelsize: int = 15 if fig_style else 13
    ax1.tick_params(axis='both', labelsize=labelsize)
    ax2.tick_params(axis='both', labelsize=labelsize)

    # Plot the rates for each voltage
    for i, V in enumerate(Vs):
        label = r"$\tilde{V}=$" + f" {V:.2f}"
        ax1.plot(s, rate[i], label=label, color=colors[i])

    # Plot the overpotentials for each voltage
    for i, V in enumerate(Vs):
        label = r"$\tilde{V}=$" + f" {V:.2f}"
        ax2.plot(s, eta[i], label=label, color=colors[i], linestyle='--')

    # Horizontal line for equilibrium (rate = 0)
    ax1.axhline(y=0.0, color='black', linestyle='--')
    # Vertical lines for equilibrum SOC
    for i, V in enumerate(Vs):
        ax1.axvline(x=soc_eq[i], color=colors[i], linestyle='--', alpha=0.5)

    # Additional formatting
    ax1.legend(frameon=True, fancybox=True)
    # ax1.grid(axis='x')
    ax1.grid(linestyle='--')

    # Save the figure
    fig.savefig(dir_fig / '01_rate_vs_soc.png', bbox_inches='tight')

# *************************************************************************************************
def main():
    """Build the plot"""
    # Selected sample voltages and colors
    Vs: NumpyFloatArray = np.array([0.0, 1.0, 2.0])
    colors: list[str] = ['blue', 'red', 'green']
    fig_style: bool = True
    # Delegate the plotting to plot_rate()
    plot_rate(Vs=Vs, colors=colors, fig_style=fig_style)

# *************************************************************************************************
if __name__ == '__main__':
    main()
