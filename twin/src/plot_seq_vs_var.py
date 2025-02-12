import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from pathlib import Path

from constants import V_T, ne

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
def plot_seq(fig_style: bool) -> None:
    """
    Build the plot for equilibrium SOC vs. applied voltage.
    INPUTS:
            fig_style:  Use style appropriate for a figure, e.g. no title, larger fonts, etc.    
    """
    # Voltage range
    v_max: float = 80.0
    v_min: float = -v_max
    v_step: float = 0.25
    # Sample values for the voltage, in millivolts
    V: NumpyFloatArray = np.arange(v_min, v_max, v_step)
    # The nondimensionalized applied voltage
    vt: NumpyFloatArray = ne * V / V_T
    # Calculate equilbium SOC for each voltage
    soc_eq: NumpyFloatArray = 1.0 / (1.0 + np.exp(-vt))

    # Create a figure for rate vs. state of charge
    fig, ax1 = plt.subplots()
    # Set the title
    if not fig_style:
        ax1.set_title(r'Rate and Overpotential vs. SOC', fontsize=20)
    # Set axis labels
    fontsize: int = 20 if fig_style else 14
    ax1.set_ylabel(r'Equilibrium State of Charge $s_{\textrm{eq}}$', fontsize=fontsize)
    ax1.set_xlabel(r'Applied Voltage $V_{\textrm{ar}}$(mV)', fontsize=fontsize)
    # Set limits
    ax1.set_xlim([v_min, v_max])

    # Plot the rates for each voltage
    label = 'soc'
    ax1.plot(V, soc_eq, label=label, color='blue')

    # Additional formatting
    ax1.grid(axis='x')
    ax1.grid(linestyle='--')

    # Add second x-axis
    def fwd(x):
        return x / V_T
    
    def inv(x):
        return x * V_T
    
    ax2 = ax1.secondary_xaxis('top', functions=(fwd, inv))
    ax2.set_xlabel(r'Nondimensionalized Voltage $\tilde{V}_{\textrm{ar}}$', fontsize=fontsize)

    # Increase the font size of the tick labels
    labelsize: int = 16 if fig_style else 13
    ax1.tick_params(axis='both', labelsize=labelsize)
    ax2.tick_params(axis='both', labelsize=labelsize)

    # Save the figure
    fig.savefig(dir_fig / '02_seq_vs_var.png', bbox_inches='tight')

# *************************************************************************************************
def main():
    """Build the plot"""
    # Delegate the plotting to plot_rate()
    fig_style: bool = True
    plot_seq(fig_style=fig_style)

# *************************************************************************************************
if __name__ == '__main__':
    main()
