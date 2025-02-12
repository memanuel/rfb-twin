import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.stats import linregress
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Local imports
from utils import FlowSim, ReactSim, load_flow_sim_path, load_react_sim_path
from plot_utils import plot_speed, plot_pressure, plot_soc

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# *************************************************************************************************
# Top level directory with simulation data for flows at different pressures
dir_flow: Path = Path('11_stokes_flow')
# Top level directory with simulation data for reactions at the mass transport limit at different pressures
dir_react: Path = Path('12_mass_transport')
# File with the utilization and flow rate data
path_util: Path = Path('12_mass_transport', 'util.csv')

# Directory for output figures
dir_fig_flow: Path = Path('figs', '02_flow')
dir_fig_react: Path = Path('figs', '03_react')
dir_fig_stokes: Path = Path('figs', '04_stokes_flow')

# Physical constants
ne: int = 2
F: float = 96485.3329
c0: float = 20.0E-3
rho: float = 997.0479
mu: float = 0.0008891
length: float = 640E-6
area: float = 640E-6 * 160E-6

# Set Matplotlib parameters
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14

# *************************************************************************************************
def make_P_str(P: int) -> str:
    """Make a string for the pressure"""
    exp: int = int(np.floor(np.log10(float(P))))
    man: int = int(np.round(P / np.power(10.0, exp)))
    P_str: str = f'P{man:d}E{exp:d}'
    return P_str

# *************************************************************************************************
def load_flow_sim(P: int, stage: int, level: int) -> FlowSim:
    """Load the flow simulation for the selected pressure, stage and level"""
    # Generate the path to the flow simulation data
    P_str: str = make_P_str(P)
    dir_sim: Path = dir_flow / P_str
    # Delegate to load_flow_sim_path
    sim: FlowSim = load_flow_sim_path(dir_sim=dir_sim, stage=stage, level=level)
    # Bind the pressure to the simulation object
    sim.p_in = P
    return sim

# *************************************************************************************************
def load_react_sim(P: int, stage: int, level: int) -> ReactSim:
    """Load the reaction simulation for the selected pressure, stage and level"""
    # Generate the path to the reaction simulation data
    P_str: str = make_P_str(P)
    dir_sim: Path = dir_react / P_str
    # Delegate to load_flow_sim_path
    sim: ReactSim =  load_react_sim_path(dir_sim=dir_sim, stage=stage, level=level)
    # Bind the pressure to the simulation object
    sim.p_in = P
    return sim

# *************************************************************************************************
def load_util():
    """
    Load the summary results with all the flow rates and utilizations
    RETURNS:
    df: DataFrame with the following columns:
        P: Pressure in millipascals
        Q: Flow rate in mL / hour
        I: Current in milliamps
        util: utilization
    """
    # Load the data including pressure, flow rate and utilization
    df: pd.DataFrame = pd.read_csv(path_util)
    # Convert flow rate from mL / hour to liters per second
    Q_Lps: NumpyFloatArray = df['Q'].to_numpy() * 1.0E-3 / 3600.0
    # Maximum current in Amperes at 100% utilization
    I_max_A: NumpyFloatArray = Q_Lps * c0 * ne * F
    # Convert max current to milliamps
    I_max_mA: NumpyFloatArray = I_max_A * 1.0E3
    # Add the current to the DataFrame; this is max current times utilization
    df['I'] = I_max_mA * df['util'].to_numpy()
    # Convert the flow rate to a nominal flow speed in meters per second
    Q_m3ps: NumpyFloatArray = Q_Lps * 1.0E-3
    v_mps: NumpyFloatArray = Q_m3ps / area
    v: NumpyFloatArray = v_mps * 1.0E2
    # Add the nominal flow speed in cm/s to the DataFrame
    df['v'] = v
    # Calculate the Reynolds number
    Re: NumpyFloatArray = rho * v * length / mu
    # Add the Reynolds number to the DataFrame
    df['Re'] = Re
    return df

# *************************************************************************************************
def compare_flows(P0: int, P1: int):
    """Compare the flow simulations at two pressures"""
    # Use the finest level for the comparison
    level: int = 1
    stage: int = level + 1
    # Load the flow simulations
    sim0: FlowSim = load_flow_sim(P=P0, stage=stage, level=level)
    sim1: FlowSim = load_flow_sim(P=P1, stage=stage, level=level)
    # Extract the flow velocities
    u0: NumpyFloatArray = np.nan_to_num(sim0.velocity, nan=0.0)
    u1_act: NumpyFloatArray = np.nan_to_num(sim1.velocity, nan=0.0)
    # Scale u1 down by the pressure ratio
    u1: NumpyFloatArray = u1_act * (P0 / P1)
    # Difference in flow velocities
    du: NumpyFloatArray = u1 - u0
    # RMS difference
    rms: float = np.sqrt(np.mean(np.square(du)))
    # Relative RMS difference
    rms_rel: float = rms / np.sqrt(np.mean(np.square(u0)))
    # Report the RMS difference
    print(f"RMS difference between flows at {P0} and {P1}:")
    print(f"Absolute: {rms:8.3e} cm/s")
    print(f"Relative: {rms_rel:8.3e}")

# *************************************************************************************************
def plot_flow_sims(Ps: list[int], stage: int, level: int, kk: list[int], fig_style: bool):
    """
    Build all the flow simulations for the requested pressures.
    INPUTS:
        Ps - list of pressures to process
        stage - stage of the simulation to process
        level - level of the simulation to process
        kk - list of z-slices to plot
        fig_style - Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    print('Building flow plots...')
    # Iterate over the pressures
    serial: int = 1
    for P in tqdm(Ps):
        # Load the flow simulation data
        sim: FlowSim = load_flow_sim(P=P, stage=stage, level=level)
        # Iterate over the z-slices
        for k in kk:
            # Path object for the figures
            P_str: str = make_P_str(sim.p_in)
            fname_spd: str = f'{serial:02d}_speed_{P_str}_z{k:02d}.png'
            fname_prs: str = f'{serial:02d}_pressure_{P_str}_z{k:02d}.png'
            path_spd: Path = dir_fig_flow / fname_spd
            path_prs: Path = dir_fig_flow / fname_prs
            # Plot the flow speed
            plot_speed(sim=sim, k=k, fig_style=fig_style, path=path_spd)
            # Plot the pressure
            plot_pressure(sim=sim, k=k, fig_style=fig_style, path=path_prs)
            # Increment the serial number
            serial += 1

# *************************************************************************************************
def plot_react_sims(Ps: list[int], stage: int, level: int, kk: list[int], fig_style: bool):
    """
    Build all the flow simulations for the requested pressures.
    INPUTS:
        Ps - list of pressures to process
        stage - stage of the simulation to process
        level - level of the simulation to process
        kk - list of z-slices to plot
        fig_style - Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    print('Building reaction plots...')
    # Iterate over the pressures
    serial: int = 1
    for P in tqdm(Ps):
        # Load the flow simulation data
        sim: ReactSim = load_react_sim(P=P, stage=stage, level=level)
        # Iterate over the z-slices
        for k in kk:
            # Path object for the figure
            P_str: str = make_P_str(sim.p_in)
            fname: str = f'{serial:02d}_soc_{P_str}_z{k:02d}.png'
            path: Path = dir_fig_react / fname
            # Plot the state of charge
            plot_soc(sim=sim, k=k, fig_style=fig_style, path=path)
            serial += 1

# **********************************************************************************************************************
def plot_flow(df: pd.DataFrame, fig_style: bool):
    """
    Inputs:
    df: DataFrame with the following columns:
        P: Pressure in pascals
        Q: Flow rate in milliliters per hour
        Re: Reynolds number
    fig_style:  Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    # Extract the pressure
    P: NumpyFloatArray = df['P'].to_numpy()
    # Extract the flow rate
    Q: NumpyFloatArray = df['Q'].to_numpy()
    # Extract the Reynolds number
    Re: NumpyFloatArray = df['Re'].to_numpy()

    # Take logs for regression fit
    log_P: NumpyFloatArray = np.log(P)
    log_Q: NumpyFloatArray = np.log(Q)

    # Min and max pressure for plotting and fitting
    P_min: float = 0.8
    P_max: float = 150000.0
    # Min and max flow rate for plotting
    Q_min: float = 0.030
    Q_max: float = 2000.0

    # Linear regression - consistent with model Q = C * P^alpha
    reg = linregress(x=log_P, y=log_Q)
    # Extract the regression slope
    alpha: float = reg.slope
    # Extract the regression intercept
    C: float = np.exp(reg.intercept)
    # The regression R2 value
    reg_r2: float = reg.rvalue ** 2
    # The regression p value
    reg_p: float = reg.pvalue
    # Fitted line
    P_fit: NumpyFloatArray = np.geomspace(P_min, P_max, 129)
    Q_fit: NumpyFloatArray = C * np.power(P_fit, alpha)

    # Report the fit parameters
    print("Linear Regression log(Q) ~ log(P)")
    print(f"Q = {C:0.6f} * P^{alpha:0.6f}")
    print(f"R2 = {reg_r2:0.9f}")
    print(f"p = {reg_p:8.2e}")
    # Name for the fit series
    fit_label: str = f'Fit: $Q = {C:0.4f} \\; P^{{{alpha:0.4f}}}$'

    # Calculate the flow rate and pressure corresponding to the onset of turbulence (Re=3000)
    Q_vs_Re: CubicSpline = CubicSpline(Re, Q)
    P_vs_Re: CubicSpline = CubicSpline(Re, P)
    Re_turb: float = 3000.0
    Q_turb: float = Q_vs_Re(Re_turb)
    P_turb: float = P_vs_Re(Re_turb)
    print(f"Flow rate and pressure at Re=3000:")
    print(f"Q: {Q_turb:0.4f} mL/hour")
    print(f"P: {P_turb:0.4f} Pa")

    # ******************************************************************************************************************
    # Create a figure for flow rate vs. pressure
    fig, ax = plt.subplots()
    # Set plot title
    if not fig_style:
        ax.set_title('Simulated Flow Rate vs. Pressure')
    else:
        ax.set_title(None)
    # Set axis labels
    fontsize: int = 20 if fig_style else 14
    ax.set_xlabel('Pressure (Pa)', fontsize=fontsize)
    ax.set_ylabel('Flow Rate (mL / hour)', fontsize=fontsize)
    # Increase the font size of the tick labels
    labelsize: int = 16 if fig_style else 13
    ax.tick_params(axis='both', labelsize=labelsize)
    # Set limits
    ax.set_xlim([P_min, P_max])
    ax.set_ylim([Q_min, Q_max])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Plot the flow rate vs. pressure
    markersize: float = 12.0 if fig_style else 8.0
    ax.plot(P, Q, linewidth=0.0, marker='o', markersize=markersize, markerfacecolor='red', label='Simulation')
    # Plot the fit
    linewidth = 2.0 if fig_style else 1.0
    ax.plot(P_fit, Q_fit, linewidth=linewidth, color='black', label=fit_label)
    # Add horizontal line for expected onset of turbulence
    ax.axhline(y=Q_turb, linewidth=linewidth, color='black', linestyle='--', label=f'Q={Q_turb:0.2f} mL/hour')
    # Add vertical line for expected onset of turbulence
    ax.axvline(x=P_turb, linewidth=linewidth, color='black', linestyle='--', label=f'P={P_turb:0.2f} Pa')
    # Additional formatting
    legendsize: int = 14 if fig_style else 10
    ax.legend(fontsize=legendsize, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
    gridwidth: float = 1.0 if fig_style else 0.5
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=gridwidth)
    # Tight layout for the figure because there is no title
    if fig_style:
        fig.tight_layout()
    # Save the figure
    fig.savefig(dir_fig_stokes / '01_flow_Q.png', bbox_inches='tight')

    # ******************************************************************************************************************
    # Create a figure for Reynolds number vs. pressure
    fig, ax = plt.subplots()
    # Set the title and labels
    ax.set_title('Reynolds Number vs. Pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Reynolds Number (Dimensionless)')

    # Set limits
    ax.set_xlim([P_min, P_max])
    # ax.set_ylim([Q_min, Q_max])
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot the Reynolds Number vs. pressure
    ax.plot(P, Re, linewidth=0.0, marker='o', markersize=8.0, markerfacecolor='red', label='Simulation')
    # Add horizontal line for expected onset of turbulence
    ax.axhline(y=3000, color='black', linestyle='--', label='Re=3000 (Turbulent)')

    # Additional formatting
    ax.grid(linestyle='--')
    ax.legend(frameon=True, fancybox=True)

    # Save the figure
    fig.savefig(dir_fig_stokes / '02_flow_Re.png', bbox_inches='tight')

# **********************************************************************************************************************
def plot_resistance(df: pd.DataFrame, fig_style: bool):
    """
    Inputs:
    df: DataFrame with the following columns:
        P: Pressure in pascals
        Q: Flow rate in milliliters per hour
        Re: Reynolds number
    fig_style:  Use style appropriate for a figure, e.g. no title, larger fonts, etc.
    """
    # Extract the pressure
    P: NumpyFloatArray = df['P'].to_numpy()
    # Extract the flow rate
    Q: NumpyFloatArray = df['Q'].to_numpy()
    # Extract the Reynolds number
    Re: NumpyFloatArray = df['Re'].to_numpy()
    # Calculate the hydraulic resistance
    R: NumpyFloatArray = P / Q
    # Mask to separate stokes and turbulent regimes
    mask_s: NumpyBoolArray = (Re < 3000)
    mask_t: NumpyBoolArray = ~mask_s
    
    # Extract the stokes and turbulent pressures
    P_s: NumpyFloatArray = P[mask_s]
    P_t: NumpyFloatArray = P[mask_t]
    # Extract the stokes and turbulent resistance
    R_s: NumpyFloatArray = R[mask_s]
    R_t: NumpyFloatArray = R[mask_t]

    # Take log for regression fit
    log_P_s: NumpyFloatArray = np.log(P_s)
    log_R_s: NumpyFloatArray = np.log(R_s)

    # Min and max pressure for plotting and fitting
    P_min: float = 0.8
    P_max: float = 150000.0
    # Min and max resistance for plotting
    R_min: float = 25.0
    R_max: float = 80.0

    # Linear regression - consistent with model R = alpha + beta * log(P)
    # reg = linregress(x=P_s, y=R_s)
    reg = linregress(x=log_P_s, y=log_R_s)
    # Extract the regression beta
    beta: float = reg.slope
    # Extract the regression intercept
    alpha: float = reg.intercept
    # The regression R2 value
    reg_r2: float = reg.rvalue ** 2
    # The regression p value
    reg_p: float = reg.pvalue
    # Fitted line
    P_fit: NumpyFloatArray = np.geomspace(P_min, P_max, 129)
    # R_fit: NumpyFloatArray = alpha + beta * P_fit
    R_fit: NumpyFloatArray = np.exp(alpha) * np.power(P_fit, beta)

    # Report the fit parameters
    print("Linear Regression log(R) ~ log(P) (Stokes Regime)")
    # print(f"R  = {alpha:0.6f} + {beta:0.6f} log(P)")
    print(f"R  = {np.exp(alpha):0.6f} \\cdot P^{beta:0.6f}")
    print(f"R2 = {reg_r2:0.9f}")
    print(f"p  = {reg_p:8.2e}")
    # Name for the fit series
    # fit_label: str = f"Fit (Stokes): $R = {alpha:0.4f} \\; + {beta:0.4f} \\ln(P)$"
    fit_label: str = f"Fit (Stokes): $\\ln(R) = {alpha:0.4f} \\; + {beta:0.4f} \\ln(P)$"

    # Calculate the pressure corresponding to the onset of turbulence (Re=3000)
    P_vs_Re: CubicSpline = CubicSpline(Re, P)
    Re_turb: float = 3000.0
    P_turb: float = P_vs_Re(Re_turb)

    # Create a figure
    fig, ax = plt.subplots()

    # Set plot title
    if not fig_style:
        ax.set_title('Simulated Hydraulic Resistance vs. Pressure')
    else:
        ax.set_title(None)
    # Set axis labels
    fontsize: int = 20 if fig_style else 14
    ax.set_xlabel('Pressure (Pa)', fontsize=fontsize)
    ax.set_ylabel('Hydraulic Resistance (mL / hour / Pa)', fontsize=fontsize)
    # Increase the font size of the tick labels
    labelsize: int = 16 if fig_style else 13
    ax.tick_params(axis='both', labelsize=labelsize)

    # Set limits
    ax.set_xlim([P_min, P_max])
    ax.set_ylim([R_min, R_max])
    ax.set_xscale('log')
    ax.set_yscale('log', base=2)

    # Plot the data
    markersize: float = 12.0 if fig_style else 8.0
    ax.plot(P_s, R_s, linewidth=0.0, marker='o', markersize=markersize, markerfacecolor='blue', label='Sim. (Stokes)')
    ax.plot(P_t, R_t, linewidth=0.0, marker='o', markersize=markersize, markerfacecolor='red' , label='Sim. (Turbulent)')
    # Plot the fit
    linewidth = 2.0 if fig_style else 1.0
    ax.plot(P_fit, R_fit, linewidth=linewidth, color='black', label=fit_label)
    # Add vertical line for expected onset of turbulence
    ax.axvline(x=P_turb, linewidth=linewidth, color='black', linestyle='--', label=f'P={P_turb:0.2f} Pa')

    # Additional formatting
    legendsize: int = 14 if fig_style else 10
    ax.legend(fontsize=legendsize, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
    gridwidth: float = 1.0 if fig_style else 0.5
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=gridwidth)

    # Tight layout for the figure because there is no title
    if fig_style:
        fig.tight_layout()

    # Save the figure
    fig.savefig(dir_fig_stokes / '03_flow_R.png', bbox_inches='tight')

# **********************************************************************************************************************
def plot_util(df: pd.DataFrame):
    """
    Inputs:
    df: DataFrame with the following columns:
        P: Pressure in pascals
        I: Current in milliamps
        util: utilization
    """
    # Extract the pressure
    P: NumpyFloatArray = df['P'].to_numpy()
    # Extract the current
    I: NumpyFloatArray = df['I'].to_numpy()
    # Extract the utilization
    U: NumpyFloatArray = df['util'].to_numpy()
    # Extract the Reynolds number
    # Re: NumpyFloatArray = df['Re'].to_numpy()

    # Take logs for regression fit
    log_P: NumpyFloatArray = np.log(P)
    log_I: NumpyFloatArray = np.log(I)

    # Min and max pressure for plotting and fitting
    P_min: float = 0.8
    P_max: float = 150000.0
    # Min and max current for plotting
    # I_min: float = 0.030
    # I_max: float = 2000.0
    # Min and max utilization for plotting
    U_min: float = 0.250
    U_max: float = 0.700

    # Linear regression - consistent with model Q = C * P^alpha
    reg = linregress(x=log_P, y=log_I)
    # Extract the regression slope
    alpha: float = reg.slope
    # Extract the regression intercept
    C: float = np.exp(reg.intercept)
    # The regression R2 value
    reg_r2: float = reg.rvalue ** 2
    # The regression p value
    reg_p: float = reg.pvalue
    # Fitted line
    P_fit: NumpyFloatArray = np.geomspace(P_min, P_max, 129)
    I_fit: NumpyFloatArray = C * np.power(P_fit, alpha)

    # Report the fit parameters
    print("Linear Regression log(I) ~ log(P)")
    print(f"I = {C:0.6f} * P^{alpha:0.6f}")
    print(f"R2 = {reg_r2:0.9f}")
    print(f"p = {reg_p:8.2e}")
    # Name for the fit series
    fit_label: str = f'Fit: $Q = {C:0.4f} \\; P^{{{alpha:0.4f}}}$'

    # ******************************************************************************************************************
    # Create a figure for current vs. pressure
    fig, ax = plt.subplots()
    # Set the title and labels
    ax.set_title('Simulated Current vs. Pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Current (mA)')
    # Set limits
    ax.set_xlim([P_min, P_max])
    # ax.set_ylim([Q_min, Q_max])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Plot the current vs. pressure
    ax.plot(P, I, linewidth=0.0, marker='o', markersize=8.0, markerfacecolor='red', label='Simulation')
    # Plot the fit
    ax.plot(P_fit, I_fit, linewidth=1.0, color='black', label=fit_label)
    # Additional formatting
    ax.grid(linestyle='--')
    ax.legend(frameon=True, fancybox=True)
    # Save the figure
    fig.savefig(dir_fig_stokes / '04_current.png')

    # ******************************************************************************************************************
    # Create a figure for utilization vs. pressure
    fig, ax = plt.subplots()
    # Set the title and labels
    ax.set_title('Utilization vs. Pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Utilization (Dimensionless)')

    # Set limits
    ax.set_xlim([P_min, P_max])
    ax.set_ylim([U_min, U_max])
    ax.set_xscale('log')
    # ax.set_yscale('log')

    # Plot the Reynolds Number vs. pressure
    ax.plot(P, U, linewidth=0.0, marker='o', markersize=8.0, markerfacecolor='red', label='Simulation')

    # Additional formatting
    ax.grid(linestyle='--')
    ax.legend(frameon=True, fancybox=True)

    # Save the figure
    fig.savefig(dir_fig_stokes / '05_util.png')

# *************************************************************************************************
def main():
    """Build all the plots for the Stokes flow study - both flow and reaction at mass transport limit."""
    # Actions to perform
    do_flow_sims: bool = False
    do_react_sims: bool = True
    do_stokes: bool = False
    do_comp: bool = False

    # Plot style
    fig_style:  bool = True

    # Pressures to process
    # Ps: list[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    Ps: list[int] = [10]
    # Stage to process for flow and reaction
    stage_flow: int = 2
    stage_react: int = 1
    # Level of the simulation to process; levels start from 0
    level_flow: int = 1
    level_react: int = 0
    # Z-slices to plot
    kk_flow: list[int] = [32, 48, 64]
    kk_react: list[int] = [16, 24, 32]

    # Build the flow plots if requested
    if do_flow_sims:
        plot_flow_sims(Ps=Ps, stage=stage_flow, level=level_flow, kk=kk_flow, fig_style=fig_style)

    # Build the reaction plots if requested
    if do_react_sims:
        plot_react_sims(Ps=Ps, stage=stage_react, level=level_react, kk=kk_react, fig_style=fig_style)

    # Summarize the Stokes flow if requested
    if do_stokes:
        # Load the utilization data
        df: pd.DataFrame = load_util()
        # Plot the flow data
        plot_flow(df=df, fig_style=fig_style)
        # Plot the resistance data
        plot_resistance(df=df, fig_style=fig_style)
        # Plot the utilization data
        plot_util(df=df)

    # Compare flows at pressures spanning the Stokes regime
    if do_comp:
        # Low pressure vs. reference
        P0: int = 1
        P1: int = 200
        compare_flows(P0=P0, P1=P1)

        # high pressure vs. reference
        P0: int = 200
        P1: int = 2000
        compare_flows(P0=P0, P1=P1)

# *************************************************************************************************
if __name__ == '__main__':
    main()
