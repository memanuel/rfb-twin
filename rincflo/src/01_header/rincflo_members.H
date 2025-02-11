// rincflo_members.H
// Fragment of Rincflo.H
// Private members (data) of Rincflo
#pragma once
#include <rincflo_macros.H>

private:

// Is there an embedded boundary in the geometry?
bool m_has_boundary = true;
// Is Rincflo simulating the flow?
bool m_do_flow = false;
// Is Rincflo simulation the reaction?
bool m_do_reaction = false;
// Is Rincflo analyzing the flow and reaction at the end?
bool m_do_analysis = false;
// Is Rincflo writing out numpy data to directory numpy?
bool m_write_numpy = false;
// Is Rincflo reading numpy data from directory numpy?
bool m_read_numpy = false;

// Verbosity level of console output
int m_verbose = 1;

// The nominal height of a 2D simulation
#if (AMREX_IS_2D)
Real m_height = 0.0;
#endif

// Member variables for initial conditions

// Enumeration of special problem types, e.g. 0 for default, 31 for Poiseuille flow
int m_probtype = 0;
// Fluid velocity initial conditions - x component
Real m_ic_u = 0.0;
// Fluid velocity initial conditions - y component
Real m_ic_v = 0.0;
// Fluid velocity initial conditions - z component
Real m_ic_w = 0.0;
// Pressure initial conditions
Real m_ic_p = 0.0;

Vector<Real> m_ic_t;
// Simulation time at each old - previous step; used for fillpatching
Vector<Real> m_t_old;
// Simulation time at each old - current step; used for fillpatching
Vector<Real> m_t_new;

// Times

// The cumulative elapsed time of the simulation
Real m_cur_time     = 0.0;
// The length of the current time step
Real m_dt           = -1.0;
// The legnth of the previous time step
Real m_prev_dt      = -1.0;
// The length of the double old time step
Real m_prev_prev_dt = -1.0;

// Maximum ratio that the time step can increase from the last time step
static constexpr Real m_max_dt_ratio = 1.09050773266525768967;
// static constexpr Real m_max_dt_ratio = std::pow(2.0, 1.0 / 8.0);

// Counter for number of flow steps taken
int m_nstep         = 0;

// Stop flow simulation if cur_time reaches stop_time OR nstep reaches max_step
// OR steady_state = true AND steady_state_tol is reached

// The maximum simulation time before the simulation is terminated (if positive)
Real m_stop_time = -1.0;
// The maximum number of steps taken before the simulation is terminated (if positive)
int m_max_step = -1;
// Is there a termination condition for the flow simulation based on steady state?
bool m_steady_state = true;
// The tolerance 
Real m_steady_state_tol = 1.0e-4;

// Flags indicating the steady state criterion used for flow simulation

// Is the steady state condition for flow max(|u-u_o|+|v-v_o|+|w-w_o|)/dt < tol?
bool m_tol_flow_rate_comp = false;     
// Is the steady state condition for flow ||V-V_o|| / dt < tol?
bool m_tol_flow_rate_mag = false;
// Is the steady state condition for flow ||V-V_o|| / ||V_o|| < tol?
bool m_tol_flow_relative = true;
// Is the steady state condition fro flow // ||V-V_o|| / (dt*||V_o||) < tol?
bool m_tol_flow_relativerate = false;

// Options to control time stepping of the flow simulation

// The CFL parameter for fluid flow; default is 0.75, but must be set in input file
Real m_cfl = 0.00;
// The schedule of CFL parameters for fluid flow
Vector<Real> m_cfl_sched {{0.00, 0.00}};
// The current fixed time step
Real m_fixed_dt = -1.0;
// The schedule of fixed time steps
Vector<Real> m_fixed_dt_sched {{0.00, 0.00}};

// Flag - is the timestep determined automatically?
bool m_react_automatic_dt = true;
// Flag - are we using a single fixed time step?
bool m_using_fixed_dt_single = false;
// Flag - are we using a fixed_dt schedule?
bool m_using_fixed_dt_sched = false;
// Flag - are we using fixed_dt at all?
bool m_using_fixed_dt = false;
// Flag - are we using a single fixed CFL?
bool m_using_cfl_single = false;
// Flag - are we using a CFL schedule?
bool m_using_cfl_sched = false;
// Flag - are we using CFL for time stepping?
bool m_using_cfl = false;
// Flag - are we using a single fixed reaction CFL?
bool m_using_cfl_react_single = false;
// Flag - are we using a reaction CFL schedule?
bool m_using_cfl_react_sched = false;
// Flag - are we using reaction CFL for time stepping?
bool m_using_cfl_react = false;
// Flag - was the last reaction step rolled back due to convergence failure?
bool m_last_react_step_rollback = false;

// Count number of stagnant flow time steps
int m_stagnant_flow_steps = 0;
// Maximum number of stagnant reaction steps before early termination (if positive); otherwise ignored
int m_stagnant_flow_steps_max = 0;

// Initial projection / iterations

// Should we do an initial projection to a diverence free velocity field?
bool m_do_initial_proj = true;
// How many initial iterations should we do with a reduced time step in the flow simulation?
int  m_initial_iterations = 0;
// What is the shrinkage factor for the initial time steps in the flow simulation?
Real m_init_shrink = 0.10;

// Use Boussinesq approximation for buoyancy?
bool m_use_boussinesq = false;

// Use tensor solve (as opposed to separate solves for each velocity component)
bool use_tensor_solve = false;

// Use separate solves for each velocity component with lagged tensor terms
bool use_tensor_correction = false;

// AMR / refinement settings

// Should we cover multiply cut cells?
bool m_cover_multiple_cuts = true;
// Should we refine cut cells? 1 yes, 0 no
int m_refine_cutcells = 1;
// Number of cells to grow around tagged cells
int m_ngrow = 4;
// Should we refine refine a coarse level by coarsening it? Default is 1
int m_build_coarse_level_by_coarsening = 1;
// Do refinement during fluid simulation every m_regrid_flow_int steps; negative means don't regrid this way
int m_regrid_flow_int = -1;  
// Do refinement during reaction simulation every m_regrid_react_int  many steps; negative means don't regrid this way
int m_regrid_react_int = -1; 

// 0=none 1=flow 2=react 3=both
int m_do_rho_ref=0;
int m_do_velx_ref=0;
int m_do_vely_ref=0;
int m_do_velz_ref=0;
int m_do_dervel_ref=0;
int m_do_conc1_ref=0;
int m_do_conc2_ref=0;
int m_do_conc1_ref_below=0;   
int m_do_conc2_ref_below=0;
int m_do_derconc_ref=0;
int m_do_soc_ref=0;
int m_exclude_z_dir_for_dervel_ref=0;

// Interval for writing status update to console
int m_status_int = 1;

// Plotting 
// The interval in steps between plot files
int m_plot_int = -1;
// The interval in seconds between plot files
int m_plot_int_sec = -1;

// Flags for saving data in plot files
// Embedded boundary
int m_plt_cell_type   = 1;
int m_plt_vfrac       = 1;
int m_plt_afrac       = 1;
int m_plt_volume      = 1;
int m_plt_area        = 1;
// Basic fluid flow
int m_plt_velmag      = 1;
int m_plt_velx        = 1;
int m_plt_vely        = 1;
int m_plt_velz        = 1;
int m_plt_p           = 1;
// Basic reaction
int m_plt_conc        = 1;
int m_plt_soc         = 1;
// Electrochemistry and reaction kinetics
int m_plt_epotL       = 1;
int m_plt_epotS       = 1;
int m_plt_potdiff     = 1;
int m_plt_overpot     = 1;
int m_plt_curr        = 1;
int m_plt_curr_dens   = 1;
int m_plt_src         = 1;
int m_plt_kappaL      = 1;
int m_plt_divcgpot    = 1;
int m_plt_flxrhs      = 0;
// Advanced fluid flow
int m_plt_laps        = 0;
int m_plt_cnvt        = 0;
int m_plt_rho         = 0;
int m_plt_gpx         = 1;
int m_plt_gpy         = 0;
int m_plt_gpz         = 0;
int m_plt_vort        = 0;

// Dump plotfiles at as close as possible to the designated period *without* changing dt
Real m_plot_per_approx = -1.0;    

// Dump plotfiles at exactly the designated period by changing dt
Real m_plot_per_exact  = -1.0;

// The last step when a plot was generated
int m_last_plt = -1;

// Prefix for plot filenames
string m_plot_file{"plt"};

// The interval in steps to save checkpoints; -1 signifies don't save any checkpoints
int m_check_int = -1;
// The interval in seconds to save checkpoints
int m_check_int_sec = -1;

// The last step when a checkpoint was generated
int m_last_chk = -1;

// The number of digits appended to the filename for checkpoints
static constexpr int m_chk_min_digits = 7;

// Prefix for the name of checkpoint files
string m_check_file {"chk"};

// The filename of the checkpoint file
string m_restart_file {""};

// The directory (relative) where numpy input flow data is located
string m_numpy_input_dir_flow {""};

// The directory (relative) where numpy input reaction data is located
string m_numpy_input_dir_react {""};

// Apply this multiplicative scale factors to velocity data loaded from numpy; also apply to pressure gradient
Real m_vel_scale_factor = 1.0;

// How many levels of flow data should be loaded from numpy? Missing levels will be upsampled
int m_numpy_lev_flow = max_level + 1;

// What offset should be used when loading numpy flow data from input files?
int m_numpy_lev_offset_flow = 0;

// How many levels of reaction data should be loaded from numpy? Missing levels will be upsampled
int m_numpy_lev_react = max_level + 1;

// What offset should be used when loading numpy reaction data from input files?
int m_numpy_lev_offset_react = 0;

// The directory (relative) where numpy data is to be saved; empty string means to use default numpy_rstep
string m_numpy_output_dir {""};

// The interval in steps to save numpy checkpoints for flow simulations; -1 signifies don't save any checkpoints
int m_flow_numpy_int = -1;
// The interval in steps to save numpy checkpoints for reaction simulations; -1 signifies don't save any checkpoints
int m_react_numpy_int = -1;
// Verbosity of numpy output
int m_numpy_verbose = 0;

// ***************************************************************
// MAC projection
// ***************************************************************
// Verbosity and MLMG parameters are ParmParse with "mac_proj" in the inputs file
// Examples: mac_proj.verbose = 1
//           mac_proj.bottom_verbose = 1
//           mac_proj.maxiter
//           mac_proj.bottom_maxiter
//           mac_proj.bottom_rtol
//           mac_proj.bottom_atol
//           mac_proj.bottom_solver
// More info at "amrex/Src/LinearSolvers/Projections/AMReX_MacProjector.cpp"

std::unique_ptr<MacProjector> mac_proj;
// MLMG configuration for the MAC solve
string m_mac_bottom_solver = "hypre";
int m_mac_mg_verbose = 0;
int m_mac_mg_bottom_verbose = 0;
int m_mac_mg_maxiter = 200;
int m_mac_mg_bottom_maxiter = 200;
int m_mac_mg_max_coarsening_level = 100;
Real m_mac_mg_rtol = 1.0e-12;
Real m_mac_mg_atol = -1.0;

// ***************************************************************
// Nodal solve
// ***************************************************************

// Verbosity and MLMG parameters are ParmParse with "nodal_proj" in the inputs file
// Examples: 
// nodal_proj.verbose = 1
// nodal_proj.bottom_verbose = 1
// nodal_proj.maxiter
// nodal_proj.bottom_maxiter
// nodal_proj.bottom_rtol
// nodal_proj.bottom_atol
// nodal_proj.bottom_solver
// More info at "amrex/Src/LinearSolvers/Projections/AMReX_NodalProjector.cpp"

std::unique_ptr<NodalProjector> nodal_proj;
// MLMG configuration for the nodal solve
string m_nodal_bottom_solver = "bicgcg";
int m_nodal_mg_verbose = 0;
int m_nodal_mg_bottom_verbose = 0;
int m_nodal_mg_maxiter = 200;
int m_nodal_mg_bottom_maxiter = 200;
int m_nodal_mg_max_coarsening_level = 100;
Real m_nodal_mg_rtol = 1.0e-12;
Real m_nodal_mg_atol = -1.0;

// ***************************************************************
// background pressure at node (0,0,0).
Real m_p000 = 0.0;
Vector<Real> m_gp0{{0.0, 0.0, 0.0}};
Vector<Real> m_gravity{{0.0, 0.0, 0.0}};
Vector<Real> m_delp{{0.0, 0.0, 0.0}};
// Is the density constant throughout the simulation?
bool m_constant_density = true;
// Should Rincflo advect species concentrations?
bool m_advect_conc    = false;
bool m_test_species_conservation = false;

#if (AMREX_USE_EB)
string m_advection_type = "MOL";
string m_redistribution_type = "FluxRedist";
#else
string m_advection_type = "Godunov";
#endif

#if (AMREX_USE_EB)
// If using Godunov with EB, default to PLM
bool m_godunov_ppm         = false;
#else
// If using Godunov with no EB, default to PPM
bool m_godunov_ppm         = true;
#endif

// If using Godunov, default to including viscous/diffusive terms in the forcing terms
bool m_godunov_include_diff_in_forcing = false;

// If using Godunov, when to add the forcing terms in the construction of the "trans" velocities
bool m_godunov_use_forces_in_trans = false;

// If using Godunov, if true then don't include the pressure gradient in the forcing term passed into 
// the godunov routine; instead  use gradient of mac phi which contains the full pressure
bool m_use_mac_phi_in_godunov = false;

// The method for calculating diffusion
DiffusionType m_diff_type = DiffusionType::Implicit;

// Fluid properties
FluidModel m_fluid_model;
// Density (if constant) [kg/m^3] at 25 C https://www.usgs.gov/special-topics/water-science-school/science/water-density
Real m_ro_0 = 997.0;
// Absolute viscosity (Newtonian fluid) [kg/(m*s)] https://byjus.com/chemistry/viscosity-of-water/
Real m_mu = 0.00089;
Real m_n_0 = 0.0;
Real m_tau_0 = 0.0;
Real m_papa_reg = 0.0;
Real m_eta_0 = 0.0;

// Is the diffusivity of each species a constant?
bool m_constant_Ds = true;
// Scalar diffusive coefficient of each species [m^2/s]
Vector<Real> m_D_s;

// The eb2 threshold for cells with a small volume fraction (almost all solid) when tagging cells
Real m_eb2_small_volfrac = 1.0 / (1024.0);
// The ebs threshold for cells with a large volume fraction (almost all liquid) when tagging cells
Real m_eb2_large_volfrac = 1.0 - m_eb2_small_volfrac;

// Reaction properties
// Single CFL parameter for reaction
Real m_cfl_react = 0.00;
// The schedule of CFL parameters for fluid flow
Vector<Real> m_cfl_react_sched {{0.00, 0.00}};
// Which type of reaction model? linear, special or ButlerVolmer
ReactionModel m_reaction_model = ReactionModel::none;
// Flag: are we including the electromigration term?
bool m_electromigration = false;
// Flag: are we using mass transfer?
bool m_mass_transfer = false;

// Prefactor of each each species produced (+) or consumed (-) in the electrochemical reaction
Vector<Real> m_s_pref_conc; 
// Prefactor of epotL in constructing right hand side (rhs) or linear equation
Real m_s_pref_epotL;
// Prefactor of epotS in constructing right hand side (rhs) or linear equation
Real m_s_pref_epotS;

// Charge of each simulated species
Vector<int> m_z_s;              
// Diffusivity [m^2/s] of extra species (transport not explictly modeled)
Real m_D_ex = -1;
// Charge of extra species (transport not explictly modeled)
int m_z_ex = 0;
// Is there an extra species whose concentration is determined by charge neutrality, e.g. HS04- ?
bool m_extra_species = false;   

// Auxiliary (derived) physical constants
Vector<Real> m_Dz;                  // =D*z
Vector<Real> m_Dz_ex;               // =z*(D-D_ex)
Vector<Real> m_Dzsq;                // =D*z*z  
Vector<Real> m_Dzsq_ex;             // =z*(D*z-D_ex*z_ex)
Vector<Real> m_zDFoverRT;           // =zDF/RT

// Bounds and constraints on concentrations of redox species and SOC
Real m_conc_redox_tot = 0.0;        // total concentration of redox active species (oxidized + reduced); constant
Real m_conc_ox_min = 0.0;           // minimum concentration of the oxidized species
Real m_conc_ox_max = 0.0;           // maximum concentration of the oxidized species
Real m_conc_red_min = 0.0;          // minimum concentration of the reduced species
Real m_conc_red_max = 0.0;          // maximum concentration of the reduced species
Real m_soc_min = 0.0;               // minimum SOC
Real m_soc_max = 1.0;               // maximum SOC


// Time scale for fluid flow
Real m_flow_in = 0.0;          // fluid flow rate [m^3/s] at the inlet; updated in AnalyzeFlow
Real m_flow_out = 0.0;         // fluid flow rate [m^3/s] at the outlet; updated in AnalyzeFlow
Real m_t_flow = 0.0;                // time for reactant to flow across the domain; updated in AnalyzeFlow
Real m_t_min = 0.0;                 // minimum time to run simulation (currently only applied in Nernst model)

// Electrochemical reaction description

// number of chemical species
int m_nspec = 0;                
// number of electrons exchanged in the redox reaction
int m_n_electrons = 0;
// number of protons generated in the redox reaction (it will become source prefactor for species 3)
int m_n_protons = 0;
// constant for (species0) anodic charge transfer coeff in Butler-Volmer equation
Real m_alpha_0 = 0.5;
// constant for (species1) cathodic charge transfer coeff in Butler-Volmer equation
Real m_alpha_1 = 0.5;
// is this the special case where alpha_0 == alpha_1?
bool m_same_alpha = true;
// initial concentration of protons in mM; 2000 corresponds to 1.0 M H2SO4
Real m_init_protons = -1.0;
bool m_fix_protons_mem = false;

// Rate constant [m/s]
Real m_k_0 = 0.0;    // previously defaulted to 7.2E-5 for AQDS; now set to 0.0 to force user to put in a real value
// Conductivity of solid [S/m]  estimated ~1000 for carbon fibers in literature
Real m_kappa_s = 1000.0;
// (Effective) conductivity of liquid (if negative will be calculated based on species concentration)
Real m_kappa_l = -1.0;  
// Should we solve for the potential on the electrode in the ButlerVolmer model?
bool m_solve_epotS = false;
// What is the applied potential on the electrode in the ButlerVolmer model?
Real m_V_electrode = NaN;
// What is the applied potential on the membrane in the ButlerVolmer model?
Real m_V_membrane = NaN;
// Option to include concentration-dependent ionic flux in epotL equation
bool m_calc_flux_epotL = true;
// Overpotential is clamped to at most m_overpot_max millivolts for numerical stability;
Real m_overpot_max = 0.100;
// Maximum consumption of reactant in any one time step; dynamically tuned
Real m_reactant_consumption_max = 1.0 / 256.0;
// Global limit during adaptive tuning of maximum consumption of reactant in any one time-step
Real m_reactant_consumption_max_limit = 1.0 / 64.0;
// Never auto-tune m_reactant_consumption_max below this threshold (e.g. due to stagnant steps)
static constexpr Real m_reactant_consumption_max_limit_lo = 1.0 / 1024.0;

// Set the method for (iteratively) updating boundary conditions at membrane
MembraneBCType m_membc_type = MembraneBCType::none;

// The location of the membrane
MembraneLocation m_mem_loc = MembraneLocation::ylo;
// Membrane properties
Real m_mem_Res=-1.0;        //  [Ohm] -> Res(istance)=res(istivity)*Lx  (assuming mem location=ylo)
Real m_mem_kappa=10.0;      //  conductivity [S/m]
Real m_mem_thick=100e-6;    //  membrance thickness [m]

// Standard potential in Nernst equation [V]
// Real m_E_0 = 0.0;

// Some chemistry constants known at compile time
// RT/F ~ 0.02568 [V]
// F/RT ~ 38.925 [1/V]
// F^2/RT~3.75*10^6

// Ideal gas constant = 8.3145 J/(mol*K) 
static constexpr Real m_R_const = 8.31446261815324;
// Faraday's constant [C/mol]
static constexpr Real m_F_const =  96485.3329;
// Temperature [K] (298.15 K = 25 C, room temperature)
static constexpr Real m_T_const = 298.15;
// F^2 / RT
static constexpr Real m_F2_over_RT_const = (m_F_const*m_F_const)/(m_R_const*m_T_const);

// These are non-const placeholders because we don't know the number of electrons 
// n_electrons until after we parse the input file

// nF (in coulombs / mole)
Real m_nF_const = 0.0;
// RT / nF (in volts V)
Real m_RT_over_nF_const = 0.0;
// nF / RT (in inverse volts 1/V)
Real m_nF_over_RT_const = 0.0;

// Parameters for special redox model
// Applied voltage for the special redox model
Real m_DVapp = 0.000;

// Type of convergence tolerance for state of charge in reaction sim; "abs_step" or "abs_rate".
ToleranceType m_tol_type_soc = ToleranceType::abs_rate;
// Type of convergence tolerance for potential in reaction sim; "abs_step" or "abs_rate".
ToleranceType m_tol_type_epot = ToleranceType::abs_rate;

// Steady state tolerance - state of charge
Real m_react_tol_soc = 1.0E-2;
// Steady state tolerance - electric potential
Real m_react_tol_epot = 1.0E-2;
// Steady state tolerance - relative flux difference (current vs. net flux)
Real m_react_tol_flux = 5.0E-2;
// Steady state tolerance - relative mass transport current difference (BV current vs. mass transport current)
// Only used in Nernst model
Real m_react_tol_mt = 5.0E-2;

// Write convergence process during simulations to two CSV files for fluid flow and reaction
std::fstream grid_info {std::fstream("profile/grid_info.csv", std::ios_base::out)};
std::fstream cvg_flow {std::fstream("profile/convergence_flow.csv", std::ios_base::app)};
std::fstream cvg_react {std::fstream("profile/convergence_react.csv", std::ios_base::app)};

// Time-stepping scheme
int m_react_timestepping_method = 1;

// How to handle nonlinearity
int m_react_nonlin_method = 1;

// For fixed-point convergence
Real m_fixpoint_conc_tol = 1.0E-9;      // fixed point tolerance - concentration
Real m_fixpoint_epot_tol = 1.0E-9;      // fixed point tolerance - potential
int m_fixpoint_max_iter = 100;          // maximum number of fixed point iterations

// Relaxation (within one fixed-point iteration)
Real m_fixpoint_omega = 1.0;
bool m_fixpoint_relax = false;

// Step counters for reaction simulation
// Number of reaction steps taken
int m_rstep = 0;
// Number of reaction steps taken when we started running the simulation (e.g. from earlier runs)
int m_start_rsteps = 0;
// Minimum number of reaction steps for early termination
int m_min_rsteps = 0;
// Maximum number of reaction steps for early termination
int m_max_rsteps = -1;
// Count number of failed Picard iterations
int m_fail_count_picard_iter = 0;
// Count number of stagnant reaction time steps
int m_stagnant_steps = 0;
// Maximum number of stagnant reaction steps before early termination
int m_stagnant_steps_max = 16000;

// Changes in state variables (velocity, concentration, epot) over a time step
// Used for keeping track of convergence and determining if a time step was stagnant

// Change in velocity over the last time step used for convergence testing; one entry per level
Vector<Real> m_change_vel;
// Smallest change in velocity over a time step seen so far; one entry per level, used to test stagnation condition
Vector<Real> m_change_min_vel;
// Change in concentration over the last time step used for convergence testing; one entry per level
Vector<Real> m_change_soc;
// Change in potential over the last time step used for convergence testing; one entry per level
Vector<Real> m_change_epot;
// Smallest change in concentration over a time step used for convergence testing; one entry per level, used to test stagnation condition
Vector<Real> m_change_min_soc;
// Smallest change in potential over a time step used for convergence testing; one entry per level, used to test stagnation condition
Vector<Real> m_change_min_epot;
// Full history of changes in concentration
Vector< Vector<Real>> m_change_hist_soc;

// The current in amps calculated from Butler-Volmer kinetics on the wire
Real m_current = 0.0;
// The maximum current in amps with full utilization, i.e.
// current_max = n_e F C0 Q. n_e is the number of electrons exchanged in the redox reaction, F is Faraday's constant,
// C0 is the concentration of the redox active species, and Q is volumetric flow rate
Real m_current_max = 0.0;
// The utilization factor of the electrode; dimensionless. utilization = current / current_max, where
Real m_utilization = 0.0;

// Flux equivalent current; take net charge flowing out minus net charge flowing in, convert to amps
Real m_curr_flux = 0.0;
// Current vs. flux difference; this is (current - curr_flux); SIGNED quantity in amps
Real m_flux_diff = 0.0;
// Relative difference between current and net flux over a time step; dimensionless
Real m_flux_diff_rel = 0.0;
// Smallest relative flux difference over a time step; dimensionless
Real m_min_flux_diff_rel = std::numeric_limits<Real>::max();
// Full history of relative flux difference; dimensionless
Vector<Real> m_hist_flux_diff_rel;

// The current in amps caluclated from mass transport on the wire; only defined in the Nernst model
Real m_current_mt = 0.0;
// Relative difference between the current calculated two ways in the Nernst model: BV kinetics vs. mass transport
Real m_curr_diff_rel = 0.0;
// Smallest relative mass transport difference over a time step; dimensionless
Real m_min_curr_diff_rel = std::numeric_limits<Real>::max();

// Relaxation parameter for updating overpot in the Nernst model
Real m_nernst_omega {0.0};
// Does the Nernst model use explicit diffusion for time stepping?
bool m_nernst_explicit_diffusion {false};

// For plotting/checkpoint during reaction
// Interval for generating an AMReX plot
int m_react_plot_int = -1;
// Interval for generating an AMReX checkpoint
int m_react_check_int = -1;
// Prefix used to name AMReX plots of the reaction, e.g. reactplt0012345
string m_react_plot_file{"reactplt"};
// Prefix used to name AMReX checkpoint of the reaction, e.g. reactchk0012345
string m_react_check_file{"reactchk"};
// The interval between reaction checkpoints and plots (in seconds); negative means don't run time based checkpoints
int m_react_int_sec = -1;
// The last time we plotted or checkpointed the reaction (in seconds)
Real m_react_status_time = 0.0;

#if (DEBUG_RFB)
// Number of MPI processors for debug printing
static constexpr int debug_log_size {16};

// Filestream handles for debug printing
mutable vector<ofstream> fs_debug; 

// The depth of the call stack for debugging
mutable int stack_depth {0};
#endif

// Should the reaction simulation reinitialize the concentration of all the species?
bool m_react_reinit_all_conc = true;
// Should the reaction simulation reinitialize the concentration of some of the species?
bool m_react_reinit_some_conc = false;
// What is the index of the first species whose concentration is to be reinitialized?
int m_react_reinit_conc_from = 2;
// The initial time of the reaction simulation; one entry per species
Vector<Real> m_react_init_t;
// Is the electric potential in the liquid reinitialized?
bool m_react_reinit_epotL = true;
// Is the electric potential in the solid reinitialized?
bool m_react_reinit_epotS = true;
// Shoule we zero out the reaction step counter when we restart the simulation?
bool m_react_zero_step_counter = false;
// Manually set the reaction step counter to this value when restarting a simulation
int m_react_init_step_counter = -1;

// Level data and embedded boundary factory    
// #include <rincflo_levels.H>
Vector<std::unique_ptr<LevelData> > m_leveldata;
Vector<std::unique_ptr<FabFactory<FArrayBox> > > m_factory;

// GPU arrays for boundary condiditions
GpuArray<BC, SpaceDim*2> m_bc_type;
GpuArray<Real, SpaceDim*2> m_bc_pressure;
GpuArray<Real, SpaceDim*2> m_bc_density;
GpuArray<GpuArray<Real, SpaceDim>, SpaceDim*2> m_bc_velocity;
GpuArray<Vector<Real> , SpaceDim*2> m_bc_conc;
GpuArray<Real, SpaceDim*2> m_bc_epotL;
GpuArray<Real, SpaceDim*2> m_bc_epotS;

// Vector cannot be used on gpu, so ...
GpuArray<Real const*, SpaceDim*2> m_bc_conc_d;
DeviceVector<Real> m_bc_conc_raii;
Vector<BCRec> m_bcrec_velocity;
DeviceVector<BCRec> m_bcrec_velocity_d;
Vector<BCRec> m_bcrec_density;
DeviceVector<BCRec> m_bcrec_density_d;
Vector<BCRec> m_bcrec_conc;
DeviceVector<BCRec> m_bcrec_conc_d;
Vector<BCRec> m_bcrec_force;
DeviceVector<BCRec> m_bcrec_force_d;
Vector<BCRec> m_bcrec_epotL;
DeviceVector<BCRec> m_bcrec_epotL_d;
Vector<BCRec> m_bcrec_epotS;
DeviceVector<BCRec> m_bcrec_epotS_d;
// Conserved quantities    
Vector<int> m_iconserv_velocity;
DeviceVector<int> m_iconserv_velocity_d;
Vector<int> m_iconserv_density;
DeviceVector<int> m_iconserv_density_d;
Vector<int> m_iconserv_conc;
DeviceVector<int> m_iconserv_conc_d;

// Operators
std::unique_ptr<DiffusionTensorOp> m_diffusion_tensor_op;
std::unique_ptr<DiffusionScalarOp> m_diffusion_scalar_op;
std::unique_ptr<ReactionScalarOp> m_reaction_scalar_op;
std::unique_ptr<ReactionEliquidOp> m_reaction_eliquid_op;
std::unique_ptr<ReactionEsolidOp> m_reaction_esolid_op;
