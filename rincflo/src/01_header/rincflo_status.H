// rincflo_status.H
// Fragment of Rincflo.H
// Methods in Rincflo for getting the status of the simulation, usually for control flow.
#pragma once
#include <rincflo_macros.H>

// *****************************************************************************************************************
// Quick functions to tell the caller something about the simulation setup, usually for flow control

public:

// Is this a restart from an AMReX checkpoint on disk?
inline bool is_restart() const
    {return (!m_restart_file.empty());}

// Should Rincflo simulate the fluid flow?
inline bool do_flow() const
    {return (m_do_flow);}

// Should Rincflo simulate the chemical reaction?
inline bool do_reaction() const
    {return (m_do_reaction);}

// Does this simulation have an embedded boundary?
inline bool has_eb() const
#if (AMREX_USE_EB)
{return (!EBFactory(0).isAllRegular());}
#else
{return false;}
#endif

// Should Rincflo write to numpy?
inline bool do_WriteNumpy() const
    {return (m_write_numpy);}

// Should Rincflo read from numpy?
inline bool do_ReadNumpy() const
    {return (m_read_numpy);}

// Should Rincflo perform analysis?
inline bool do_Analysis() const
    {return (m_do_analysis);}

// Is Rincflo running a reaction model at all? If so, the LevelData includes reaction data
inline bool is_Reaction() const
    {return (m_reaction_model != ReactionModel::none);}

// Should Rincflo advect species concentrations?
inline bool do_AdvectConc() const
    {return (m_advect_conc);}

// Should Rincflo consider the electromigration term?
inline bool do_Electromigration() const
    {return (m_electromigration);}

// Is the active simulation type flow or reaction?
inline bool is_ActiveSimReact() const
    {return m_do_reaction || (m_rstep > 0);}

// How many steps have been run? Choose m_step or m_r_step based on the model type
inline int ActiveStepNum() const 
    {return is_ActiveSimReact() ? m_rstep : m_nstep;}

// Does this model calculate potential (e.g. BV) or is it assumed to match DVapp (e.g. special)?
inline bool is_PotComputed() const
    {return (m_reaction_model == Rincflo::ReactionModel::ButlerVolmer);}

// Are we using implicit diffusion in the reaction simulation?
inline bool is_ImplicitDiffusion() const 
    {return (m_diff_type==DiffusionType::Implicit);}

// Does the simulation need the divergence of tau for forcing temrs?
inline bool need_DivTau () const 
    {return ( m_godunov_include_diff_in_forcing || DiffusionType::Implicit != m_diff_type );}

// Do we need the Laplacian of concentrations?
inline bool need_Laplacian () const 
    {return (m_react_timestepping_method==0 || m_calc_flux_epotL );}

// Do we need to assemble a term with the flux on the right hand side?
inline bool need_FluxRhs () const 
    {return (m_reaction_model == ReactionModel::ButlerVolmer && m_calc_flux_epotL);}

// Do we need the electric conductivity kappa?
inline bool need_Kappa () const 
{
    //$ actually needed only if kappa not constant..
    return (m_reaction_model == ReactionModel::ButlerVolmer); 
}

// Do we need to update the membrane boundary conditions?
inline bool need_MembreaneBcUpdate () const 
    {return (m_membc_type != MembraneBCType::none && m_reaction_model == ReactionModel::ButlerVolmer);}

// Are we using a tensor solve in the flow simulation?
inline bool useTensorSolve () const 
    {return use_tensor_solve;}

// Are we using a tensor correction in the flow simulation?
inline bool useTensorCorrection () const 
    {return use_tensor_correction;}

public:
