// rincflo_get.H
// Fragment of Rincflo.H
// All the get methods in Rincflo
#pragma once
#include <rincflo_macros.H>

private:

// Get operators    
DiffusionTensorOp* get_diffusion_tensor_op ();
DiffusionScalarOp* get_diffusion_scalar_op ();
ReactionScalarOp* get_reaction_scalar_op ();
ReactionEliquidOp* get_reaction_eliquid_op ();
ReactionEsolidOp* get_reaction_esolid_op ();  

// Get multifabs in level data
Vector<MultiFab*> get_velocity_new () noexcept;
Vector<MultiFab*> get_velocity_old () noexcept;

Vector<MultiFab*> get_density_new () noexcept;
Vector<MultiFab*> get_density_old () noexcept;

Vector<MultiFab*> get_conc_new () noexcept;
Vector<MultiFab*> get_conc_old () noexcept;
Vector<MultiFab*> get_conc_old_nernst () noexcept;

Vector<MultiFab*> get_mac_phi () noexcept;

Vector<MultiFab*> get_vel_forces () noexcept;
Vector<MultiFab*> get_tra_forces () noexcept;

Vector<MultiFab*> get_conv_dconc_dt_new () noexcept;
Vector<MultiFab*> get_conv_dconc_dt_old () noexcept;

Vector<MultiFab*> get_conc_adv_new () noexcept;

Vector<MultiFab*> get_conv_dv_dt_new () noexcept;
Vector<MultiFab*> get_conv_dv_dt_old () noexcept;

Vector<MultiFab*> get_conv_drho_dt_new () noexcept;
Vector<MultiFab*> get_conv_drho_dt_old () noexcept;

Vector<MultiFab*> get_divtau_new () noexcept;
Vector<MultiFab*> get_divtau_old () noexcept;

Vector<MultiFab*> get_laps_new () noexcept;
Vector<MultiFab*> get_laps_old () noexcept;

Vector<MultiFab*> get_epotL_new () noexcept;
Vector<MultiFab*> get_epotL_old () noexcept;

Vector<MultiFab*> get_epotS_new () noexcept;
Vector<MultiFab*> get_epotS_old () noexcept;

Vector<MultiFab*> get_pot_diff () noexcept;
Vector<MultiFab*> get_overpot () noexcept;

Vector<MultiFab*> get_soc_new () noexcept;
Vector<MultiFab*> get_soc_old () noexcept;

Vector<MultiFab*> get_source () noexcept;
Vector<MultiFab*> get_source_mol () noexcept;
Vector<MultiFab*> get_current () noexcept;

Vector<MultiFab*> get_kappa () noexcept;
Vector<MultiFab*> get_flxrhs () noexcept;
Vector<MultiFab*> get_divcgpot () noexcept;

// Get constant multifabs in level data
Vector<MultiFab const*> get_velocity_new_const () const noexcept;
Vector<MultiFab const*> get_velocity_old_const () const noexcept;

Vector<MultiFab const*> get_density_new_const () const noexcept;
Vector<MultiFab const*> get_density_old_const () const noexcept;

Vector<MultiFab const*> get_conc_new_const () const noexcept;
Vector<MultiFab const*> get_conc_old_const () const noexcept;
Vector<MultiFab const*> get_conc_old_nernst_const () const noexcept;

Vector<MultiFab const*> get_vel_forces_const () const noexcept;
Vector<MultiFab const*> get_tra_forces_const () const noexcept;

Vector<MultiFab const*> get_conc_adv_new_const () const noexcept;
Vector<MultiFab const*> get_conv_dconc_dt_new_const () const noexcept;

Vector<MultiFab const*> get_epotL_new_const () const noexcept;
Vector<MultiFab const*> get_epotL_old_const () const noexcept;

Vector<MultiFab const*> get_epotS_new_const () const noexcept;
Vector<MultiFab const*> get_epotS_old_const () const noexcept;

Vector<MultiFab const*> get_pot_diff_const () const noexcept;
Vector<MultiFab const*> get_overpot_const () const noexcept;

Vector<MultiFab const*> get_soc_new_const () const noexcept;

Vector<MultiFab const*> get_source_mol_const () const noexcept;

Vector<MultiFab const*> get_source_const () const noexcept;

Vector<MultiFab const*> get_laps_new_const () const noexcept;
Vector<MultiFab const*> get_laps_old_const () const noexcept;

Vector<MultiFab const*> get_kappa_const () const noexcept;
Vector<MultiFab const*> get_flxrhs_const () const noexcept;

// Get boundary condition types
Array<LinOpBCType,SpaceDim>
get_projection_bc (Orientation::Side side) const noexcept;

Array<LinOpBCType,SpaceDim>
get_DiffuseVelocity_bc (Orientation::Side side, int comp) const noexcept;

Vector<Array<LinOpBCType,SpaceDim> >
get_diffuse_tensor_bc (Orientation::Side side) const noexcept;

Array<LinOpBCType,SpaceDim>
get_DiffuseScalar_bc (Orientation::Side side) const noexcept;

Array<LinOpBCType,SpaceDim>
get_react_scalar_bc (Orientation::Side side, int comp) const noexcept;

Array<LinOpBCType,SpaceDim>
get_react_eliquid_bc (Orientation::Side side) const noexcept;

Array<LinOpBCType,SpaceDim>
get_react_esolid_bc (Orientation::Side side) const noexcept;

// Get conserved quantities
Vector<int> const& get_velocity_iconserv () const noexcept { return m_iconserv_velocity; }
Vector<int> const& get_density_iconserv () const noexcept { return m_iconserv_density; }
Vector<int> const& get_conc_iconserv () const noexcept { return m_iconserv_conc; }

// Get device pointers for conserved quantities
int const* get_velocity_iconserv_device_ptr () const noexcept 
    {return m_iconserv_velocity_d.data(); }
int const* get_density_iconserv_device_ptr () const noexcept 
    {return m_iconserv_density_d.data(); }
int const* get_conc_iconserv_device_ptr () const noexcept 
    {return m_iconserv_conc_d.data(); }

// Get boundary conditions
Vector<BCRec> const& get_velocity_bcrec () const noexcept
    {return m_bcrec_velocity;}
Vector<BCRec> const& get_density_bcrec () const noexcept
    {return m_bcrec_density;}
Vector<BCRec> const& get_conc_bcrec () const noexcept
    {return m_bcrec_conc;}
Vector<BCRec> const& get_force_bcrec () const noexcept 
    {return m_bcrec_force;}
Vector<BCRec> const& get_epotL_bcrec () const noexcept 
    {return m_bcrec_epotL;}
Vector<BCRec> const& get_epotS_bcrec () const noexcept 
    {return m_bcrec_epotS;}

// Get device pointers for boundary conditions
BCRec const* get_velocity_bcrec_device_ptr () const noexcept 
    {return m_bcrec_velocity_d.data();}
BCRec const* get_density_bcrec_device_ptr () const noexcept 
    {return m_bcrec_density_d.data();}
BCRec const* get_conc_bcrec_device_ptr () const noexcept 
    {return m_bcrec_conc_d.data();}
BCRec const* get_force_bcrec_device_ptr () const noexcept 
    {return m_bcrec_force_d.data();}
BCRec const* get_epotL_bcrec_device_ptr () const noexcept 
    {return m_bcrec_epotL_d.data();}
BCRec const* get_epotS_bcrec_device_ptr () const noexcept 
    {return m_bcrec_epotS_d.data();}
