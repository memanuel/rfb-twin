#pragma once
#include <rincflo_macros.H>

#include <AMReX_MLMG.H>

#if (AMREX_USE_EB)
#include <AMReX_MLEBABecLap.H>
#endif
#include <AMReX_MLABecLaplacian.H>

class Rincflo;

class DiffusionScalarOp
{
public:
  DiffusionScalarOp (Rincflo* a_rincflo);

  void DiffuseScalar (amrex::Vector<amrex::MultiFab*> const& phi,
		       amrex::Vector<amrex::MultiFab*> const& density,
		       amrex::Vector<amrex::MultiFab const*> const& eta,
		       amrex::Real dt);

  void diffuse_vel_components (
			       amrex::Vector<amrex::MultiFab*> const& phi,
			       amrex::Vector<amrex::MultiFab*> const& density,
			       amrex::Vector<amrex::MultiFab const*> const& eta,
			       amrex::Real dt);

  void ComputeLaplacian (amrex::Vector<amrex::MultiFab*> const& laps,
		     amrex::Vector<amrex::MultiFab const*> const& phi,
		     amrex::Vector<amrex::MultiFab const*> const& density,
		     amrex::Vector<amrex::MultiFab const*> const& eta);
  
  void ComputeDivTau (amrex::Vector<amrex::MultiFab*> const& laps,
		       amrex::Vector<amrex::MultiFab const*> const& phi,
		       amrex::Vector<amrex::MultiFab const*> const& density,
		       amrex::Vector<amrex::MultiFab const*> const& eta);

private:

  void readParameters ();

  Rincflo* m_rincflo;

#if (AMREX_USE_EB)
  std::unique_ptr<amrex::MLEBABecLap> m_eb_scal_solve_op;
  std::unique_ptr<amrex::MLEBABecLap> m_eb_scal_apply_op;
  std::unique_ptr<amrex::MLEBABecLap> m_eb_vel_solve_op;
  std::unique_ptr<amrex::MLEBABecLap> m_eb_vel_apply_op;
#endif
  std::unique_ptr<amrex::MLABecLaplacian> m_reg_scal_solve_op;
  std::unique_ptr<amrex::MLABecLaplacian> m_reg_scal_apply_op;
  std::unique_ptr<amrex::MLABecLaplacian> m_reg_vel_solve_op;
  std::unique_ptr<amrex::MLABecLaplacian> m_reg_vel_apply_op;

  // DiffusionOp verbosity
  int m_verbose = 0;

  // Options to control MLMG behavior
  int m_mg_verbose = 0;
  int m_mg_bottom_verbose = 0;
  int m_mg_max_iter = 100;
  int m_mg_bottom_maxiter = 100;
  int m_mg_max_fmg_iter = 0;
  int m_mg_max_coarsening_level = 100;
  int m_mg_maxorder = 2;
  int m_num_pre_smooth = 2;
  int m_num_post_smooth = 2;
  int m_num_final_smooth = 8;
  int m_num_bottom_smooth = 0;

  amrex::Real m_mg_rtol = 1.0e-12;
  amrex::Real m_mg_atol = 1.0e-14;   

  std::string m_bottom_solver_type = "bicgstab";
  
};
