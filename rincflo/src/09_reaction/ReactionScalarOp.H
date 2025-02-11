#pragma once
#include <rincflo_macros.H>

#include <AMReX_MLMG.H>

#if (AMREX_USE_EB)
#include <AMReX_MLEBABecLap.H>
#endif
#include <AMReX_MLABecLaplacian.H>

class Rincflo;

class ReactionScalarOp
{
public:
  ReactionScalarOp (Rincflo* a_rincflo, int a_ncomp);

  void react_scalar (amrex::Vector<amrex::MultiFab*> const& phi,
		     amrex::Vector<amrex::MultiFab*> const& rhs_arr,
		     amrex::Vector<amrex::MultiFab const*> const& bcoeff,
		     amrex::Vector<amrex::MultiFab const*> const& acoeff,
		     amrex::Real dt,
		     int mode);
  
  //laplacian scalar
  void ComputeLaplacian (amrex::Vector<amrex::MultiFab*> const& laps,
		     amrex::Vector<amrex::MultiFab const*> const& phi);
  
private:

  void readParameters ();
  void setOptions ();

  Rincflo* m_rincflo;

  amrex::Vector<std::unique_ptr<amrex::MLMG>> m_mlmg;  //one for each component
#if (AMREX_USE_EB)
  amrex::Vector<std::unique_ptr<amrex::MLEBABecLap>> m_eb_solve_op;  //one for each component
  std::unique_ptr<amrex::MLEBABecLap> m_eb_scal_apply_op;
#endif

  bool m_needs_update = true;  //for solve_op
  int m_ncomp;
  
  // ReactionOp verbosity
  int m_verbose = 0;

  // Options to control MLMG behavior
  int m_mg_verbose = 0; 
  int m_mg_bottom_verbose = 0;
  int m_mg_max_iter = 200;
  int m_mg_bottom_maxiter = 200;
  int m_mg_max_fmg_iter = 0;
  int m_mg_max_coarsening_level = 100;
  int m_mg_maxorder = 2;
  int m_num_pre_smooth = 2;
  int m_num_post_smooth = 2;
  int m_num_final_smooth = 8;
  int m_num_bottom_smooth = 0;

  amrex::Real m_mg_rtol = 1.0e-12;
  amrex::Real m_mg_atol = -1.0;
  amrex::Real m_bottom_rtol = 1.0e-12;

  std::string m_bottom_solver_type = "bicgstab";
};
