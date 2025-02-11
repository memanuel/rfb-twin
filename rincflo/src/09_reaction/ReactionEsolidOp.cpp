#include <rincflo.H>
#include <ReactionEsolidOp.H>

ReactionEsolidOp::ReactionEsolidOp (Rincflo* a_rincflo)
  : m_rincflo(a_rincflo)
{
  readParameters();
  
  LPInfo info_solve;
  info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
  if(m_bottom_solver_type == "hypre") {
    info_solve.setMaxCoarseningLevel(0);
  }

  int finest_level = m_rincflo->finestLevel();
  Vector<EBFArrayBoxFactory const*> ebfact;
  for (int lev = 0; lev <= finest_level; ++lev) {
    ebfact.push_back(&(m_rincflo->EBFactory(lev)));
  }
    
  m_eb_solve_op.reset(new MLEBABecLapInv(m_rincflo->Geom(0,finest_level),
					 m_rincflo->boxArray(0,finest_level),
					 m_rincflo->DistributionMap(0,finest_level),
					 info_solve, ebfact));
  m_eb_solve_op->setInvertEB(m_invert_eb);
  m_eb_solve_op->setMaxOrder(m_mg_maxorder);
  m_eb_solve_op->setDomainBC(m_rincflo->get_react_esolid_bc(Orientation::low),
			     m_rincflo->get_react_esolid_bc(Orientation::high));

  m_mlmg.reset(new MLMGInv(*m_eb_solve_op));

  setOptions();  
  
}


void ReactionEsolidOp::readParameters ()
{
  ParmParse pp("reaction_epot");
  m_verbose = m_rincflo->m_verbose;
  pp.query("verbose", m_verbose);
  pp.query("mg_verbose", m_mg_verbose);
  pp.query("mg_bottom_verbose", m_mg_bottom_verbose);
  pp.query("mg_max_iter", m_mg_max_iter);
  pp.query("mg_bottom_maxiter", m_mg_bottom_maxiter);
  pp.query("mg_max_fmg_iter", m_mg_max_fmg_iter);
  pp.query("mg_max_coarsening_level", m_mg_max_coarsening_level);
  pp.query("mg_maxorder", m_mg_maxorder);
  pp.query("mg_rtol", m_mg_rtol);
  pp.query("mg_atol", m_mg_atol);
  pp.query("bottom_solver_type", m_bottom_solver_type);
  pp.query("num_pre_smooth", m_num_pre_smooth);
  pp.query("num_post_smooth", m_num_post_smooth);
  pp.query("num_final_smooth", m_num_final_smooth);
  pp.query("num_bottom_smooth", m_num_bottom_smooth);
  pp.query("bottom_rtol", m_bottom_rtol);
}

void ReactionEsolidOp::setOptions ()
{
  m_mlmg->setInvertEB(m_invert_eb);	
  m_mlmg->setHypreOptionsNamespace("reaction_epot.hypre");
    
  // The default bottom solver is BiCG
  if (m_bottom_solver_type == "smoother")
    {
      m_mlmg->setBottomSolver(MLMG::BottomSolver::smoother);
    }
  else if (m_bottom_solver_type == "hypre")
    {
      m_mlmg->setBottomSolver(MLMG::BottomSolver::hypre);
    }
  
  // Maximum iterations for MultiGrid / ConjugateGradients
  m_mlmg->setMaxIter(m_mg_max_iter);
  m_mlmg->setMaxFmgIter(m_mg_max_fmg_iter);
  m_mlmg->setBottomMaxIter(m_mg_bottom_maxiter);
        
  // Verbosity for MultiGrid / Bottom solver
  m_mlmg->setVerbose(m_mg_verbose);
  m_mlmg->setBottomVerbose(m_mg_bottom_verbose);

  m_mlmg->setPreSmooth(m_num_pre_smooth);
  m_mlmg->setPostSmooth(m_num_post_smooth);
  m_mlmg->setFinalSmooth(m_num_final_smooth);
  m_mlmg->setBottomSmooth(m_num_bottom_smooth);

  m_mlmg->setBottomTolerance(m_bottom_rtol);
}

// 
//      alpha A - beta div ( B grad )   <--->   0 - div ( kappa grad )  u = RHS
// So the constants and variable coefficients are:
//      alpha: (scalar const)             0
//      beta:  (scalar const)             1
//      A:     (scalar field/const)       0
//      B:     (scalar field/const)       kappa
//   RHS= source  (arising from time-stepping and Picard iteration) ; stored in epotS


void ReactionEsolidOp::react_epot_solid (Vector<MultiFab*> const& epotS,
					 Vector<MultiFab*> const& rhs_arr, 
					 Vector<MultiFab const*> const& kappa)
{
  const int finest_level = m_rincflo->finestLevel();

  // vector for solution (and inhomog b.c., if any)
  Vector<MultiFab> phi;

  if(m_needs_update) //update operator
    {
      m_eb_solve_op->setScalars(0.0, 1.0);
      
      for (int lev = 0; lev <= finest_level; ++lev) {
	
	m_eb_solve_op->setACoeffs(lev, 0.0);

	if(m_rincflo->m_kappa_s > 0.0)
	  {
	    m_eb_solve_op->setBCoeffs(lev, m_rincflo->m_kappa_s);
	  }
	else
	  {
	    Array<MultiFab,SpaceDim> b = m_rincflo->AverageEpotScalToFaces(lev, 0, *kappa[lev]);	  
	    m_eb_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCenter);
	  }
	
      }

      // if kappa_s > 0 (i.e. constant) nothing changes in between time-steps...
      if(m_rincflo->m_kappa_s > 0.0)
	m_needs_update = false;
      
    }

  // prepare for solve
  if(m_verbose > 2)  
    {Print() << "reacting epot solid\n";}

  // setup rhs
  Vector<MultiFab> rhs;

  for (int lev = 0; lev <= finest_level; ++lev) {
    rhs.emplace_back(*rhs_arr[lev], amrex::make_alias, 0, 1);
    phi.emplace_back(*epotS[lev], amrex::make_alias, 0, 1);
    m_eb_solve_op->setLevelBC(lev, &phi[lev]);
  }
  
  //actual solve
  m_mlmg->solve(GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);	

}
 


