#ifndef AMREX_EBMULTIFABUTILINV_H_
#define AMREX_EBMULTIFABUTILINV_H_
#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_BCRec.H>

namespace amrex
{
  void EB_set_covered (MultiFab& mf,                                               Real   val, bool invert_eb);    
  void EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow,              Real   val, bool invert_eb);
  void EB_set_covered (MultiFab& mf, int icomp, int ncomp,            const Vector<Real>& vals, bool invert_eb);
  void EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow, const Vector<Real>& vals, bool invert_eb);

  
  void EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, const MultiFab& vol_fine,
			const MultiFab& vfrac_fine, int scomp, int ncomp, const IntVect& ratio, bool invert_eb);
  void EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp,
			int ratio, bool invert_eb);
  void EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp,
			const IntVect& ratio, bool invert_eb);

  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
			      const Array<MultiFab*,AMREX_SPACEDIM>& crse,
			      int ratio, int ngcrse, bool invert_eb);
  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
			      const Array<MultiFab*,AMREX_SPACEDIM>& crse,
			      const IntVect& ratio, int ngcrse, bool invert_eb);
  
  //  This version takes periodicity into account.
  void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
			      const Array<      MultiFab*,AMREX_SPACEDIM>& crse,
			      const IntVect& ratio, const Geometry& crse_geom, bool invert_eb);


  void EB_average_face_to_cellcenter (MultiFab& ccmf, int dcomp,
				      const Array<MultiFab const*,AMREX_SPACEDIM>& fmf, bool invert_eb);

}

#endif
