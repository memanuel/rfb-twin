#ifndef AMREX_EB_MULTIFAB_UTILINV_2D_C_H_
#define AMREX_EB_MULTIFAB_UTILINV_2D_C_H_
#include <AMReX_Config.H>
namespace amrex {

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_set_covered_nodes (int i, int j, int k, int n, int icomp, Array4<Real> const& d,
			     Array4<EBCellFlag const> const& f, Real v, bool invert_eb)
  {
    if(!invert_eb) {
      if (f(i-1,j-1,k  ).isCovered() and f(i  ,j-1,k  ).isCovered() and
	  f(i-1,j  ,k  ).isCovered() and f(i  ,j  ,k  ).isCovered())
	{
	  d(i,j,k,n+icomp) = v;
	}
    }
    else { //invert_eb
      if (f(i-1,j-1,k  ).isRegular() and f(i  ,j-1,k  ).isRegular() and
	  f(i-1,j  ,k  ).isRegular() and f(i  ,j  ,k  ).isRegular())
	{
	  d(i,j,k,n+icomp) = v;
	}
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_set_covered_nodes (int i, int j, int k, int n, int icomp, Array4<Real> const& d,
			     Array4<EBCellFlag const> const& f, Real const * AMREX_RESTRICT v, bool invert_eb)
  {
    if(!invert_eb) {
      if (f(i-1,j-1,k  ).isCovered() and f(i  ,j-1,k  ).isCovered() and
	  f(i-1,j  ,k  ).isCovered() and f(i  ,j  ,k  ).isCovered())
	{
	  d(i,j,k,n+icomp) = v[n];
	}
    }
    else { //invert_eb
      if (f(i-1,j-1,k  ).isRegular() and f(i  ,j-1,k  ).isRegular() and
	  f(i-1,j  ,k  ).isRegular() and f(i  ,j  ,k  ).isRegular())
	{
	  d(i,j,k,n+icomp) = v[n];
	}
    }
  }

  
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_avgdown_with_vol (int i, int j, int k,
			    Array4<Real const> const& fine, int fcomp,
			    Array4<Real> const& crse, int ccomp,
			    Array4<Real const> const& fv, Array4<Real const> const& vfrc,
			    Dim3 const& ratio, int ncomp, bool invert_eb)
  {
    for (int n = 0; n < ncomp; ++n) {
      Real c = 0.0;
      Real cv = 0.0;
      constexpr int kk=0;
      for (int jj = j*ratio.y; jj < (j+1)*ratio.y; ++jj) {
	for (int ii = i*ratio.x; ii < (i+1)*ratio.x; ++ii) {
	  Real tmp;
	  if(!invert_eb)
	    tmp = fv(ii,jj,kk)*vfrc(ii,jj,kk);
	  else //invert_eb
	    tmp = fv(ii,jj,kk)*(1.0-vfrc(ii,jj,kk));
	  c += fine(ii,jj,kk,n+fcomp)*tmp;
	  cv += tmp;
	}}
      if (cv > 1.e-30) {
	crse(i,j,k,n+ccomp) = c/cv;
      } else {
	crse(i,j,k,n+ccomp) = fine(i*ratio.x,j*ratio.y,kk,n+fcomp);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_avgdown (int i, int j, int k,
		   Array4<Real const> const& fine, int fcomp,
		   Array4<Real> const& crse, int ccomp,
		   Array4<Real const> const& vfrc,
		   Dim3 const& ratio, int ncomp, bool invert_eb)
  {
    for (int n = 0; n < ncomp; ++n) {
      Real c = 0.0;
      Real cv = 0.0;
      constexpr int kk=0;
      for (int jj = j*ratio.y; jj < (j+1)*ratio.y; ++jj) {
	for (int ii = i*ratio.x; ii < (i+1)*ratio.x; ++ii) {
	  Real tmp;
	  if(!invert_eb)
	    tmp = vfrc(ii,jj,kk);
	  else //invert_eb
	    tmp = (1.0-vfrc(ii,jj,kk));
	  c += fine(ii,jj,kk,n+fcomp)*tmp;
	  cv += tmp;
	}}
      if (cv > 1.e-30) {
	crse(i,j,k,n+ccomp) = c/cv;
      } else {
	crse(i,j,k,n+ccomp) = fine(i*ratio.x,j*ratio.y,kk,n+fcomp);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_avgdown_face_x (int i, int j, int k,
			  Array4<Real const> const& fine, int fcomp,
			  Array4<Real> const& crse, int ccomp,
			  Array4<Real const> const& area,
			  Dim3 const& ratio, int ncomp, bool invert_eb)
  {
    int ii = i*ratio.x;
    for (int n = 0; n < ncomp; ++n) {
      Real c = 0.0;
      Real a = 0.0;
      constexpr int kk=0;
        for (int jj = j*ratio.y; jj < (j+1)*ratio.y; ++jj) {
	  Real tmp;
	  if(!invert_eb)
	    tmp = area(ii,jj,kk);
	  else //invert_eb
	    tmp = (1.0-area(ii,jj,kk));
	  c += tmp*fine(ii,jj,kk,n+fcomp);
	  a += tmp;
        }
      if (a > 1.e-30) {
	crse(i,j,k,n+ccomp) = c/a;
      } else {
	crse(i,j,k,n+ccomp) = fine(ii,j*ratio.y,kk,n+fcomp);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_avgdown_face_y (int i, int j, int k,
			  Array4<Real const> const& fine, int fcomp,
			  Array4<Real> const& crse, int ccomp,
			  Array4<Real const> const& area,
			  Dim3 const& ratio, int ncomp, bool invert_eb)
  {
    int jj = j*ratio.y;
    for (int n = 0; n < ncomp; ++n) {
      Real c = 0.0;
      Real a = 0.0;
      constexpr int kk=0;
        for (int ii = i*ratio.x; ii < (i+1)*ratio.x; ++ii) {
	  Real tmp;
	  if(!invert_eb)
	    tmp = area(ii,jj,kk);
	  else //invert_eb
	    tmp = (1.0-area(ii,jj,kk));
	  c += tmp*fine(ii,jj,kk,n+fcomp);
	  a += tmp;
        }
      if (a > 1.e-30) {
	crse(i,j,k,n+ccomp) = c/a;
      } else {
	crse(i,j,k,n+ccomp) = fine(i*ratio.x,jj,kk,n+fcomp);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  void eb_avg_fc_to_cc (int i, int j, int k, int n, Array4<Real> const& cc,
			Array4<Real const> const& fx, Array4<Real const> const& fy,
			Array4<Real const> const& ax, Array4<Real const> const& ay,
			Array4<EBCellFlag const> const& flag, bool invert_eb)
  {
    if ( (flag(i,j,k).isCovered() && !invert_eb) || (flag(i,j,k).isRegular() && invert_eb) ) {
      cc(i,j,k,n+0) = 0.0;
      cc(i,j,k,n+1) = 0.0;
    } else {
      if (ax(i,j,k) == 0.0) {
	cc(i,j,k,n+0) = fx(i+1,j,k);
      } else if (ax(i+1,j,k) == 0.0) {
	cc(i,j,k,n+0) = fx(i,j,k);
      } else {
	cc(i,j,k,n+0) = 0.5 * (fx(i,j,k) + fx(i+1,j,k));
      }

      if (ay(i,j,k) == 0.0) {
	cc(i,j,k,n+1) = fy(i,j+1,k);
      } else if (ay(i,j+1,k) == 0.0) {
	cc(i,j,k,n+1) = fy(i,j,k);
      } else {
	cc(i,j,k,n+1) = 0.5 * (fy(i,j,k) + fy(i,j+1,k));
      }

    }
  }

}
#endif
