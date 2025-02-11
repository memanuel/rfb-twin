#include <rincflo.H>

void Rincflo::ProblemSetInflowVelocity (int grid_id, Orientation ori, Box const& bx,
					Array4<Real> const& vel, int lev, Real time)
{
  if (31 == m_probtype)
    {
      Real dyinv = 1.0 / Geom(lev).Domain().length(1);
      Real u = m_ic_u;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real y = (j+0.5)*dyinv;
			   vel(i,j,k,0) = 6. * u * y * (1.-y);
			 };
      ParallelFor(bx, func);
    }
  else if (311 == m_probtype)
    {
      Real dzinv = 1.0 / Geom(lev).Domain().length(2);
      Real u = m_ic_u;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real z = (k+0.5)*dzinv;
			   vel(i,j,k,0) = 6. * u * z * (1.-z);
			 };
      ParallelFor(bx, func);
    }
  else if (41 == m_probtype)
    {
      Real dzinv = 1.0 / Geom(lev).Domain().length(2);
      Real u = m_ic_u;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real z = (k+0.5)*dzinv;
			   vel(i,j,k,0) = 0.5*z;
			 };
      ParallelFor(bx, func);
    }
  else if (32 == m_probtype)
    {
      Real dzinv = 1.0 / Geom(lev).Domain().length(2);
      Real v = m_ic_v;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real z = (k+0.5)*dzinv;
			   vel(i,j,k,1) = 6. * v * z * (1.-z);
			 };
      ParallelFor(bx, func);
    }
  else if (322 == m_probtype)
    {
      Real dxinv = 1.0 / Geom(lev).Domain().length(0);
      Real v = m_ic_v;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real x = (i+0.5)*dxinv;
			   vel(i,j,k,1) = 6. * v * x * (1.-x);
			 };
      ParallelFor(bx, func);
    }
  else if (33 == m_probtype)
    {
      Real dxinv = 1.0 / Geom(lev).Domain().length(0);
      Real w = m_ic_w;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real x = (i+0.5)*dxinv;
			   vel(i,j,k,2) = 6. * w * x * (1.-x);
			 };
      ParallelFor(bx, func);
    }
  else if (333 == m_probtype)
    {
      Real dyinv = 1.0 / Geom(lev).Domain().length(1);
      Real w = m_ic_w;
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
			 {
			   Real y = (j+0.5)*dyinv;
			   vel(i,j,k,2) = 6. * w * y * (1.-y);
			 };
      ParallelFor(bx, func);
    }
  else
    {
      const int  dir = ori.coordDir();
      const Real bcv = m_bc_velocity[ori][dir];
      auto func =
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept 
			 { 
			   vel(i,j,k,dir) = bcv; 
			 };
      ParallelFor(bx, func);
    };
}
