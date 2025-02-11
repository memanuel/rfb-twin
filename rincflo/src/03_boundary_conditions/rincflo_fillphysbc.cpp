#include <rincflo.H>

void Rincflo::fillphysbc_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > physbc(geom[lev], get_velocity_bcrec(),
                                                        RincfloVelFill{m_probtype, m_bc_velocity});
    physbc.FillBoundary(vel, 0, SpaceDim, IntVect(ng), time, 0);
}

void Rincflo::fillphysbc_density (int lev, Real time, MultiFab& density, int ng)
{
    PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > physbc(geom[lev], get_density_bcrec(),
                                                        RincfloDenFill{m_probtype, m_bc_density});
    physbc.FillBoundary(density, 0, 1, IntVect(ng), time, 0);
}

void Rincflo::fillphysbc_conc (int lev, Real time, MultiFab& conc, int ng)
{
    if (m_nspec > 0) {
        PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > physbc
	  (geom[lev], get_conc_bcrec(), RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
        physbc.FillBoundary(conc, 0, m_nspec, IntVect(ng), time, 0);
    }
}

void Rincflo::fillphysbc_epotL (int lev, Real time, MultiFab& epotL, int ng)
{
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > physbc(geom[lev], get_epotL_bcrec(),
						       RincfloEpotFill{m_probtype, m_bc_epotL});
  physbc.FillBoundary(epotL, 0, 1, IntVect(ng), time, 0);
}

void Rincflo::fillphysbc_epotS (int lev, Real time, MultiFab& epotS, int ng)
{
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > physbc(geom[lev], get_epotS_bcrec(),
						       RincfloEpotFill{m_probtype, m_bc_epotS});
  physbc.FillBoundary(epotS, 0, 1, IntVect(ng), time, 0);
}

