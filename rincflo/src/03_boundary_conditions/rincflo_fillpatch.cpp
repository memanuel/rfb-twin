#include <rincflo.H>

// *********************************************************************************************************************
void Rincflo::fillpatch_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    if (lev == 0) 
    {
        PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > physbc
            (geom[lev], get_velocity_bcrec(), RincfloVelFill{m_probtype, m_bc_velocity});
        FillPatchSingleLevel(
            vel, IntVect(ng), time,
            {&(m_leveldata[lev]->velocity_o),
            &(m_leveldata[lev]->velocity)},
            {m_t_old[lev], m_t_new[lev]}, 0, 0, SpaceDim, geom[lev],
            physbc, 0);
    } 
    else 
    {
        const auto& bcrec = get_velocity_bcrec();
        PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > cphysbc
            (geom[lev-1], bcrec, RincfloVelFill{m_probtype, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > fphysbc
            (geom[lev], bcrec, RincfloVelFill{m_probtype, m_bc_velocity});
        #if (AMREX_USE_EB)
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
        #else
        Interpolater* mapper = &cell_cons_interp;
        #endif
        FillPatchTwoLevels(
            vel, IntVect(ng), time,
            {&(m_leveldata[lev-1]->velocity_o),
            &(m_leveldata[lev-1]->velocity)},
            {m_t_old[lev-1], m_t_new[lev-1]},
            {&(m_leveldata[lev]->velocity_o),
            &(m_leveldata[lev]->velocity)},
            {m_t_old[lev], m_t_new[lev]},
            0, 0, SpaceDim, geom[lev-1], geom[lev],
            cphysbc, 0, fphysbc, 0,
            refRatio(lev-1), mapper, bcrec, 0);
    }
}

// *********************************************************************************************************************
void Rincflo::fillpatch_density (int lev, Real time, MultiFab& density, int ng)
{
    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > physbc(
            geom[lev], get_density_bcrec(), RincfloDenFill{m_probtype, m_bc_density});
        FillPatchSingleLevel(
            density, IntVect(ng), time,
            {&(m_leveldata[lev]->density_o),
            &(m_leveldata[lev]->density)},
            {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
            physbc, 0);
    } 
    else 
    {
        const auto& bcrec = get_density_bcrec();
        PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > cphysbc
            (geom[lev-1], bcrec, RincfloDenFill{m_probtype, m_bc_density});
        PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > fphysbc
            (geom[lev], bcrec, RincfloDenFill{m_probtype, m_bc_density});
        #if (AMREX_USE_EB)
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
        #else
        Interpolater* mapper = &cell_cons_interp;
        #endif
        FillPatchTwoLevels(
            density, IntVect(ng), time,
            {&(m_leveldata[lev-1]->density_o),
            &(m_leveldata[lev-1]->density)},
            {m_t_old[lev-1], m_t_new[lev-1]},
            {&(m_leveldata[lev]->density_o),
            &(m_leveldata[lev]->density)},
            {m_t_old[lev], m_t_new[lev]},
            0, 0, 1, geom[lev-1], geom[lev],
            cphysbc, 0, fphysbc, 0,
            refRatio(lev-1), mapper, bcrec, 0);
    }
}

// *********************************************************************************************************************
void Rincflo::fillpatch_conc (int lev, Real time, MultiFab& conc, int ng)
{
    if (lev == 0) 
    {
        PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > physbc
            (geom[lev], get_conc_bcrec(), RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
        FillPatchSingleLevel(
            conc, IntVect(ng), time,
            {&(m_leveldata[lev]->conc_o), &(m_leveldata[lev]->conc)},
            {m_t_old[lev], m_t_new[lev]}, 
            0, 0, m_nspec, geom[lev], physbc, 0);
    } 
    else 
    {
        const auto& bcrec = get_conc_bcrec();
        PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > cphysbc
            (geom[lev-1], bcrec, RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
        PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > fphysbc
            (geom[lev], bcrec, RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
        #if (AMREX_USE_EB)
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
        #else
        Interpolater* mapper = &cell_cons_interp;
        #endif
        FillPatchTwoLevels(
            conc, IntVect(ng), time,
            {&(m_leveldata[lev-1]->conc_o), &(m_leveldata[lev-1]->conc)},
            {m_t_old[lev-1], m_t_new[lev-1]}, 
            {&(m_leveldata[lev]->conc_o), &(m_leveldata[lev]->conc)}, 
            {m_t_old[lev], m_t_new[lev]},
            0, 0, m_nspec, geom[lev-1], geom[lev],
            cphysbc, 0, fphysbc, 0,
            refRatio(lev-1), mapper, bcrec, 0);
    }
}

// *********************************************************************************************************************
void Rincflo::fillpatch_gradp (int lev, Real time, MultiFab& gradp, int ng)
{
    if (lev == 0) 
    {
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > physbc
            (geom[lev], get_force_bcrec(), RincfloForFill{m_probtype});
        FillPatchSingleLevel(
            gradp, IntVect(ng), time,
            {&(m_leveldata[lev]->gradp)}, {time},
            0, 0, SpaceDim, geom[lev], physbc, 0);
    } 
    else 
    {
        const auto& bcrec = get_force_bcrec();
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > cphysbc
            (geom[lev-1], bcrec, RincfloForFill{m_probtype});
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > fphysbc
            (geom[lev], bcrec, RincfloForFill{m_probtype});
        #if (AMREX_USE_EB)
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
        #else
        Interpolater* mapper = &cell_cons_interp;
        #endif
        FillPatchTwoLevels(
            gradp, IntVect(ng), time,
            {&(m_leveldata[lev-1]->gradp)}, {time},
            {&(m_leveldata[lev]->gradp)}, {time},
            0, 0, SpaceDim, geom[lev-1], geom[lev],
            cphysbc, 0, fphysbc, 0,
            refRatio(lev-1), mapper, bcrec, 0);
    }
}

// *********************************************************************************************************************
void Rincflo::fillpatch_force (Real time, Vector<MultiFab*> const& force, int ng)
{
    const int ncomp = force[0]->nComp();
    const auto& bcrec = get_force_bcrec();
    int lev = 0;
    {
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > physbc
            (geom[lev], bcrec, RincfloForFill{m_probtype});
        FillPatchSingleLevel(
            *force[lev], IntVect(ng), time,
            {force[lev]}, {time},
            0, 0, ncomp, geom[lev],
            physbc, 0);
    }
    for (lev = 1; lev <= finest_level; ++lev)
    {
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > cphysbc
            (geom[lev-1], bcrec, RincfloForFill{m_probtype});
        PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > fphysbc
            (geom[lev  ], bcrec, RincfloForFill{m_probtype});
        Interpolater* mapper = &pc_interp;
        FillPatchTwoLevels(
            *force[lev], IntVect(ng), time,
            {force[lev-1]}, {time},
            {force[lev  ]}, {time},
            0, 0, ncomp, geom[lev-1], geom[lev],
            cphysbc, 0, fphysbc, 0,
            refRatio(lev-1), mapper, bcrec, 0);
    }
}

// *********************************************************************************************************************
void Rincflo::fillcoarsepatch_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    const auto& bcrec = get_velocity_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > cphysbc
        (geom[lev-1], bcrec, RincfloVelFill{m_probtype, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<RincfloVelFill> > fphysbc
        (geom[lev], bcrec, RincfloVelFill{m_probtype, m_bc_velocity});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    InterpFromCoarseLevel(
        vel, IntVect(ng), time,
        m_leveldata[lev-1]->velocity, 0, 0, SpaceDim,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
}

void Rincflo::fillcoarsepatch_density (int lev, Real time, MultiFab& density, int ng)
{
    const auto& bcrec = get_density_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > cphysbc
        (geom[lev-1], bcrec, RincfloDenFill{m_probtype, m_bc_density});
    PhysBCFunct<GpuBndryFuncFab<RincfloDenFill> > fphysbc
        (geom[lev], bcrec, RincfloDenFill{m_probtype, m_bc_density});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    InterpFromCoarseLevel(
        density, IntVect(ng), time,
        m_leveldata[lev-1]->density, 0, 0, 1,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
}

// *********************************************************************************************************************
void Rincflo::fillcoarsepatch_conc (int lev, Real time, MultiFab& conc, int ng)
{
    const auto& bcrec = get_conc_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > cphysbc
        (geom[lev-1], bcrec, RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
    PhysBCFunct<GpuBndryFuncFab<RincfloTracFill> > fphysbc
        (geom[lev], bcrec, RincfloTracFill{m_probtype, m_nspec, m_bc_conc_d});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    InterpFromCoarseLevel(
        conc, IntVect(ng), time,
        m_leveldata[lev-1]->conc, 0, 0, m_nspec,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
}

// *********************************************************************************************************************
void Rincflo::fillcoarsepatch_gradp (int lev, Real time, MultiFab& gradp, int ng)
{
    const auto& bcrec = get_force_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > cphysbc
        (geom[lev-1], bcrec, RincfloForFill{m_probtype});
    PhysBCFunct<GpuBndryFuncFab<RincfloForFill> > fphysbc
        (geom[lev], bcrec, RincfloForFill{m_probtype});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    InterpFromCoarseLevel(
        gradp, IntVect(ng), time,
        m_leveldata[lev-1]->gradp, 0, 0, SpaceDim,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
}


// *********************************************************************************************************************
void Rincflo::fillpatch_epotL (int lev, Real time, MultiFab& epotL, int ng)
{
  if (lev == 0) 
  {
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > physbc(
        geom[lev], get_epotL_bcrec(), RincfloEpotFill{m_probtype, m_bc_epotL});   
    FillPatchSingleLevel(
        epotL, IntVect(ng), time,
        {&(m_leveldata[lev]->epotL_o),
        &(m_leveldata[lev]->epotL)},
        {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
        physbc, 0);
  } 
  else 
  {
    const auto& bcrec = get_epotL_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > cphysbc
      (geom[lev-1], bcrec, RincfloEpotFill{m_probtype, m_bc_epotL});
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > fphysbc
      (geom[lev], bcrec, RincfloEpotFill{m_probtype, m_bc_epotL});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
      (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    FillPatchTwoLevels(
        epotL, IntVect(ng), time,
        {&(m_leveldata[lev-1]->epotL_o),
        &(m_leveldata[lev-1]->epotL)},
        {m_t_old[lev-1], m_t_new[lev-1]},
        {&(m_leveldata[lev]->epotL_o),
        &(m_leveldata[lev]->epotL)},
        {m_t_old[lev], m_t_new[lev]},
        0, 0, 1, geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
  }
}

// *********************************************************************************************************************
void Rincflo::fillcoarsepatch_epotL (int lev, Real time, MultiFab& epotL, int ng)
{
  const auto& bcrec = get_epotL_bcrec();
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > cphysbc
    (geom[lev-1], bcrec, RincfloEpotFill{m_probtype, m_bc_epotL});
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > fphysbc
    (geom[lev], bcrec, RincfloEpotFill{m_probtype, m_bc_epotL});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
      (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    InterpFromCoarseLevel(
        epotL, IntVect(ng), time,
        m_leveldata[lev-1]->epotL, 0, 0, 1,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);    
}

// *********************************************************************************************************************
void Rincflo::fillpatch_epotS (int lev, Real time, MultiFab& epotS, int ng)
{
  if (lev == 0) 
  {
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > physbc(
        geom[lev], get_epotS_bcrec(), RincfloEpotFill{m_probtype, m_bc_epotS});   
    FillPatchSingleLevel(
        epotS, IntVect(ng), time,
        {&(m_leveldata[lev]->epotS_o),
            &(m_leveldata[lev]->epotS)},
        {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
        physbc, 0);
  } 
  else 
  {
    const auto& bcrec = get_epotS_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > cphysbc
      (geom[lev-1], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > fphysbc
      (geom[lev], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
      (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&ebinv_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    FillPatchTwoLevels(
        epotS, IntVect(ng), time,
        {&(m_leveldata[lev-1]->epotS_o),
        &(m_leveldata[lev-1]->epotS)},
        {m_t_old[lev-1], m_t_new[lev-1]},
        {&(m_leveldata[lev]->epotS_o),
        &(m_leveldata[lev]->epotS)},
        {m_t_old[lev], m_t_new[lev]},
        0, 0, 1, geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
  }
}

// *********************************************************************************************************************
void Rincflo::fillcoarsepatch_epotS (int lev, Real time, MultiFab& epotS, int ng)
{
  const auto& bcrec = get_epotS_bcrec();
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > cphysbc
    (geom[lev-1], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});
  PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > fphysbc
    (geom[lev], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});

    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
      (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&ebinv_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif

    InterpFromCoarseLevel(
        epotS, IntVect(ng), time,
        m_leveldata[lev-1]->epotS, 0, 0, 1,
        geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);    
}

// MSE 2023-02-21 - only use this for bootstrapping in the Nernst model
// *********************************************************************************************************************
void Rincflo::fillpatch_overpot(int lev, Real time, MultiFab& overpot, int ng)
{
  if (lev > 0)  
  {
    // As a hack treat the boudnary condition for the overpotential as the same as the solid potential
    const auto& bcrec = get_epotS_bcrec();
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > cphysbc
      (geom[lev-1], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});
    PhysBCFunct<GpuBndryFuncFab<RincfloEpotFill> > fphysbc
      (geom[lev], bcrec, RincfloEpotFill{m_probtype, m_bc_epotS});
    #if (AMREX_USE_EB)
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
      (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
    #else
    Interpolater* mapper = &cell_cons_interp;
    #endif
    FillPatchTwoLevels(overpot, IntVect(ng), time,
        {&(m_leveldata[lev-1]->overpot), &(m_leveldata[lev-1]->overpot)},
        {m_t_old[lev-1], m_t_new[lev-1]},
        {&(m_leveldata[lev]->overpot), &(m_leveldata[lev]->overpot)},
        {m_t_old[lev], m_t_new[lev]},
        0, 0, 1, geom[lev-1], geom[lev],
        cphysbc, 0, fphysbc, 0,
        refRatio(lev-1), mapper, bcrec, 0);
  }
}
