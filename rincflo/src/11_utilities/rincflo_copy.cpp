#include <rincflo.H>

// *********************************************************************************************************************
// velocity
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_velocity (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_velocity(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_velocity (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->velocity_o, m_leveldata[lev]->velocity, 0, 0, SpaceDim, ng);
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_velocity (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_velocity(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_velocity (int lev, IntVect const& ng)
{
  MultiFab::Copy(m_leveldata[lev]->velocity, m_leveldata[lev]->velocity_o, 0, 0, SpaceDim, ng);
}

// *********************************************************************************************************************
// density
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_density (IntVect const& ng)
{
  for (int lev = 0; lev <= finest_level; ++lev) 
    {CopyNewToOld_density(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_density (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->density_o, m_leveldata[lev]->density, 0, 0, 1, ng);
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_density (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_density(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_density (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->density, m_leveldata[lev]->density_o, 0, 0, 1, ng);
}

// *********************************************************************************************************************
// conc_adv
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conc_adv (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_conc_adv(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conc_adv (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->conc_adv_o,m_leveldata[lev]->conc_adv, 0, 0, m_nspec, ng);}
}

// *********************************************************************************************************************
// conc
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conc (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_conc(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conc (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->conc_o,m_leveldata[lev]->conc, 0, 0, m_nspec, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_conc (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_conc(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_conc (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->conc, m_leveldata[lev]->conc_o, 0, 0, m_nspec, ng);}
}

// *********************************************************************************************************************
// conv_dconc_dt
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conv_dconc_dt (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_conv_dconc_dt(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_conv_dconc_dt (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->conv_dconc_dt_o,
            m_leveldata[lev]->conv_dconc_dt, 0, 0, m_nspec, ng);
}

// *********************************************************************************************************************
// epotL
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_epotL (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_epotL(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_epotL (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->epotL_o,m_leveldata[lev]->epotL, 0, 0, 1, ng);
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_epotL (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_epotL(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_epotL (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->epotL,m_leveldata[lev]->epotL_o, 0, 0, 1, ng);
}

// *********************************************************************************************************************
// epotS
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_epotS (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_epotS(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_epotS (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->epotS_o, m_leveldata[lev]->epotS, 0, 0, 1, ng);
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_epotS (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_epotS(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_epotS (int lev, IntVect const& ng)
{
    MultiFab::Copy(m_leveldata[lev]->epotS,m_leveldata[lev]->epotS_o, 0, 0, 1, ng);
}

// *********************************************************************************************************************
// soc
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_soc (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOld_soc(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOld_soc (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->soc_o,m_leveldata[lev]->soc, 0, 0, 1, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_soc (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNew_soc(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNew_soc (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->soc,m_leveldata[lev]->soc_o, 0, 0, 1, ng);}
}

// *********************************************************************************************************************
// p_react
// *********************************************************************************************************************

// *********************************************************************************************************************
void Rincflo::CopyNewToOldPicard_react (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyNewToOldPicard_react(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyNewToOldPicard_react (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->conc_op,m_leveldata[lev]->conc, 0, 0, m_nspec, ng); }
    if(m_reaction_model == ReactionModel::ButlerVolmer)    
    {
        MultiFab::Copy(m_leveldata[lev]->epotL_op, m_leveldata[lev]->epotL, 0, 0, 1, ng);
        if(m_solve_epotS) 
            {MultiFab::Copy(m_leveldata[lev]->epotS_op,m_leveldata[lev]->epotS, 0, 0, 1, ng);}
    }
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNewPicard_react (IntVect const& ng)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {CopyOldToNewPicard_react(lev, ng);}
}

// *********************************************************************************************************************
void Rincflo::CopyOldToNewPicard_react (int lev, IntVect const& ng)
{
    if (m_nspec > 0) 
        {MultiFab::Copy(m_leveldata[lev]->conc,m_leveldata[lev]->conc_op, 0, 0, m_nspec, ng); }
    if(m_reaction_model == ReactionModel::ButlerVolmer)    
    {
        MultiFab::Copy(m_leveldata[lev]->epotL, m_leveldata[lev]->epotL_op, 0, 0, 1, ng);
        if(m_solve_epotS) 
            {MultiFab::Copy(m_leveldata[lev]->epotS,m_leveldata[lev]->epotS_op, 0, 0, 1, ng);}
    }
}