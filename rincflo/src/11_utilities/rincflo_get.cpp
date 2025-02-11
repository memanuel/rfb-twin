#include <rincflo.H>

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_velocity_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->velocity));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_velocity_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->velocity_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_density_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->density));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_density_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->density_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conc_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conc_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_o));}
      return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conc_old_nernst () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_on));}
      return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_mac_phi() noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->mac_phi));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_epotL_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotL));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_epotL_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotL_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_epotS_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotS));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_epotS_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev)
        {r.push_back(&(m_leveldata[lev]->epotS_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_pot_diff () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->pot_diff));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_overpot() noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->overpot));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_dv_dt_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_dv_dt_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_dv_dt_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_dv_dt));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_drho_dt_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_drho_dt)); }
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_drho_dt_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_drho_dt_o));}
      return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_dconc_dt_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_dconc_dt));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conv_dconc_dt_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_dconc_dt_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_conc_adv_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_adv));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_divtau_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->divtau_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_divtau_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->divtau));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_laps_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->laps_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_laps_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev)
        {r.push_back(&(m_leveldata[lev]->laps));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_soc_new () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->soc));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_soc_old () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->soc_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_source_mol () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->source_mol));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_current () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->current));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_source () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->source));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_kappa () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->kappa));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_flxrhs () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->flxrhs));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab*> Rincflo::get_divcgpot () noexcept
{
    Vector<MultiFab*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->divcgpot));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_velocity_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->velocity));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_velocity_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->velocity_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_density_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->density));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_density_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->density_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_conc_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_conc_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_conc_old_nernst_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_on));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_epotS_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotS_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_epotL_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotL));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_epotL_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->epotL_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_epotS_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev)
        {r.push_back(&(m_leveldata[lev]->epotS));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_pot_diff_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev)
        {r.push_back(&(m_leveldata[lev]->pot_diff));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_overpot_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev)
        {r.push_back(&(m_leveldata[lev]->overpot));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_conv_dconc_dt_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conv_dconc_dt));}
    return r;
}

//get_conc_adv_new_const
// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_conc_adv_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->conc_adv));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_soc_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->soc));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_source_mol_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->source_mol));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_source_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->source));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_laps_new_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->laps));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_laps_old_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->laps_o));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_kappa_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->kappa));}
    return r;
}

// *********************************************************************************************************************
Vector<MultiFab const*> Rincflo::get_flxrhs_const () const noexcept
{
    Vector<MultiFab const*> r;
    r.reserve(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {r.push_back(&(m_leveldata[lev]->flxrhs));}
    return r;
}
