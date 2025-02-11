#include <rincflo.H>

void Rincflo::Advance()
{
    #define FUNC_NAME "Rincflo::Advance"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Start timing current time step
    Real strt_step = second();

    // Compute time step size
    int initialisation = 0;
    bool explicit_diffusion = (m_diff_type == DiffusionType::Explicit);
    ComputeDt(initialisation, explicit_diffusion);

    // Set new and old time to correctly use in fillpatching
    UpdateTimeArrays();

    if (m_verbose > 0)
    {
        Print() << format("\nStep {:d}: from old_time {:0.8f} to new time {:0.8f} with dt = {:6.2e}.\n",
            m_nstep+1, m_cur_time, m_cur_time + m_dt, m_dt);
    }

    CopyNewToOld_velocity();
    CopyNewToOld_density();
    CopyNewToOld_conc();

    const int ng = nghost_state();
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        fillpatch_velocity(lev, m_t_old[lev], m_leveldata[lev]->velocity_o, ng);
        fillpatch_density(lev, m_t_old[lev], m_leveldata[lev]->density_o, ng);
        if (do_AdvectConc()) 
            {fillpatch_conc(lev, m_t_old[lev], m_leveldata[lev]->conc_o, ng);}
    }
    
    ApplyPredictor();

    if (m_advection_type == "MOL") 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            fillpatch_velocity(lev, m_t_new[lev], m_leveldata[lev]->velocity, ng);
            fillpatch_density(lev, m_t_new[lev], m_leveldata[lev]->density, ng);
            if (do_AdvectConc()) 
                {fillpatch_conc(lev, m_t_new[lev], m_leveldata[lev]->conc, ng);}
        }
        ApplyCorrector();
    }
  
    // Stop timing current time step
    Real end_step = second() - strt_step;
    ReduceRealMax(end_step, IOProcessorNumber());
    if (m_verbose > 0)
      {Print() << format("Time per step {:10.8f} \n", end_step);}
    
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
