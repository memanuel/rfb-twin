#include <rincflo.H>

// *********************************************************************************************************************
// check if steady state is reached
// *********************************************************************************************************************
bool Rincflo::ReactionSteadyStateReached()
{
    #define FUNC_NAME "Rincflo::ReactionSteadyStateReached"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Was the last reaction step rolled back due to convergence failure?
    // If so, must return false or we'll get a spurious early convergence since new = old after the rollback.
    if (m_last_react_step_rollback)
        {return false;}

    // Calculate the current according to BV reaction kinetics and separately save the mass transport current
    if (m_reaction_model == ReactionModel::Nernst)
    {
        m_current = CalcTotalCurrent_BV();
        m_current_mt = m_leveldata[finest_level]->current.sum(0);

        // Special case - when calculating limiting current, the BV current is zero, so use mass transport current
        if (m_current == 0.0)
            {m_current = m_current_mt;}

        m_curr_diff_rel = (m_current_mt - m_current) / max(m_current, m_current_mt);
        if (m_rstep > m_min_rsteps)
            {m_min_curr_diff_rel = min(m_min_curr_diff_rel, abs(m_curr_diff_rel));}
    }
    // Accumulate the current MultiFab to get the total current; this is the current according to BV kinetics
    else
        {m_current = m_leveldata[finest_level]->current.sum(0);}

    // Is the steady state criterion met for current difference (BV vs. mass transport)? 
    // Only applicable in Nernst model.
    bool is_steady_state_mt = (m_reaction_model == ReactionModel::Nernst) ? 
        abs(m_curr_diff_rel) < m_react_tol_mt : true;

    // Is this step stagnant in terms of current difference?
    bool is_stagnant_step_mt = (m_reaction_model == ReactionModel::Nernst) ? 
        (m_min_curr_diff_rel < abs(m_curr_diff_rel)) : false;

    // Calculate the utilization
    m_utilization = m_current / m_current_max;

    // Calculate the difference between the current on the wire and the current implied by flux at the outlet
    m_curr_flux = CalcNetFluxCurrent();
    m_flux_diff = m_curr_flux - m_current;

    // Update m_min_flux_diff and m_hist_flux_diff
    m_flux_diff_rel = m_flux_diff / m_current;
    m_min_flux_diff_rel = min(m_min_flux_diff_rel, abs(m_flux_diff_rel));
    m_hist_flux_diff_rel.push_back(m_flux_diff_rel);

    // Is the steady state criterion met for flux difference?
    bool is_steady_state_flux = abs(m_flux_diff_rel) < m_react_tol_flux;

    // Index in change history for stagnation comparison
    // int idx_lookback_flux = max(m_hist_flux_diff_rel.size() - 101L, 0L);
    // Is the difference between current and flux-equivalent current stagnant on this reaction step?
    // bool is_stagnant_step_flux = (m_min_flux_diff_rel < flux_diff_rel) && 
    //                              (m_hist_flux_diff_rel[idx_lookback_flux] < flux_diff_rel);
    bool is_stagnant_step_flux = (m_min_flux_diff_rel < abs(m_flux_diff_rel));

    // Is the steady state criterion for SOC met on each level?
    bool is_steady_state_soc[finest_level + 1];
    // Is the steady state criterion for potential met on each level?
    bool is_steady_state_epot[finest_level + 1];
    // Is the SOC stagnant on each level?
    bool is_stagnant_soc[finest_level+1];
    // Is the potential change stagnant on each level?
    // bool is_stagnant_epot[finest_level+1];
    // Is the SOC stagnant on this reaction step?
    bool is_stagnant_step_soc = true;
    // Is the potential stagnant on this reaction step?
    // bool is_stagnant_step_epot = true;

    // Prefactor to convert absolute change over one time step to rate of change per flow interval
    const Real rate_factor = m_t_flow / m_dt;

    // Iterate over levels to check change in SOC and potential
    for(int lev = 0; lev <= finest_level; lev++)
    {
        auto const& flag = EBFactory(lev).getMultiEBCellFlagFab();
        // Number of cells in the domain on this level
        long n_cell_dom = Geom(lev).Domain().numPts();

        // The current and previous state of charge
        MultiFab const& soc = m_leveldata[lev]->soc;
        MultiFab const& soc_o = m_leveldata[lev]->soc_o;

        // Assemble a MultiFab with the change in SOC over the time step
        MultiFab soc_chng_mf;
        soc_chng_mf.define(soc.boxArray(), soc.DistributionMap(), 1, 0);
        MultiFab::Copy(soc_chng_mf, soc, 0, 0, 1, 0);
        MultiFab::Subtract(soc_chng_mf, soc_o, 0, 0, 1, 0);

        // The change in the SOC on this level computed with the selected method
        // Alias this from m_change_soc[lev] for code legibility.
        Real change_soc = 0.0;

        // The sum of squares of the change in SOC
        // Real soc_sumsq = SumSquareOverActive(lev, soc_chng_mf, 0);
        Real soc_sumsq = SumSquareOverLiquid(lev, soc_chng_mf, 0);
        // The number of active cells in the domain on this level
        // long n_cell_act = CountActiveCells(lev);
        // The number of liquid cells in the domain on this level
        long n_cell = CountLiquidCells(lev);
        // The root mean square change in the SOC on this level
        Real soc_chng_RMS = std::sqrt(soc_sumsq / n_cell);

        // Determine the change in SOC according to the specified convergence metric
        switch (m_tol_type_soc)
        {
            case ToleranceType::abs_step:
                change_soc = soc_chng_RMS;
                break;
            case ToleranceType::abs_rate:
                change_soc = soc_chng_RMS * rate_factor;
                break;
        }

        // Write the change in SOC on this level computed with the selected method to m_change_soc[lev]
        m_change_soc[lev] = change_soc;
        // Add this change to the history
        m_change_hist_soc[lev].push_back(change_soc);
        // is_steady_state_soc is satisfied when the change in concentration is below the threshold
        is_steady_state_soc[lev] = (change_soc < m_react_tol_soc);
        // Update m_change_min_soc[lev] - smallest change encountered so far on this level
        // Only apply update when past the minimum number of steps
        // This condition is necessary because the first handful of steps can have tiny changes before the
        // prefactor [A][B] in the Butler-Volmer equation increases from its very low initial 
        // value when [H2-AQDS] is tiny.
        if (m_rstep > m_min_rsteps)
            {m_change_min_soc[lev] = min(m_change_min_soc[lev], change_soc);}
        // Index in change history for stagnation comparison
        // int idx_lookback = max(m_change_hist_soc[lev].size() - 101L, 0L);
        // Is the SOC stagnant on this level?
        // is_stagnant_soc[lev] = (m_change_min_soc[lev] < change_soc) && 
        //                        (m_change_hist_soc[lev][idx_lookback] < change_soc);
        is_stagnant_soc[lev] = (m_change_min_soc[lev] < change_soc);
        // Is the SOC stagnant overall on this reaction step?
        is_stagnant_step_soc = is_stagnant_step_soc && is_stagnant_soc[lev];
        // Report change in SOC if requested
        if (m_verbose > 0)
        {
            switch (m_tol_type_soc)
            {
                case ToleranceType::abs_step:
                    Print() << format("RMS change in SOC = {:5.3e} per step.\n", change_soc);
                    break;
                case ToleranceType::abs_rate:
                    Print() << format("RMS change rate of SOC = {:5.3e} per flow period.\n", change_soc );
                    break;
            }
        }

        // Is potential at steady state on this level?
        is_steady_state_epot[lev] = true;
        // Check only for reaction models with epot
        if (m_reaction_model == ReactionModel::ButlerVolmer) 
        {
            Real change_epot = 0.0;
            MultiFab const& epotL   = m_leveldata[lev]->epotL;
            MultiFab const& epotL_o = m_leveldata[lev]->epotL_o;

            // Assemble a MultiFab with the change in epotL over the time step
            MultiFab epot_chng_mf;
            epot_chng_mf.define(epotL.boxArray(), epotL.DistributionMap(), 1, 0);
            MultiFab::Copy(epot_chng_mf, epotL, 0, 0, 1, 0);
            MultiFab::Subtract(epot_chng_mf, epotL_o, 0, 0, 1, 0);

            // The L2-norm of the change in potential
            Real epot_chng_L2 = epot_chng_mf.norm2(0);
            // The RMS of the change in the potential
            Real epot_chng_RMS = epot_chng_L2 / std::sqrt(n_cell_dom);

            // Determine the change in epot according to the specified convergence metric
            switch (m_tol_type_epot)
            {
                case ToleranceType::abs_step:
                    change_epot = epot_chng_RMS;
                    break;

                case ToleranceType::abs_rate:
                    change_epot = epot_chng_RMS * rate_factor;
                    break;
            }
            
            // The change in potential on this level computed with the selected method
            m_change_epot[lev] = change_epot;
            // is_steady_state_epot is satisfied when the change in potential is below the threshold
            is_steady_state_epot[lev] = (change_epot < m_react_tol_epot);
            // Update m_change_min_epot, but only when past the minimum number of steps
            // This condition is necessary because the first handful of steps can have tiny changes before the
            // prefactor [A][B] in the Butler-Volmer equation increases from its very low initial 
            // value when [H2-AQDS] is tiny.
            if (m_rstep > m_min_rsteps)
                {m_change_min_epot[lev] = min(m_change_min_epot[lev], change_epot);}
            // Is the potential stagnant on this level?
            // is_stagnant_epot[lev] = m_change_min_epot[lev] < change_epot;
            // Is the potential stagnant overall on this reaction step?
            // is_stagnant_step_epot = is_stagnant_step_epot && is_stagnant_epot[lev];
            // Report change in epot if requested
            if (m_verbose > 0)
            {
                switch (m_tol_type_epot)
                {
                    case ToleranceType::abs_step:
                        Print() << format("RMS change in epot = {:5.3e} per step.\n", change_epot);
                        break;
                    case ToleranceType::abs_rate:
                        Print() << format("RMS change rate of epot = {:5.3e} per flow period.\n", change_epot );
                        break;
                }
            }
        } // if / ButlerVolmer
    } // for / levels

    // Have we reached steady state?
    bool reached = true;
    // The change in SOC according to the selected metric
    Real change_soc {0.0};
    // The change in epot according to the selected metric
    Real change_epot {0.0};
    
    // Need to satisfy conditions on concentration and potential on every level
    for(int lev = 0; lev <= finest_level; lev++)
    {
        reached = reached && is_steady_state_soc[lev] && is_steady_state_epot[lev];
        change_soc = max(change_soc, m_change_soc[lev]);
        if (m_reaction_model == ReactionModel::ButlerVolmer) 
            {change_epot = max(change_epot, m_change_epot[lev]);}
    }
    
    // is_steady_state_flux is a global conditions (not per level)
    reached = reached && is_steady_state_flux;

    // Increment stagnant step counter if necessary
    if (is_stagnant_step_soc && is_stagnant_step_flux && is_stagnant_step_mt)
    {
        ++m_stagnant_steps;
        // Status message
        if (m_verbose > 0)
        {
            Print() << format("Stagnant step: count={:5d}, change_soc={:5.3e}, change_min_soc[0]={:5.3e}.\n", 
                m_stagnant_steps, change_soc, m_change_min_soc[0], is_stagnant_step_soc);
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return reached;     
}
