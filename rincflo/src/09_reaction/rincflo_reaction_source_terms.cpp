#include <rincflo.H>

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// setup for calc source term                                             //
// -----------------------------------------------------------------------//
void Rincflo::CalcSourceTermAll ()
{
    // The mutltifabs we're writing to: pot_diff, overpot and source
    Vector<MultiFab*> const& pot_diff = get_pot_diff(); 
    Vector<MultiFab*> const& overpot = get_overpot(); 
    Vector<MultiFab*> const& source_mol = get_source_mol();
    Vector<MultiFab*> const& source = get_source();
    Vector<MultiFab*> const& conc = get_conc_new();

    // The multifabs we're reading from
    Vector<MultiFab const*> const& epotL = get_epotL_new_const();
    Vector<MultiFab const*> const& epotS = get_epotS_new_const();
    Vector<MultiFab const*> const& vel = get_velocity_new_const();
 
    // For each level, build the boundary factory and delegate to CalcSourceTerm where necessary
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // The area over volume ratio from the geometry
        const auto& mf_area_over_volume = m_leveldata[lev]->area_over_volume;
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*source[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto& ld = *m_leveldata[lev];
            Box const& bx = mfi.tilebox();
            auto const& fact = EBFactory(lev);
            EBCellFlagFab const& flag_fab = fact.getMultiEBCellFlagFab()[mfi];
            auto typ = flag_fab.getType(bx);

            // Calculate the source term on all boxes that contain at least some boundary cells
            if (fabIsBoundary(typ))
            {
                Array4<Real const> const& area_over_volume = ld.area_over_volume.const_array(mfi);
                Array4<EBCellFlag const> const& flag = flag_fab.const_array();
                CalcSourceTerm(lev, bx,
                        pot_diff[lev]->array(mfi),
                        overpot[lev]->array(mfi),
                        source_mol[lev]->array(mfi),
                        source[lev]->array(mfi),
                        conc[lev]->array(mfi),
                        epotL[lev]->const_array(mfi),
                        epotS[lev]->const_array(mfi),
                        area_over_volume,
                        flag);
            }
        }
    }
}

// *********************************************************************************************************************
// ---------------------------------------------------------------------- //
// calc source term                                                       //
// -----------------------------------------------------------------------//
void Rincflo::CalcSourceTerm (
            int lev, 
            Box const& bx,
            Array4<Real> const& pot_diff,
            Array4<Real> const& overpot,
            Array4<Real> const& source_mol,
            Array4<Real> const& source,
            Array4<Real> const& conc,
            Array4<Real const> const& epotL,
            Array4<Real const> const& epotS,
            Array4<Real const> const& area_over_volume,
            Array4<EBCellFlag const> const& flag)
{
    // The inverse step size
    const Real dx_inv = Geom(lev).InvCellSize(0);

    // Do we adjust overpotential for proton concentration?
    const bool op_proton_adj = (m_reaction_model==ReactionModel::ButlerVolmer) && (m_init_protons > 0.0);
    // Are we covering multiply cut cells?
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    // Local copies of variables
    const Real DVapp = m_DVapp;
    const Real T0 = m_conc_redox_tot;
    const Real RT_over_nF = m_RT_over_nF_const;
    const Real nF_over_RT = m_nF_over_RT_const;
    const Real soc_min = m_soc_min;
    const Real soc_max = m_soc_max;
    const Real overpot_max = m_overpot_max;
    const Real k0 = m_k_0;

    // Different approach for each model; code for special and ButlerVolmer is very similar
    switch (m_reaction_model)
    {
        case ReactionModel::none:
        { 
            // Function to set source term in the case of no reaction model
            auto func_none = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {source(i,j,k) = 0.0;};
            ParallelFor(bx, func_none);
            break;
        }

        // Don't need to compute source term in Nernst model; concentrations are treated as a boundary condition
        case ReactionModel::Nernst: 
            break;

        case ReactionModel::SpecialRedox:
        {
            // Function to set source term in special reaction model
            auto func_special = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (flag(i,j,k).isBoundary(cover_multiple_cuts))
                {
                    // Calculate the state of charge (SOC) of the cell; clamp it to the allowed range
                    Real s = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1)), soc_min, soc_max);

                    // The potential difference is assumed to be delta V in the special model; doesn't need to be updated

                    // Calculate the overpotential eta
                    // eta = DeltaPhi - DeltaPhi_eq
                    // DeltaPhi = m_DVapp is the applied voltage; the reference is 0.5 SOC
                    // DeltaPhi_eq = (RT/nF) log (C_1/C_0)

                    // The leading terms in the overpotential
                    overpot(i,j,k) = DVapp - RT_over_nF * log( s / (1.0 - s) );

                    // Clamp the overpotential to the allowed range
                    overpot(i,j,k) = std::clamp(overpot(i,j,k), -overpot_max, overpot_max);

                    // exchange current when alpha0 = alpha1 = 0.5
                    // j0 = n_el * F * k_0 * Sqrt(C_1 * C_0)
                    // n_el and Faraday const NOT included here (for species transport) and multiplied later for epot
                    // Real j0 = m_k_0 * sqrt(conc(i,j,k,0) * conc(i,j,k,1));
                    Real j0 = k0 * T0 * sqrt(s * (1.0 - s));

                    // Butler-Volmer equation when alpha0 = alpha1 = 0.5
                    // s = 2 j0 * sinh( n_e eta / (RT/F) / 2 )
                    // source term in units of millimoles / second; includes area term
                    // source_mol(i,j,k) = 2.0 * j0 * sinh(0.5 * m_nF_over_RT_const * overpot(i,j,k))* afrac(i,j,k) * cell_area;
                    // normalize the source term by dividing out by the volume
                    // source(i,j,k) = source_mol(i,j,k) / (vfrac(i,j,k) * cell_volume);
                    // Combine into one operation using precomputed area_over_volume multifab
                    source(i,j,k) = 2.0 * j0 * sinh(0.5 * nF_over_RT * overpot(i,j,k)) * area_over_volume(i,j,k);
                }
            };
            ParallelFor(bx, func_special);
            break;
        }
        case ReactionModel::ButlerVolmer:
        {
            // Local copies of member variables
            const int n_protons = m_n_protons;
            const Real init_protons = m_init_protons;
            const Real alpha0 = m_alpha_0;
            const Real alpha1 = m_alpha_1;

            // Function to set source term in Butler-Volmer model
            auto func_bv = 
            [=, this] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if( flag(i,j,k).isBoundary(cover_multiple_cuts))
                {
                    // Calculate the state of charge (SOC) of the cell; clamp it to the allowed range
                    Real s = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1)), soc_min, soc_max);

                    // The potential difference is calculated numerically in ButlerVolmer
                    // Reverse the usual sign convention so a positive potential difference a higher reductive current!
                    // This is very important. pot_diff here has a minus sign compared to the usual definition!
                    // This choice was made so that the overpotential is positive when the cell is reducing,
                    // and so the source term is monotonically increasing with the overpotential.
                    // This helps to build intuition; the overpotential is the driving force for the reaction.
                    // For example, the overpotential is higher when the state of charge s is lower (easier to reduce).
                    pot_diff(i,j,k) = epotL(i,j,k) - epotS(i,j,k);

                    // Calculate the overpotential eta
                    // eta = DeltaPhi - DeltaPhi_eq
                    // DeltaPhi = phi_s - phi_L
                    // When there are no protons:
                    // DeltaPhi_eq = E_0 - (RT/nF) log (C_1/C_0)
                    // When there are protons:
                    // DeltaPhi_eq = E_0 - (RT/nF) log (C_1/C_0) - (Rt/nf) log(C_2/init_prot)
                    // In either case, clamp the overpotential to +/- overpot_max for numerical stability

                    // The leading terms in the overpotential
                    overpot(i,j,k) = pot_diff(i,j,k) - RT_over_nF * log(s / (1.0 - s));

                    // Adjustment for protons if applicable; more protons makes it easier to reduce AQDS
                    if (op_proton_adj)
                        {overpot(i,j,k) += RT_over_nF * n_protons * log(conc(i,j,k,2)/init_protons);}
                    
                    // Clamp the overpotential
                    overpot(i,j,k) = std::clamp(overpot(i,j,k), -overpot_max, overpot_max);

                    // Specialized calculation when alpha = 1/2
                    if(m_same_alpha) 
                    {
                        // exchange current when alpha0 = alpha1 = 0.5
                        // j0 = n_el * F * k_0 * Sqrt(C_1 * C_0)
                        // n_el and Faraday const NOT included here (for species transport) and multiplied later for epot
                        Real j0 = k0 * T0 * sqrt(s * (1.0 - s));

                        // Butler-Volmer equation when alpha0 = alpha1 = 0.5
                        // S = 2 j0 * sinh( n_e eta / (RT/F) / 2 )
                        // source term in units of millimoles / second; includes area term
                        // source_mol(i,j,k) = 2.0 * j0 * sinh(0.5 * m_nF_over_RT_const * overpot(i,j,k)) * afrac(i,j,k) * cell_area;
                        source(i,j,k) = 2.0 * j0 * sinh(0.5 * nF_over_RT * overpot(i,j,k)) * area_over_volume(i,j,k);
                    }
                    // Generalized calculation when alpha != 1/2
                    else 
                    {                            
                        // exchange current in general case when alpha0 != alpha1
                        // j0=n_el * F * k_0 * C_1^{alpha_1} * C_0^{alpha_0}
                        // n_el and Faraday const not included here (for species transport) and multiplied later for epot
                        // Real j0 = m_k_0 * pow(conc(i,j,k,0), m_alpha_0) * pow(conc(i,j,k,1), m_alpha_1);
                        Real j0 = k0 * T0 * pow(s, alpha0) * pow((1.0 - s), alpha1);

                        // Butler-Volmer equation in general
                        // eta_tilde = eta / (RT/nF) = nF * eta / RT is the dimensionless overpotential
                        // i.e. the eta / thermal_voltage including factor for number of electrons
                        // it's the ratio of how energetically favored the reduction reaction is per thermal energy level
                        Real eta_tilde = nF_over_RT * overpot(i,j,k);
                        // source term in units of millimoles / second; includes area term
                        // source_mol(i,j,k) = j0 * (exp(m_alpha_0*eta_tilde) - exp(-m_alpha_1*eta_tilde)) * afrac(i,j,k) * call_area;
                        source(i,j,k) = j0 * (exp(alpha0*eta_tilde) - exp(-alpha1*eta_tilde)) * area_over_volume(i,j,k);
                    }

                    // DEBUG
                    // Report error if source term blows up
                    if(!std::isfinite(source(i,j,k)))
                    {
                        bool is_boundary = flag(i,j,k).isBoundary(cover_multiple_cuts);
                        bool is_covered = flag(i,j,k).isCovered();
                        string cell_type = "liquid";
                        if (is_boundary) {cell_type = "boundary";}
                        if (is_covered) {cell_type = "solid";}
                        Real soc = std::clamp(conc(i,j,k,1) / (conc(i,j,k,0) + conc(i,j,k,1)), soc_min, soc_max);
                        Print() << format(
                            "SOURCE PROBLEM! ButlerVolmer model, lev={:d}, (i,j,k)=({:d}, {:d}, {:d}) cell_type= {:s}.\n", 
                            lev, i, j, k, cell_type);
                        Print() << format(
                            "C0={:f}, C1={:f}, soc={:f}, overpot={:f}, pot_diff={:f}, epotS={:f}. epotL={:f}. area/vol = {:e}, source={:f}.\n",
                            conc(i,j,k,0), conc(i,j,k,1), soc, overpot(i,j,k), pot_diff(i,j,k), epotS(i,j,k), epotL(i,j,k),
                            area_over_volume(i,j,k), source(i,j,k));
                        // HACK - write in reasonable placeholder values
                        // source(i,j,k) = 0.0;
                        // overpot(i,j,k) = 0.0;
                        // Real soc = (m_soc_min + m_soc_max) / 2.0;
                        // conc(i,j,k,0) = T0 * (1.0 - soc);
                        // conc(i,j,k,1) = T0 * soc;
                        Abort();
                    }
                }
            };
            ParallelFor(bx, func_bv);
            break;
        }
    } // switch(m_reaction_model)
}
