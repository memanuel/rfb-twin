#include <rincflo.H>

void
Rincflo::Compute_MacProjectedVelocities (
    Vector<MultiFab const*> const& vel,
    AMREX_D_DECL(
    Vector<MultiFab*> const& u_mac,
    Vector<MultiFab*> const& v_mac,
    Vector<MultiFab*> const& w_mac),
    AMREX_D_DECL(
    Vector<MultiFab*> const& inv_rho_x,
    Vector<MultiFab*> const& inv_rho_y,
    Vector<MultiFab*> const& inv_rho_z),
    Vector<MultiFab*> const& vel_forces,
    Real time)
{
    #define FUNC_NAME "Rincflo::Compute_MacProjectedVelocities"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    Real l_dt = m_dt;

    auto mac_phi = get_mac_phi();
    Vector<Array<MultiFab const*,SpaceDim> > inv_rho(finest_level+1);

    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(
        inv_rho[lev][0] = inv_rho_x[lev];,
        inv_rho[lev][1] = inv_rho_y[lev];,
        inv_rho[lev][2] = inv_rho_z[lev];)
    }

    // Initialize (or redefine the beta in) the MacProjector
    if (mac_proj->needInitialization()) 
    {
        LPInfo lp_info;
        lp_info.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);
        #if (! AMREX_USE_EB)
        if (m_constant_density) 
        {
            Vector<BoxArray> ba;
            Vector<DistributionMapping> dm;
            for (auto const& ir : inv_rho) 
            {
                ba.push_back(ir[0]->boxArray());
                dm.push_back(ir[0]->DistributionMap());
            }
            mac_proj->initProjector(ba, dm, lp_info, 1.0/m_ro_0);
        } 
        else
        #endif

        mac_proj->initProjector(lp_info, inv_rho);
        // DEBUG_PRINT("Completed mac_proj->initProjector().\n");
        mac_proj->setDomainBC(get_projection_bc(Orientation::low), get_projection_bc(Orientation::high));
        // DEBUG_PRINT("Completed mac_proj->initProjector() and mac_proj->setDomainBC()\n");
    } 
    else 
    {
        #if (! AMREX_USE_EB)
        if (m_constant_density) 
            {mac_proj->updateBeta(1.0/m_ro_0);  // unnecessary unless m_ro_0 changes.} 
        else
        #endif
        mac_proj->updateBeta(inv_rho);
        DEBUG_PRINT("Completed mac_proj->updateBeta(inv_rho)\n");
    }

    Vector<Array<MultiFab,SpaceDim> > m_fluxes;
    m_fluxes.resize(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < SpaceDim; ++idim) 
        {
            m_fluxes[lev][idim].define(
                convert(grids[lev], IntVect::TheDimensionVector(idim)),
                dmap[lev], 1, 0, MFInfo(), Factory(lev));
        }
    }
    DEBUG_PRINT("Initialized m_fluxes.\n");

    if (m_use_mac_phi_in_godunov)
    {
        #if (AMREX_USE_EB)
        mac_proj->getFluxes(amrex::GetVecOfArrOfPtrs(m_fluxes), mac_phi, MLMG::Location::FaceCentroid);
        DEBUG_PRINT("Completed mac_proj->getFluxes().\n");
        #else
        mac_proj->getFluxes(amrex::GetVecOfArrOfPtrs(m_fluxes), mac_phi, MLMG::Location::FaceCenter);
        #endif
    } 
    else 
    {
        for (int lev=0; lev <= finest_level; ++lev)
        {
            for (int idim = 0; idim < SpaceDim; ++idim) 
                 {m_fluxes[lev][idim].setVal(0.);}
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        mac_phi[lev]->FillBoundary(geom[lev].periodicity());
        #if (AMREX_USE_EB)
        const EBFArrayBoxFactory* ebfact = &EBFactory(lev);
        #endif

        // Predict normal velocity to faces -- note that the {u_mac, v_mac, w_mac}
        // returned from this call are on face CENTROIDS

        if (m_advection_type == "Godunov") 
        {
            #if (AMREX_USE_EB)
            if (ebfact->isAllRegular())
            {
                godunov::predict_godunov(
                    time, 
                    AMREX_D_DECL(*u_mac[lev], *v_mac[lev], *w_mac[lev]), 
                    *vel[lev], *vel_forces[lev], 
                    get_velocity_bcrec(), get_velocity_bcrec_device_ptr(), 
                    geom[lev], l_dt, m_godunov_ppm, m_godunov_use_forces_in_trans,
                    AMREX_D_DECL(m_fluxes[lev][0], m_fluxes[lev][1], m_fluxes[lev][2]), 
                    m_use_mac_phi_in_godunov);
                DEBUG_PRINT("Completed godunov::predict_godunov() on all regular region.\n");
            }
            else
            {
                ebgodunov::predict_godunov(
                    time,
                    AMREX_D_DECL(*u_mac[lev], *v_mac[lev], *w_mac[lev]),
                    *vel[lev], *vel_forces[lev], 
                    get_velocity_bcrec(), get_velocity_bcrec_device_ptr(), 
                    ebfact, geom[lev], l_dt,
                    AMREX_D_DECL(m_fluxes[lev][0], m_fluxes[lev][1], m_fluxes[lev][2]), 
                    m_use_mac_phi_in_godunov);
                DEBUG_PRINT("Completed ebgodunov::predict_godunov().\n");
            }
            #else
            {
                godunov::predict_godunov(
                    time, 
                    AMREX_D_DECL(*u_mac[lev], *v_mac[lev], *w_mac[lev]), 
                    *vel[lev], *vel_forces[lev], 
                    get_velocity_bcrec(), get_velocity_bcrec_device_ptr(), 
                    geom[lev], l_dt, m_godunov_ppm, m_godunov_use_forces_in_trans,
                    AMREX_D_DECL(m_fluxes[lev][0], m_fluxes[lev][1], m_fluxes[lev][2]), 
                    m_use_mac_phi_in_godunov);
            }
            #endif
        } 
        else if (m_advection_type == "MOL") 
        {
            mol::predict_vels_on_faces(
                lev, AMREX_D_DECL(*u_mac[lev], *v_mac[lev], *w_mac[lev]), 
                *vel[lev], get_velocity_bcrec(), get_velocity_bcrec_device_ptr(), 
                #if (AMREX_USE_EB)
                ebfact,
                #endif
                geom);
            DEBUG_PRINT("Completed mol::predict_vels_on_faces().\n");
        } 
        else 
        {
            Abort("Don't know this advection type");
        }
    }
    Vector<Array<MultiFab*,SpaceDim> > mac_vec(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(
        mac_vec[lev][0] = u_mac[lev];,
        mac_vec[lev][1] = v_mac[lev];,
        mac_vec[lev][2] = w_mac[lev];)
    }

    mac_proj->setUMAC(mac_vec);
    DEBUG_PRINT("Completed mac_proj->setUMAC(mac_vec).\n");

    if (m_verbose > 2) 
        {Print() << "MAC Projection:\n";}
    // Perform MAC projection
    if (m_use_mac_phi_in_godunov)
    {
        DEBUG_PRINT("mac_proj: m_mac_phi is true; calling mac_phi->mult with m_dt/2.\n");
        for (int lev=0; lev <= finest_level; ++lev)
            {mac_phi[lev]->mult(m_dt/2.,0,1,1);}
        DEBUG_PRINT(std::format("mac_proj->project(mac_phi, {:5.3e}, {:5.3e})\n", m_mac_mg_rtol, m_mac_mg_atol));
        mac_proj->project(mac_phi,m_mac_mg_rtol,m_mac_mg_atol);
        DEBUG_PRINT("Completed mac_proj->project() in godunov.\n");

        for (int lev=0; lev <= finest_level; ++lev)
            {mac_phi[lev]->mult(2./m_dt,0,1,1);}
    } 
    else 
    {
        DEBUG_PRINT("mac_proj: m_mac_phi is false.\n");
        DEBUG_PRINT(std::format("mac_proj->project(m_mac, {:5.3e}, {:5.3e})\n", m_mac_mg_rtol, m_mac_mg_atol));
        mac_proj->project(m_mac_mg_rtol,m_mac_mg_atol);
        DEBUG_PRINT("Completed mac_proj->project().\n");
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
