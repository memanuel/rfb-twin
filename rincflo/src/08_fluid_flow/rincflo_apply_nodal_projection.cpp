#include <rincflo.H>

Array<amrex::LinOpBCType,SpaceDim>
Rincflo::get_projection_bc (Orientation::Side side) const noexcept
{
    Array<LinOpBCType,SpaceDim> r;
    for (int dir = 0; dir < SpaceDim; ++dir) 
    {
        if (geom[0].isPeriodic(dir)) 
            {r[dir] = LinOpBCType::Periodic;} 
        else 
        {
            auto bc = m_bc_type[Orientation(dir,side)];
            switch (bc)
            {
                case BC::pressure_inflow:
                case BC::pressure_outflow:
                {
                    r[dir] = LinOpBCType::Dirichlet;
                    break;
                }
                case BC::mass_inflow:
                case BC::slip_wall:
                case BC::no_slip_wall:
                case BC::charging_wall_pot:
                case BC::charging_wall_cur:
                case BC::mix_wall_potS:
                case BC::mix_wall_potL:
                case BC::mix_wall_potL_currS:
                case BC::mix_wall_currL_potS:
                {
                    r[dir] = LinOpBCType::Neumann;
                    break;
                }
                default:
                    Abort("get_projection_bc: undefined BC type");
            };
        }
    }
    return r;
}

//
// Computes the following decomposition:
//
//    u + dt grad(phi) / ro = u*,     where div(u) = 0
//
// where u* is a non-div-free velocity field, stored
// by components in u, v, and w. The resulting div-free
// velocity field, u, overwrites the value of u* in u, v, and w.
//
// phi is an auxiliary function related to the pressure p by the relation:
//
//     new p  = phi
//
// except in the initial projection when
//
//     new p  = old p + phi     (nstep has its initial value -1)
//
// Note: scaling_factor equals dt except when called during initial projection, when it is 1.0
//
void Rincflo::ApplyProjection (Vector<MultiFab const*> density, Real time, Real scaling_factor, bool incremental)
{
    #define FUNC_NAME "Rincflo::ApplyProjection"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // If we have dropped the dt substantially for whatever reason,
    // use a different form of the approximate projection that
    // projects (U^*-U^n + dt Gp) rather than (U^* + dt Gp)

    bool proj_for_small_dt = (time > 0.0 && m_dt < 0.1 * m_prev_dt);

    if (m_verbose > 0) 
        {Print() << "Apply projection:\n";}

    // Add the ( grad p /ro ) back to u* (note the +dt)
    if (!incremental)
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& u = ld.velocity.array(mfi);
                Array4<Real const> const& rho = density[lev]->const_array(mfi);
                Array4<Real const> const& gradp = ld.gradp.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real soverrho = scaling_factor / rho(i,j,k);
                    AMREX_D_TERM(
                    u(i,j,k,0) += gradp(i,j,k,0) * soverrho;,
                    u(i,j,k,1) += gradp(i,j,k,1) * soverrho;,
                    u(i,j,k,2) += gradp(i,j,k,2) * soverrho;)
                 };
                ParallelFor(bx, func);
            }
        }
    }

    // Define "vel" to be U^* - U^n rather than U^*
    if (proj_for_small_dt || incremental)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab::Subtract(m_leveldata[lev]->velocity, m_leveldata[lev]->velocity_o, 0, 0, SpaceDim, 0);
        }
    }

    Vector<amrex::MultiFab> sigma(finest_level+1);
    if (!m_constant_density)
    {
        for (int lev = 0; lev <= finest_level; ++lev )
        {
            sigma[lev].define(grids[lev], dmap[lev], 1, 0, MFInfo(), *m_factory[lev]);
            #if (USE_OPENMP)
            #pragma omp parallel if (Gpu::notInLaunchRegion())
            #endif
            for (MFIter mfi(sigma[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& sig = sigma[lev].array(mfi);
                Array4<Real const> const& rho = density[lev]->const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    sig(i,j,k) = scaling_factor / rho(i,j,k);
                };
                ParallelFor(bx, func);
            }
        }
    }

    auto bclo = get_projection_bc(Orientation::low);
    auto bchi = get_projection_bc(Orientation::high);

    Vector<MultiFab*> vel;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        vel.push_back(&(m_leveldata[lev]->velocity));
        vel[lev]->setBndry(0.0);
        if (!proj_for_small_dt and !incremental) 
            {SetInflowVelocity(lev, time, *vel[lev], 1);}
    }

    LPInfo info;
    info.setMaxCoarseningLevel(m_nodal_mg_max_coarsening_level);
    if(m_nodal_bottom_solver == "hypre" ) 
        {info.setMaxCoarseningLevel(0);}

    if (m_constant_density)
    {
        Real constant_sigma = scaling_factor / m_ro_0;
        nodal_proj.reset(new NodalProjector(vel, constant_sigma, Geom(0,finest_level), info));
    }
    else
    {
        nodal_proj.reset(new NodalProjector(vel, GetVecOfConstPtrs(sigma), Geom(0,finest_level), info));
    }
    nodal_proj->setDomainBC(bclo, bchi);
    nodal_proj->project(m_nodal_mg_rtol, m_nodal_mg_atol);

    // Define "vel" to be U^{n+1} rather than (U^{n+1}-U^n)
    if (proj_for_small_dt || incremental)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Add(m_leveldata[lev]->velocity, m_leveldata[lev]->velocity_o, 0, 0, SpaceDim, 0);}
    }


    // Get phi and fluxes
    auto phi = nodal_proj->getPhi();
    auto gradphi = nodal_proj->getGradPhi();

    for(int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(ld.gradp,TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            Box const& tbx = mfi.tilebox();
            Box const& nbx = mfi.nodaltilebox();
            Array4<Real> const& gp_lev = ld.gradp.array(mfi);
            Array4<Real> const& p_lev = ld.pressure.array(mfi);
            Array4<Real const> const& gp_proj = gradphi[lev]->const_array(mfi);
            Array4<Real const> const& p_proj = phi[lev]->const_array(mfi);
            if (incremental) 
            {
                auto func_t =
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    gp_lev(i,j,k,n) += gp_proj(i,j,k,n);
                };
                ParallelFor(tbx, SpaceDim, func_t);
                auto func_n = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    p_lev (i,j,k) += p_proj(i,j,k);
                };
                ParallelFor(nbx, func_n);
            } 
            else 
            {
                auto func_t =
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    gp_lev(i,j,k,n) = gp_proj(i,j,k,n);
                };
                ParallelFor(tbx, SpaceDim, func_t);
                auto func_n = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    p_lev(i,j,k) = p_proj(i,j,k);
                };
                ParallelFor(nbx, func_n);
            }
        }
    }

    for (int lev = finest_level-1; lev >= 0; --lev) 
    {
        #if (AMREX_USE_EB)
        amrex::EB_average_down(m_leveldata[lev+1]->gradp, m_leveldata[lev]->gradp, 0, SpaceDim, refRatio(lev));
        #else
        amrex::average_down(m_leveldata[lev+1]->gradp, m_leveldata[lev]->gradp, 0, SpaceDim, refRatio(lev));
        #endif
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
