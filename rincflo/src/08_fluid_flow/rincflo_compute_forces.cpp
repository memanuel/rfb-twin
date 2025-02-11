#include <rincflo.H>

void Rincflo::ComputeTraForces (Vector<MultiFab*> const& tra_forces,
                                 Vector<MultiFab const*> const& density)
{
    // NOTE: this routine must return the force term for the update of (rho s), NOT just s.
    if (do_AdvectConc()) {
#if (USE_OPENMP)
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (MFIter mfi(*tra_forces[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) 
            {
                Box const& bx = mfi.tilebox();
                Array4<Real>       const& tra_f = tra_forces[lev]->array(mfi);
                Array4<Real const> const& rho   =    density[lev]->const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    // For now we don't have any external forces on the scalars
                    tra_f(i,j,k,n) = 0.0;
    
                    // Return the force term for the update of (rho s), NOT just s.
                    tra_f(i,j,k,n) *= rho(i,j,k);
                };
                ParallelFor(bx, m_nspec, func);
            }
        }
    }
}

void Rincflo::ComputeVelForces (Vector<MultiFab*> const& vel_forces,
                                 Vector<MultiFab const*> const& velocity,
                                 Vector<MultiFab const*> const& density,
                                 Vector<MultiFab const*> const& conc_old,
                                 Vector<MultiFab const*> const& conc_new,
                                 bool include_pressure_gradient)
{
    for (int lev = 0; lev <= finest_level; ++lev)  
       ComputeVelForcesOnLevel (lev, *vel_forces[lev], *velocity[lev], *density[lev], 
                                         *conc_old[lev], *conc_new[lev], include_pressure_gradient);
}

void Rincflo::ComputeVelForcesOnLevel (int lev,
                                                MultiFab& vel_forces,
					   const MultiFab& /*velocity*/,
                                          const MultiFab& density,
                                          const MultiFab& conc_old,
                                          const MultiFab& conc_new,
                                          bool include_pressure_gradient)
{

    GpuArray<Real,3> l_gravity{m_gravity[0],m_gravity[1],m_gravity[2]};
    GpuArray<Real,3> l_gp0{m_gp0[0], m_gp0[1], m_gp0[2]};

#if (USE_OPENMP)
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(vel_forces,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
            Box const& bx = mfi.tilebox();
            Array4<Real>       const& vel_f =  vel_forces.array(mfi);
            Array4<Real const> const&   rho =     density.const_array(mfi);
            Array4<Real const> const& gradp = m_leveldata[lev]->gradp.const_array(mfi);

            if (m_use_boussinesq) {
                // This uses a Boussinesq approximation where the buoyancy depends on first conc rather than density
                Array4<Real const> const& conc_o = conc_old.const_array(mfi);
                Array4<Real const> const& tra_n = conc_new.const_array(mfi);
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    int n = 0; // Potential temperature

                    Real rhoinv = 1.0/rho(i,j,k);
                    Real ft = 0.5 * (conc_o(i,j,k,n) + tra_n(i,j,k,n));

                    if (include_pressure_gradient)
                    {
                        AMREX_D_TERM(
                        vel_f(i,j,k,0) = -gradp(i,j,k,0)*rhoinv + l_gravity[0] * ft;,
                        vel_f(i,j,k,1) = -gradp(i,j,k,1)*rhoinv + l_gravity[1] * ft;,
                        vel_f(i,j,k,2) = -gradp(i,j,k,2)*rhoinv + l_gravity[2] * ft;)
                    } 
                    else 
                    {
                        AMREX_D_TERM(
                        vel_f(i,j,k,0) =                          l_gravity[0] * ft;,
                        vel_f(i,j,k,1) =                          l_gravity[1] * ft;,
                        vel_f(i,j,k,2) =                          l_gravity[2] * ft;)
                    }
                };
                ParallelFor(bx, func);

            } 
            else 
            {
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real rhoinv = 1.0/rho(i,j,k);

                    if (include_pressure_gradient)
                    {
                        AMREX_D_TERM(
                        vel_f(i,j,k,0) = -(gradp(i,j,k,0)+l_gp0[0])*rhoinv + l_gravity[0];,
                        vel_f(i,j,k,1) = -(gradp(i,j,k,1)+l_gp0[1])*rhoinv + l_gravity[1];,
                        vel_f(i,j,k,2) = -(gradp(i,j,k,2)+l_gp0[2])*rhoinv + l_gravity[2];)
                    } 
                    else 
                    {
                        AMREX_D_TERM(
                        vel_f(i,j,k,0) = -(               l_gp0[0])*rhoinv + l_gravity[0];,
                        vel_f(i,j,k,1) = -(               l_gp0[1])*rhoinv + l_gravity[1];,
                        vel_f(i,j,k,2) = -(               l_gp0[2])*rhoinv + l_gravity[2];)
                    }
                };
                ParallelFor(bx, func);
            }
    }
}
