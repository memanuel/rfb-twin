#include <rincflo.H>
#include <rincflo_derive_K.H>

void Rincflo::DiffFromExact (int lev, Geometry& lev_geom, Real time, Real dt,
			     MultiFab& error, int soln_comp, int err_comp) 
{
    auto const& dx = lev_geom.CellSizeArray();
    constexpr Real pi = std::numbers::pi;
    constexpr Real twopi = 2.0*pi;
    constexpr Real fourpi = 4.0*pi;

    // Taylor-Green vortices
    if (1 == m_probtype)
    {
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution 
            Array4<Real> const& err = error.array(mfi);
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                AMREX_D_TERM(
                Real x = (i+0.5)*dx[0];,
                Real y = (j+0.5)*dx[1];,
                Real z = (k+0.5)*dx[2];)
                Real exact;
                if (err_comp == SpaceDim || err_comp == SpaceDim+1) 
                {  
                    // pressure 
                    exact = 0.25 * std::cos(fourpi*x) + 0.25 * std::cos(fourpi*y);    
                } 
                else if (err_comp == 0) 
                { 
                    // u
                    exact =  AMREX_D_TERM(std::sin(twopi*x), *std::cos(twopi*y), *std::cos(twopi*z));
                } 
                else if (err_comp == 1) 
                {
                    // v
                    exact = -AMREX_D_TERM(std::cos(twopi*x), *std::sin(twopi*y), *std::cos(twopi*z));
                #if (AMREX_IS_3D)
                } 
                else if (err_comp == 2) 
                { 
                    // w
                    exact = 0.;
                #endif
                }
                err(i,j,k,soln_comp) -= exact;
            };
            ParallelFor(bx, func);
        }
    // Decaying Taylor vortex
    } 
    else if (2 == m_probtype) 
    {
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution 
            Array4<Real> const& err = error.array(mfi);

            constexpr Real u0 = 1.;
            constexpr Real v0 = 1.;
            constexpr Real visc_coef = 0.001;

            Real omega = pi * pi * visc_coef;
            auto func = 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                AMREX_D_TERM(
                Real x = (i+0.5)*dx[0];,
                Real y = (j+0.5)*dx[1];,
                Real z = (k+0.5)*dx[2];)
                Real exact;
                if (err_comp == SpaceDim || err_comp == SpaceDim+1) 
                {  
                    // pressure 
                    Real t_p = time - 0.5*dt;
                    exact = -0.25 * ( std::cos(twopi*(x-u0*t_p)) + std::cos(twopi*(y-v0*t_p)) ) * std::exp(-4.*omega*t_p);   
                } 
                else if (err_comp == 0) 
                { 
                    // u
                    // TODO: Check this - 2D and 3D cases should be different
                    exact =  u0 - std::cos(pi*(x-u0*time)) * std::sin(pi*(y-v0*time)) * std::exp(-2.*omega*time);
                } 
                else if (err_comp == 1) 
                { 
                    // v
                    // TODO: Check this - 2D and 3D cases should be different
                    exact = v0 + std::sin(pi*(x-u0*time)) * std::cos(pi*(y-v0*time)) * std::exp(-2.*omega*time);
                #if (AMREX_IS_3D)
                } 
                else if (err_comp == 2) 
                { 
                    // w
                    exact = 0.;
                #endif
                }
                err(i,j,k,soln_comp) -= exact;
            };
            ParallelFor(bx, func);
        }
    } 
    else 
        {Abort("Currently TGV is the only problem with an exact solution implemented");}
}
