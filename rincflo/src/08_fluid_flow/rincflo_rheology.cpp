#include <rincflo.H>
#include <rincflo_derive_K.H>

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real expterm (amrex::Real nu) noexcept
{
    return (nu < 1.e-9) ? (1.0-0.5*nu+nu*nu*(1.0/6.0)-(nu*nu*nu)*(1./24.))  : -std::expm1(-nu)/nu;
}

// *********************************************************************************************************************
struct NonNewtonianViscosity
{
    Rincflo::FluidModel fluid_model;
    amrex::Real mu, n_flow, tau_0, eta_0, papa_reg;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real sr) const noexcept 
    {
        switch (fluid_model)
        {
            case Rincflo::FluidModel::powerlaw:
            {
                return mu * std::pow(sr,n_flow-1.0);
            }
            case Rincflo::FluidModel::Bingham:
            {
                return mu + tau_0 * expterm(sr/papa_reg) / papa_reg;
            }
            case Rincflo::FluidModel::HerschelBulkley:
            {
                return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr/papa_reg)/papa_reg;
            }
            case Rincflo::FluidModel::deSouzaMendesDutra:
            {
                return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr*(eta_0/tau_0))*(eta_0/tau_0);
            }
            default:
            {
                return mu;
            }
        };
    }
};

}   // end anonymous namespace

// *********************************************************************************************************************
void Rincflo::ComputeViscosity (
    Vector<MultiFab*> const& vel_eta,
    Vector<MultiFab*> const& rho,
    Vector<MultiFab*> const& vel,
    Real time, int nghost)
{
    for (int lev = 0; lev <= finest_level; ++lev) 
        {ComputeViscosityAtLevel(lev, vel_eta[lev], rho[lev], vel[lev], geom[lev], time, nghost);}
}

// *********************************************************************************************************************
void Rincflo::ComputeViscosityAtLevel (
    int lev,
    MultiFab* vel_eta,
    MultiFab* /*rho*/,
    MultiFab* vel,
    Geometry& lev_geom,
    Real /*time*/, int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
        {vel_eta->setVal(m_mu, 0, 1, nghost);}
    else
    {
        NonNewtonianViscosity non_newtonian_viscosity;
        non_newtonian_viscosity.fluid_model = m_fluid_model;
        non_newtonian_viscosity.mu = m_mu;
        non_newtonian_viscosity.n_flow = m_n_0;
        non_newtonian_viscosity.tau_0 = m_tau_0;
        non_newtonian_viscosity.eta_0 = m_eta_0;
        non_newtonian_viscosity.papa_reg = m_papa_reg;

        #if (AMREX_USE_EB)
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
        #endif

        AMREX_D_TERM(
        Real idx = 1.0 / lev_geom.CellSize(0);,
        Real idy = 1.0 / lev_geom.CellSize(1);,
        Real idz = 1.0 / lev_geom.CellSize(2);)

        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(*vel_eta,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.growntilebox(nghost);
            Array4<Real> const& eta_arr = vel_eta->array(mfi);
            Array4<Real const> const& vel_arr = vel->const_array(mfi);
            #if (AMREX_USE_EB)
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            bool is_boundary = fabIsBoundary(typ);
            if (typ == FabType::covered)
            {
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    eta_arr(i,j,k) = 0.0;
                };
                ParallelFor(bx, func);
            }
            else if (is_boundary)
            {
                auto const& flag_arr = flag_fab.const_array();
                const bool cover_multiple_cuts = m_cover_multiple_cuts;
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real sr = rincflo_strainrate_eb(
                        i,j,k, AMREX_D_DECL(idx,idy,idz), vel_arr, cover_multiple_cuts, flag_arr(i,j,k));
                    eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                };
                ParallelFor(bx, func);
            }
            else
            #endif
            {
                auto func = 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real sr = rincflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);
                    eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                };
                ParallelFor(bx, func);
            }
        }
    }
}

// *********************************************************************************************************************
void Rincflo::ComputeSpeciesDiffCoeff (Vector<MultiFab*> const& tra_eta, int nghost)
{
    for (auto mf : tra_eta) 
    {
        for (int n = 0; n < m_nspec; ++n) 
            {mf->setVal(m_D_s[n], n, 1, nghost);}
    }
}
