// rincflo_amrex.H
// Fragment of Rincflo.H
// Methods in Rincflo for dealing with AMReX implementation details
#pragma once
#include <rincflo_macros.H>

private:

FabFactory<FArrayBox> const&
Factory (int lev) const noexcept 
{
    return *m_factory[lev];
}

#if (AMREX_USE_EB)
EBFArrayBoxFactory const&
EBFactory (int lev) const noexcept 
{
    return static_cast<EBFArrayBoxFactory const&>(*m_factory[lev]);
}
#endif

// Number of ghost cells for field arrays.
int nghost_state () const 
{
    if (has_eb())
        {return 4;}
    else
        {return (m_advection_type != "MOL") ? 3 : 2;}
}

// For Godunov, we need 1 ghost cell in addition to the Box we are filling
// For MOL    , we need 0 ghost cells
int nghost_force () const 
{
    if (m_advection_type == "MOL") 
        {return 0;}
    else 
        {return 1;}
} 

int nghost_mac () const 
{
    #if (AMREX_USE_EB)
    if (has_eb()) 
        {return (m_advection_type == "Godunov") ? 4 : 3;}
    // If the EB is all regular or we are not using EB, then it depends on whether we are using MOL or Godunov
    #endif
    return (m_advection_type == "Godunov") ? 1 : 0;
}

#if (AMREX_USE_EB)
constexpr int nghost_eb_basic () const 
    {return 4;}

constexpr int nghost_eb_volume () const 
    {return 4;}

constexpr int nghost_eb_full () const 
    {return 4;}

// The number of ghost cells of needed for the embedded boundary in various modes
constexpr Vector<int> ngrow() const 
    {return amrex::Vector<int> {nghost_eb_basic(), nghost_eb_volume(), nghost_eb_full()};}

#endif

inline bool fabIsLiquid(FabType typ) const
    {return typ == FabType::regular;}

inline bool fabIsBoundary(FabType typ) const
    {return typ == FabType::singlevalued || (m_cover_multiple_cuts && typ == FabType::multivalued);}

inline bool fabIsSolid(FabType typ) const
    {return typ == FabType::covered;}

private:

// Fill fine patches
void fillpatch_velocity (int lev, Real time, MultiFab& vel, int ng);
void fillpatch_density  (int lev, Real time, MultiFab& density, int ng);
void fillpatch_conc     (int lev, Real time, MultiFab& conc, int ng);
void fillpatch_gradp    (int lev, Real time, MultiFab& gradp, int ng);
void fillpatch_epotL    (int lev, Real time, MultiFab& epotL, int ng);
void fillpatch_epotS    (int lev, Real time, MultiFab& epotS, int ng);
void fillpatch_overpot  (int lev, Real time, MultiFab& overpot, int ng);
void fillpatch_force    (Real time, Vector<MultiFab*> const& force, int ng);

// Fill coarse patches
void fillcoarsepatch_velocity   (int lev, Real time, MultiFab& vel, int ng);
void fillcoarsepatch_density    (int lev, Real time, MultiFab& density, int ng);
void fillcoarsepatch_conc       (int lev, Real time, MultiFab& conc, int ng);
void fillcoarsepatch_gradp      (int lev, Real time, MultiFab& , int ng);
void fillcoarsepatch_epotL      (int lev, Real time, MultiFab& epotL, int ng);
void fillcoarsepatch_epotS      (int lev, Real time, MultiFab& epotS, int ng);

// Fill boundary conditions
void fillphysbc_velocity    (int lev, Real time, MultiFab& vel, int ng);
void fillphysbc_density     (int lev, Real time, MultiFab& density, int ng);
void fillphysbc_conc        (int lev, Real time, MultiFab& conc, int ng);
void fillphysbc_epotL       (int lev, Real time, MultiFab& epotL, int ng);
void fillphysbc_epotS       (int lev, Real time, MultiFab& epotS, int ng);
