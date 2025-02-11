#ifndef AMREX_HYPREABECLAP3INV_H_
#define AMREX_HYPREABECLAP3INV_H_
#include <AMReX_Config.H>
#include <AMReX_Hypre.H>

#include <AMReX_iMultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_HypreIJIface.H>

#include <algorithm>

namespace amrex
{

class HypreABecLap3Inv
    : public Hypre
{
public:

    HypreABecLap3Inv (const BoxArray& grids, const DistributionMapping& dmap,
                   const Geometry& geom_, MPI_Comm comm_,
		      const iMultiFab* overset_mask = nullptr, const bool invert_eb = false);
    virtual ~HypreABecLap3Inv ();

    virtual void solve (MultiFab& soln, const MultiFab& rhs, Real rel_tol, Real abs_tol, 
                        int max_iter, const BndryData& bndry, int max_bndry_order) final;

#ifdef AMREX_USE_EB
    void setEBDirichlet (MultiFab const* beb) {  m_eb_b_coeffs = beb;   }
    void setInvertEB (bool a_invert_eb) {  m_invert_eb = a_invert_eb;   }
#endif

private :
    std::unique_ptr<HypreIJIface> hypre_ij;

    // Non-owning references to hypre matrix, rhs, and solution data
    HYPRE_IJMatrix A = NULL;
    HYPRE_IJVector b = NULL;
    HYPRE_IJVector x = NULL;

    LayoutData<HYPRE_Int> ncells_grid;
    FabArray<BaseFab<HYPRE_Int> > cell_id;
    FabArray<BaseFab<HYPRE_Int> > cell_id_vec;

    MultiFab const* m_eb_b_coeffs = nullptr;

    iMultiFab const* m_overset_mask = nullptr;

    bool m_invert_eb = false;
  
public: // for CUDA
    void prepareSolver ();
    void getSolution (MultiFab& soln);
    void loadVectors (MultiFab& soln, const MultiFab& rhs);
};

}

#endif
