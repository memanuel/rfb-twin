#ifndef AMREX_EBINVINTERPOLATER_H_
#define AMREX_EBINVINTERPOLATER_H_
#include <AMReX_Config.H>
#include <AMReX_Interpolater.H>

namespace amrex {

  class EBInvCellConservativeLinear
    : public CellConservativeLinear
  {
  public:
    
    explicit EBInvCellConservativeLinear (bool do_linear_limiting_ = true);

    virtual ~EBInvCellConservativeLinear () override;

    virtual void interp (const FArrayBox& crse,
                         int              crse_comp,
                         FArrayBox&       fine,
                         int              fine_comp,
                         int              ncomp,
                         const Box&       fine_region,
                         const IntVect&   ratio,
                         const Geometry&  crse_geom,
                         const Geometry&  fine_geom,
                         Vector<BCRec> const& bcr,
                         int              actual_comp,
                         int              actual_state,
                         RunOn            gpu_or_cpu) override;

  };

  extern EBInvCellConservativeLinear  ebinv_lincc_interp;
  extern EBInvCellConservativeLinear  ebinv_cell_cons_interp;

}

#endif
