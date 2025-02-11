#ifndef AMREX_MLMGINV_2D_K_H_
#define AMREX_MLMGINV_2D_K_H_
#include <AMReX_Config.H>

namespace amrex {

#ifdef AMREX_USE_EB
template <int R>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlmg_eb_cc_interp_r (Box const& bx, Array4<Real> const& ff, Array4<Real const> const& cc,
                          Array4<EBCellFlag const> const& flag, int nc, bool invert_eb) noexcept
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < nc; ++n) {
        for (int j = lo.y; j <= hi.y; ++j) {
            const int jc = j/R;
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                const int ic = i/R;
                if ( (flag(i,j,0).isCovered() && !invert_eb) || (flag(i,j,0).isRegular() && invert_eb) ) {
                    ff(i,j,0,n) = 0.0;
                } else {
                    ff(i,j,0,n) = cc(ic,jc,0,n);
                }
            }
        }
    }
}
#endif

}

#endif
