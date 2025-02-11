#ifndef AMREX_MLMGINV_3D_K_H_
#define AMREX_MLMGINV_3D_K_H_
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
        for (int k = lo.z; k <= hi.z; ++k) {
            const int kc = k/R;
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = j/R;
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = i/R;
                    if ( (flag(i,j,k).isCovered() && !invert_eb) || (flag(i,j,k).isRegular() && invert_eb)) {
                        ff(i,j,k,n) = 0.0;
                    } else {
                        ff(i,j,k,n) = cc(ic,jc,kc,n);
                    }
                }
            }
        }
    }
}
#endif


}
#endif
