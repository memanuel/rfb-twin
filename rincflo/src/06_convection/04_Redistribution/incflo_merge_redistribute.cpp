#if (AMREX_USE_EB)

#include <Redistribution.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

// *********************************************************************************************************************
void redistribution::merge_redistribute ( 
    Box const& bx, int ncomp,
    Array4<Real> const& dUdt_out,
    Array4<Real> const& dUdt_in,
    Array4<Real const> const& vfrac,
    Array4<int> const& itracker,
    Geometry& lev_geom)
{
    bool debug_print = false;

    const Box domain = lev_geom.Domain();

    // Note that itracker has {4 in 2D, 8 in 3D} components and all are initialized to zero
    // We will add to the first component every time this cell is included in a merged neighborhood,
    //    either by merging or being merged
    //
    // In 2D, we identify the cells in the remaining three components with the following ordering
    //
    // ^  6 7 8
    // |  4   5
    // j  1 2 3
    //   i --->
    // 
    // In 3D, We identify the cells in the remaining three components with the following ordering
    // 
    //    at k-1   |   at k  |   at k+1 
    // 
    // ^  15 16 17 |  6 7 8  |  24 25 26
    // |  12 13 14 |  4   5  |  21 22 23
    // j  9  10 11 |  1 2 3  |  18 19 20
    //   i --->
    // 
    // Note the first component of each of these arrays should never be used
    // 
    #if (AMREX_SPACEDIM == 2)
    Array<int,9> imap{0,-1, 0, 1,-1, 1,-1, 0, 1};
    Array<int,9> jmap{0,-1,-1,-1, 0, 0, 1, 1, 1};
    Array<int,9> kmap{0, 0, 0, 0, 0, 0, 0, 0, 0};
    #else
    Array<int,27>    imap{0,-1, 0, 1,-1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1};
    Array<int,27>    jmap{0,-1,-1,-1, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1};
    Array<int,27>    kmap{0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    #endif

    //  if (debug_print)
    //      Print() << " IN MERGE_REDISTRIBUTE DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    const Real small_norm = 1.e-8;

    AMREX_D_TERM(
    const auto& is_periodic_x = lev_geom.isPeriodic(0);,
    const auto& is_periodic_y = lev_geom.isPeriodic(1);,
    const auto& is_periodic_z = lev_geom.isPeriodic(2);)

    auto func_bx_1 = 
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.0)
        {
            for (int n = 0; n < ncomp; n++)
                {dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);}
        } 
        else 
        {
            // We shouldn't need to do this but just in case ...
            for (int n = 0; n < ncomp; n++)
                {dUdt_out(i,j,k,n) = 1.e100;}
        } 
    };
    ParallelFor(bx, func_bx_1);

    auto func_bx_2 = 
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
       if (vfrac(i,j,k) > 0.0)
       {
           if (itracker(i,j,k,0) > 0)
           {
               for (int n = 0; n < ncomp; n++)
               {
                   Real sum_vol = vfrac(i,j,k);
                   Real sum_upd = vfrac(i,j,k) * dUdt_in(i,j,k,n);
                   for (int i_nbor = 1; i_nbor <= itracker(i,j,k,0); i_nbor++)
                   {
                       sum_upd +=   vfrac(i+imap[itracker(i,j,k,i_nbor)],
                                          j+jmap[itracker(i,j,k,i_nbor)],
                                          k+kmap[itracker(i,j,k,i_nbor)]) *
                                  dUdt_in(i+imap[itracker(i,j,k,i_nbor)], 
                                          j+jmap[itracker(i,j,k,i_nbor)],
                                          k+kmap[itracker(i,j,k,i_nbor)],n);
                       sum_vol +=   vfrac(i+imap[itracker(i,j,k,i_nbor)],
                                          j+jmap[itracker(i,j,k,i_nbor)],
                                          k+kmap[itracker(i,j,k,i_nbor)]); 
                   } 

                   if (sum_vol < 0.5)
                   {
//                    Print() << "SUM_VOL STILL TOO SMALL at " << 
//                        IntVect(AMREX_D_DECL(i,j,k)) << " " << sum_vol << std::endl; 
                      Abort();
                   }

                   Real avg_update = sum_upd / sum_vol;
    
                   dUdt_out(i,j,k,n) = avg_update;
               }
           }
       }
    };
    ParallelFor(bx, func_bx_2);
}
#endif
