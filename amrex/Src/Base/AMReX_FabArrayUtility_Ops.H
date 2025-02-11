#ifndef AMREX_FABARRAY_UTILITY_OPS_H_
#define AMREX_FABARRAY_UTILITY_OPS_H_
#include <AMReX_Config.H>

#include <AMReX_FabArray.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Print.H>
#include <limits>

namespace amrex {

// *********************************************************************************************************************
// Operations: printCell, Add, Copy, Subtract, 
// *********************************************************************************************************************

// *********************************************************************************************************************
// printCell
template <class FAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
printCell (FabArray<FAB> const& mf, const IntVect& cell, int comp = -1,
           const IntVect& ng = IntVect::TheZeroVector())
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box& bx = amrex::grow(mfi.validbox(), ng);
        if (bx.contains(cell)) {
            if (comp >= 0) {
                amrex::AllPrint().SetPrecision(17) << " At cell " << cell << " in Box " << bx
                                                   << ": " << mf[mfi](cell, comp) << std::endl;
            } else {
                std::ostringstream ss;
                ss.precision(17);
                const int ncomp = mf.nComp();
                for (int i = 0; i < ncomp-1; ++i)
                {
                    ss << mf[mfi](cell,i) << ", ";
                }
                ss << mf[mfi](cell,ncomp-1);
                amrex::AllPrint() << " At cell " << cell << " in Box " << bx
                                  << ": " << ss.str() << std::endl;
            }
        }
    }
}

// *********************************************************************************************************************
// Copy
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    Copy(dst,src,srccomp,dstcomp,numcomp,IntVect(nghost));
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const srcFab = src.array(mfi);
            auto       dstFab = dst.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                dstFab(i,j,k,dstcomp+n) = srcFab(i,j,k,srccomp+n);
            });
        }
    }
}

// *********************************************************************************************************************
// Add
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Add (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    Add(dst,src,srccomp,dstcomp,numcomp,IntVect(nghost));
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Add (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const srcFab = src.array(mfi);
            auto       dstFab = dst.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                dstFab(i,j,k,n+dstcomp) += srcFab(i,j,k,n+srccomp);
            });
        }
    }
}

// *********************************************************************************************************************
// Subtract
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Subtract (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    Subtract(dst,src,srccomp,dstcomp,numcomp,nghost);
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Subtract (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const srcFab = src.array(mfi);
            auto       dstFab = dst.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                dstFab(i,j,k,n+dstcomp) -= srcFab(i,j,k,n+srccomp);
            });
        }
    }
}

// *********************************************************************************************************************
// Multiply
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Multiply (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    Multiply(dst,src,srccomp,dstcomp,numcomp,IntVect(nghost));
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Multiply (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const srcFab = src.array(mfi);
            auto       dstFab = dst.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                dstFab(i,j,k,n+dstcomp) *= srcFab(i,j,k,n+srccomp);
            });
        }
    }
}

// *********************************************************************************************************************
// Divide
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Divide (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
{
    Divide(dst,src,srccomp,dstcomp,numcomp,IntVect(nghost));
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Divide (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const srcFab = src.array(mfi);
            auto       dstFab = dst.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                dstFab(i,j,k,n+dstcomp) /= srcFab(i,j,k,n+srccomp);
            });
        }
    }
}

// *********************************************************************************************************************
// Abs
template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Abs (FabArray<FAB>& fa, int icomp, int numcomp, int nghost)
{
    Abs(fa,icomp,numcomp,IntVect(nghost));
}

template <class FAB,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
Abs (FabArray<FAB>& fa, int icomp, int numcomp, const IntVect& nghost)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fa,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        if (bx.ok())
        {
            auto const& fab = fa.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, numcomp, i, j, k, n,
            {
                fab(i,j,k,n+icomp) = amrex::Math::abs(fab(i,j,k,n+icomp));
            });
        }
    }
}

// *********************************************************************************************************************
// Miscellaneous operations: prefetchToHost, prefetchToDevice, OverrideSync, dtoh_memcpy, indexFromValue...
// *********************************************************************************************************************

// *********************************************************************************************************************
// prefetchToHost
template <class FAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
prefetchToHost (FabArray<FAB> const& fa, const bool synchronous = true)
{
#ifdef AMREX_USE_GPU
    if (fa.arena()->isManaged()) {
        for (MFIter mfi(fa, MFItInfo().SetDeviceSync(synchronous)); mfi.isValid(); ++mfi) {
            fa.prefetchToHost(mfi);
        }
    }
#else
    amrex::ignore_unused(fa,synchronous);
#endif
}

// *********************************************************************************************************************
// prefetchToDevice
template <class FAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
void
prefetchToDevice (FabArray<FAB> const& fa, const bool synchronous = true)
{
#ifdef AMREX_USE_GPU
    if (fa.arena()->isManaged()) {
        for (MFIter mfi(fa, MFItInfo().SetDeviceSync(synchronous)); mfi.isValid(); ++mfi) {
            fa.prefetchToDevice(mfi);
        }
    }
#else
    amrex::ignore_unused(fa,synchronous);
#endif
}

// *********************************************************************************************************************
// OverrideSync
template <class FAB, class IFAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value
                                                               && IsBaseFab<IFAB>::value> >
void
OverrideSync (FabArray<FAB> & fa, FabArray<IFAB> const& msk, const Periodicity& period)
{
    BL_PROFILE("OverrideSync()");

    if (fa.ixType().cellCentered()) return;

    const int ncomp = fa.nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fa,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto fab = fa.array(mfi);
        auto const ifab = msk.array(mfi);
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FUSIBLE ( bx, ncomp, i, j, k, n,
        {
            if (!ifab(i,j,k)) fab(i,j,k,n) = 0;
        });
    }

    FabArray<FAB> tmpmf(fa.boxArray(), fa.DistributionMap(), ncomp, 0,
                        MFInfo(), fa.Factory());
    tmpmf.setVal(0);
    tmpmf.ParallelCopy(fa, period, FabArrayBase::ADD);

    amrex::Copy(fa, tmpmf, 0, 0, ncomp, 0);
}

// *********************************************************************************************************************
// dtoh_memcpy
template <class FAB, class foo = std::enable_if_t<IsBaseFab<FAB>::value> >
void
dtoh_memcpy (FabArray<FAB>& dst, FabArray<FAB> const& src,
             int scomp, int dcomp, int ncomp)
{
    AMREX_ASSERT(isMFIterSafe(dst, src));
    AMREX_ASSERT(dst.nGrowVect() == src.nGrowVect());
#ifdef AMREX_USE_GPU
    for (MFIter mfi(dst); mfi.isValid(); ++mfi) {
        void* pdst = dst[mfi].dataPtr(dcomp);
        void const* psrc = src[mfi].dataPtr(scomp);
        Gpu::dtoh_memcpy_async(pdst, psrc, dst[mfi].nBytes(mfi.fabbox(), ncomp));
    }
#else
    Copy(dst, src, scomp, dcomp, ncomp, dst.nGrowVect());
#endif
}

template <class FAB, class foo = std::enable_if_t<IsBaseFab<FAB>::value> >
void
dtoh_memcpy (FabArray<FAB>& dst, FabArray<FAB> const& src)
{
    dtoh_memcpy(dst, src, 0, 0, dst.nComp());
}

template <class FAB, class foo = std::enable_if_t<IsBaseFab<FAB>::value> >
void
htod_memcpy (FabArray<FAB>& dst, FabArray<FAB> const& src,
             int scomp, int dcomp, int ncomp)
{
    AMREX_ASSERT(isMFIterSafe(dst, src));
    AMREX_ASSERT(dst.nGrowVect() == src.nGrowVect());
#ifdef AMREX_USE_GPU
    for (MFIter mfi(dst); mfi.isValid(); ++mfi) {
        void* pdst = dst[mfi].dataPtr(dcomp);
        void const* psrc = src[mfi].dataPtr(scomp);
        Gpu::htod_memcpy_async(pdst, psrc, dst[mfi].nBytes(mfi.fabbox(), ncomp));
    }
#else
    Copy(dst, src, scomp, dcomp, ncomp, dst.nGrowVect());
#endif
}

// *********************************************************************************************************************
// htod_memcpy
template <class FAB, class foo = std::enable_if_t<IsBaseFab<FAB>::value> >
void
htod_memcpy (FabArray<FAB>& dst, FabArray<FAB> const& src)
{
    htod_memcpy(dst, src, 0, 0, dst.nComp());
}

// *********************************************************************************************************************
// indexFromValue
template <class FAB, class foo = std::enable_if_t<IsBaseFab<FAB>::value> >
IntVect
indexFromValue (FabArray<FAB> const& mf, int comp, IntVect const& nghost,
                typename FAB::value_type value)
{
    IntVect loc;

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion())
    {
        amrex::Gpu::Buffer<int> aa({0,AMREX_D_DECL(0,0,0)});
        int* p = aa.data();
        // This is a device ptr to 1+AMREX_SPACEDIM int zeros.
        // The first is used as an atomic bool and the others for intvect.
        for (MFIter mfi(mf,MFItInfo().SetDeviceSync(false)); mfi.isValid(); ++mfi) {
            const Box& bx = amrex::grow(mfi.validbox(), nghost);
            auto const& arr = mf.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int* flag = p;
                if (*flag == 0) {
                    if (arr(i,j,k,comp) == value) {
                        if (Gpu::Atomic::Exch(flag,1) == 0) {
                            AMREX_D_TERM(p[1] = i;,
                                         p[2] = j;,
                                         p[3] = k;);
                        }
                    }
                }
            });
        }
        int const* tmp = aa.copyToHost();
        AMREX_D_TERM(loc[0] = tmp[1];,
                     loc[1] = tmp[2];,
                     loc[2] = tmp[3];);
    }
    else
#endif
    {
        bool f = false;
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {
            IntVect priv_loc = IntVect::TheMinVector();
            for (MFIter mfi(mf,true); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.growntilebox(nghost);
                auto const& fab = mf.const_array(mfi);
                AMREX_LOOP_3D(bx, i, j, k,
                {
                    if (fab(i,j,k,comp) == value) priv_loc = IntVect(AMREX_D_DECL(i,j,k));
                });
            }

            if (priv_loc.allGT(IntVect::TheMinVector())) {
                bool old;
// we should be able to test on _OPENMP < 201107 for capture (version 3.1)
// but we must work around a bug in gcc < 4.9
#if defined(AMREX_USE_OMP) && defined(_OPENMP) && _OPENMP < 201307 // OpenMP 4.0
#pragma omp critical (amrex_indexfromvalue)
#elif defined(AMREX_USE_OMP)
#pragma omp atomic capture
#endif
                {
                    old = f;
                    f = true;
                }

                if (old == false) loc = priv_loc;
            }
        }
    }

    return loc;
}

}

#endif
