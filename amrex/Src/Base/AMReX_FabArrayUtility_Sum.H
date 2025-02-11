#ifndef AMREX_FABARRAY_UTILITY_SUM_H_
#define AMREX_FABARRAY_UTILITY_SUM_H_
#include <AMReX_Config.H>

#include <AMReX_FabArray.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Print.H>
#include <limits>

namespace amrex {

// *********************************************************************************************************************
// ReduceSum
// *********************************************************************************************************************

// *********************************************************************************************************************
// ReduceSum with 1 fab and a flag
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceSum (FabArray<FAB> const& fa, int nghost, F&& f) {
    return ReduceSum(fa, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceSum_host (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB::value_type;
    value_type sm = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+:sm)
#endif
    for (MFIter mfi(fa,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        auto const& arr = fa.const_array(mfi);
        sm += f(bx, arr);
    }

    return sm;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceSum_device (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB::value_type;
    value_type sm = 0;

    {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr = fa.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b,arr) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        sm = amrex::get<0>(hv);
    }

    return sm;
}

template <class FAB, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB::value_type>
ReduceSum_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return ReduceSum_host(fa,nghost,std::forward<F>(f));
}

template <class FAB, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB::value_type>
ReduceSum_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa,nghost,f);
    amrex::Abort("ReduceSum: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceSum (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceSum_device(fa, nghost, std::forward<F>(f));
    } else {
        return fudetail::ReduceSum_host_wrapper(fa, nghost, std::forward<F>(f));
    }
}
#else
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceSum (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceSum_host(fa, nghost, std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceSum with 2 fabs and a flag
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           int nghost, F&& f) {
    return ReduceSum(fa1, fa2, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    value_type sm = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+:sm)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        sm += f(bx, arr1, arr2);
    }

    return sm;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    value_type sm = 0;

    BL_PROFILE("ReduceSum_device");

    {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        Gpu::FuseReductionSafeGuard rsg(true);
        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b, arr1, arr2) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        sm = amrex::get<0>(hv);
    }

    return sm;
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceSum_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        IntVect const& nghost, F&& f)
{
    return ReduceSum_host(fa1,fa2,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceSum_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,nghost,f);
    amrex::Abort("ReduceSum: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceSum_device(fa1,fa2,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceSum_host_wrapper(fa1,fa2,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           IntVect const& nghost, F&& f)
{
    return fudetail::ReduceSum_host(fa1,fa2,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceSum with 3 fabs and a flag
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2, FabArray<FAB3> const& fa3,
           int nghost, F&& f)
{
  return ReduceSum(fa1, fa2, fa3, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    value_type sm = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+:sm)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        const auto& arr3 = fa3.const_array(mfi);
        sm += f(bx, arr1, arr2, arr3);
    }

    return sm;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    value_type sm = 0;

    {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            const auto& arr3 = fa3.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b, arr1, arr2, arr3) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        sm = amrex::get<0>(hv);
    }

    return sm;
}

template <class FAB1, class FAB2, class FAB3, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceSum_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceSum_host(fa1,fa2,fa3,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class FAB3, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceSum_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,fa3,nghost,f);
    amrex::Abort("ReduceSum: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceSum_device(fa1,fa2,fa3,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceSum_host_wrapper(fa1,fa2,fa3,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceSum (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceSum_host(fa1,fa2,fa3,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceLogicalAnd
// *********************************************************************************************************************
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalAnd (FabArray<FAB> const& fa, int nghost, F&& f)
{
    return ReduceLogicalAnd(fa, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalAnd_host (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    int r = true;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(&&:r)
#endif
    for (MFIter mfi(fa,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr = fa.const_array(mfi);
        r = r && f(bx, arr);
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalAnd_device (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    int r = true;

    {
        ReduceOps<ReduceOpLogicalAnd> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr = fa.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                int tr = f(b,arr);
                return {tr};
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalAnd_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return ReduceLogicalAnd_host(fa,nghost,std::forward<F>(f));
}

template <class FAB, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalAnd_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa,nghost,f);
    amrex::Abort("ReduceLogicalAnd: Launch Region is off. Device lambda cannot be called by host.");
    return false;
}
}

template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalAnd (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceLogicalAnd_device(fa,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceLogicalAnd_host_wrapper(fa,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalAnd (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceLogicalAnd_host(fa,nghost,std::forward<F>(f));
}
#endif

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalAnd (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  int nghost, F&& f)
{
    return ReduceLogicalAnd(fa1, fa2, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalAnd_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                       IntVect const& nghost, F&& f)
{
    int r = true;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(&&:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        r = r && f(bx, arr1, arr2);
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalAnd_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                         IntVect const& nghost, F&& f)
{
    int r = true;

    {
        ReduceOps<ReduceOpLogicalAnd> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                int tr = f(b, arr1, arr2);
                return {tr};
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalAnd_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  IntVect const& nghost, F&& f)
{
    return ReduceLogicalAnd_host(fa1,fa2,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalAnd_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                               IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,nghost,f);
    amrex::Abort("ReduceLogicalAnd: Luanch Region is off. Device lambda cannot be called by host.");
    return false;
}
}

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalAnd (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceLogicalAnd_device(fa1,fa2,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceLogicalAnd_host_wrapper(fa1,fa2,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalAnd (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  IntVect const& nghost, F&& f)
{
    return fudetail::ReduceLogicalAnd_host(fa1,fa2,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceLogicalOr
// *********************************************************************************************************************
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalOr (FabArray<FAB> const& fa, int nghost, F&& f)
{
    return ReduceLogicalOr(fa, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalOr_host (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    int r = false;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(||:r)
#endif
    for (MFIter mfi(fa,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr = fa.const_array(mfi);
        r = r || f(bx, arr);
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalOr_device (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    int r = false;

    {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr = fa.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                int tr = f(b, arr);
                return {tr};
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalOr_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return ReduceLogicalOr_host(fa,nghost,std::forward<F>(f));
}

template <class FAB, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalOr_host (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa,nghost,f);
    amrex::Abort("ReduceLogicalOr: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalOr (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceLogicalOr_device(fa,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceLogicalOr_host_wrapper(fa,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
bool
ReduceLogicalOr (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceLogicalOr_host(fa,nghost,std::forward<F>(f));
}
#endif

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalOr (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                 int nghost, F&& f)
{
    return ReduceLogicalOr(fa1, fa2, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalOr_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                      IntVect const& nghost, F&& f)
{
    int r = false;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(||:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        r = r || f(bx, arr1, arr2);
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalOr_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        IntVect const& nghost, F&& f)
{
    int r = false;

    {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                int tr = f(b, arr1, arr2);
                return {tr};
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalOr_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                 IntVect const& nghost, F&& f)
{
    return fudetail::ReduceLogicalOr_host(fa1,fa2,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, bool>
ReduceLogicalOr_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                              IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,nghost,f);
    amrex::Abort("ReeuceLogicalOr: Launch Region is off. Device lambda cannot be called by host.");
    return false;
}
}

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalOr (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                 IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceLogicalOr_device(fa1,fa2,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceLogicalOr_host_wrapper(fa1,fa2,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
bool
ReduceLogicalOr (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                 IntVect const& nghost, F&& f)
{
    return fudetail::ReduceLogicalOr_host(fa1,fa2,nghost,std::forward<F>(f));
}
#endif

}

#endif
