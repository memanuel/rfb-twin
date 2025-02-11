#ifndef AMREX_FABARRAY_UTILITY_MAX_H_
#define AMREX_FABARRAY_UTILITY_MAX_H_
#include <AMReX_Config.H>

#include <AMReX_FabArray.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Print.H>
#include <limits>

namespace amrex {

// *********************************************************************************************************************
// ReduceMax - One FabArray
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceMax (FabArray<FAB> const& fa, int nghost, F&& f)
{
    return ReduceMax(fa, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceMax_host (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:r)
#endif
    for (MFIter mfi(fa,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr = fa.const_array(mfi);
        r = std::max(r, f(bx, arr));
    }
    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceMax_device (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

    {
        ReduceOps<ReduceOpMax> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr = fa.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b, arr) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB::value_type>
ReduceMax_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return ReduceMax_host(fa,nghost,std::forward<F>(f));
}

template <class FAB, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB::value_type>
ReduceMax_host_wrapper (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa,nghost,f);
    amrex::Abort("ReduceMax: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceMax (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceMax_device(fa,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceMax_host_wrapper(fa,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB>::value> >
typename FAB::value_type
ReduceMax (FabArray<FAB> const& fa, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceMax - Two FabArrays
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2, int nghost, F&& f)
{
    return ReduceMax(fa1, fa2, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        r = std::max(r, f(bx, arr1, arr2));
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

    {
        ReduceOps<ReduceOpMax> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

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
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        IntVect const& nghost, F&& f)
{
    return ReduceMax_host(fa1,fa2,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,nghost,f);
    amrex::Abort("ReduceMax: Launch Region is off. Device lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceMax_device(fa1,fa2,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceMax_host_wrapper(fa1,fa2,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceMax - Three FabArrays
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2, FabArray<FAB3> const& fa3,
           int nghost, F&& f)
{
    return ReduceMax(fa1, fa2, fa3, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        const auto& arr3 = fa3.const_array(mfi);
        r = std::max(r, f(bx, arr1, arr2, arr3));
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

    {
        ReduceOps<ReduceOpMax> reduce_op;
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
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class FAB3, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class FAB3, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,fa3,nghost,f);
    amrex::Abort("ReduceMax: Launch Region is off. Device lambda lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceMax_device(fa1,fa2,fa3,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceMax_host_wrapper(fa1,fa2,fa3,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class FAB3, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceMax - Four FabArrays
template <class FAB1, class FAB2, class FAB3, class FAB4, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2, 
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, int nghost, F&& f)
{
    return ReduceMax(fa1, fa2, fa3, fa4, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class FAB4, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        const auto& arr3 = fa3.const_array(mfi);
        const auto& arr4 = fa4.const_array(mfi);
        r = std::max(r, f(bx, arr1, arr2, arr3, arr4));
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class FAB4, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                  IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

    {
        ReduceOps<ReduceOpMax> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            const auto& arr3 = fa3.const_array(mfi);
            const auto& arr4 = fa4.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b, arr1, arr2, arr3, arr4) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                        IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,fa4,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                        IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,fa3,fa4,nghost,f);
    amrex::Abort("ReduceMax: Launch Region is off. Device lambda lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
           IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceMax_device(fa1,fa2,fa3,fa4,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceMax_host_wrapper(fa1,fa2,fa3,fa4,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class FAB3, class FAB4, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
           IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,fa4,nghost,std::forward<F>(f));
}
#endif

// *********************************************************************************************************************
// ReduceMax - Five FabArrays
template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2, 
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
           FabArray<FAB5> const& fa5, int nghost, F&& f)
{
    return ReduceMax(fa1, fa2, fa3, fa4, fa5, IntVect(nghost), std::forward<F>(f));
}

namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_host (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                FabArray<FAB5> const& fa5, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:r)
#endif
    for (MFIter mfi(fa1,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(nghost);
        const auto& arr1 = fa1.const_array(mfi);
        const auto& arr2 = fa2.const_array(mfi);
        const auto& arr3 = fa3.const_array(mfi);
        const auto& arr4 = fa4.const_array(mfi);
        const auto& arr5 = fa5.const_array(mfi);
        r = std::max(r, f(bx, arr1, arr2, arr3, arr4, arr5));
    }

    return r;
}
}

#ifdef AMREX_USE_GPU
namespace fudetail {
template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax_device (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                  FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                  FabArray<FAB5> const& fa5, IntVect const& nghost, F&& f)
{
    using value_type = typename FAB1::value_type;
    constexpr value_type value_lowest = std::numeric_limits<value_type>::lowest();
    value_type r = value_lowest;

    {
        ReduceOps<ReduceOpMax> reduce_op;
        ReduceData<value_type> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(fa1); mfi.isValid(); ++mfi)
        {
            const Box& bx = amrex::grow(mfi.validbox(),nghost);
            const auto& arr1 = fa1.const_array(mfi);
            const auto& arr2 = fa2.const_array(mfi);
            const auto& arr3 = fa3.const_array(mfi);
            const auto& arr4 = fa4.const_array(mfi);
            const auto& arr5 = fa5.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (Box const& b) -> ReduceTuple
            {
                return { f(b, arr1, arr2, arr3, arr4, arr5) };
            });
        }

        ReduceTuple hv = reduce_data.value();
        r = amrex::get<0>(hv);
    }

    return r;
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F>
std::enable_if_t<!amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                        FabArray<FAB5> const& fa5, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,fa4,fa5,nghost,std::forward<F>(f));
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F>
std::enable_if_t<amrex::DefinitelyNotHostRunnable<F>::value, typename FAB1::value_type>
ReduceMax_host_wrapper (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
                        FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
                        FabArray<FAB5> const& fa5, IntVect const& nghost, F&& f)
{
    amrex::ignore_unused(fa1,fa2,fa3,fa4,fa5,nghost,f);
    amrex::Abort("ReduceMax: Launch Region is off. Device lambda lambda cannot be called by host.");
    return 0;
}
}

template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
           FabArray<FAB4> const& fa5, IntVect const& nghost, F&& f)
{
    if (Gpu::inLaunchRegion()) {
        return fudetail::ReduceMax_device(fa1,fa2,fa3,fa4,fa5,nghost,std::forward<F>(f));
    } else {
        return fudetail::ReduceMax_host_wrapper(fa1,fa2,fa3,fa4,fa5,nghost,std::forward<F>(f));
    }
}
#else
template <class FAB1, class FAB2, class FAB3, class FAB4, class FAB5, class F,
          class bar = std::enable_if_t<IsBaseFab<FAB1>::value> >
typename FAB1::value_type
ReduceMax (FabArray<FAB1> const& fa1, FabArray<FAB2> const& fa2,
           FabArray<FAB3> const& fa3, FabArray<FAB4> const& fa4, 
           FabArray<FAB5> const& fa5, IntVect const& nghost, F&& f)
{
    return fudetail::ReduceMax_host(fa1,fa2,fa3,fa4,fa5,nghost,std::forward<F>(f));
}
#endif

}

#endif
