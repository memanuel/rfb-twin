#include <rincflo.H>
#include <rincflo_derive_K.H>

// Alias to positive infinity for use in this file; used for initialization in min and max
constexpr Real inf = std::numeric_limits<Real>::infinity();

// *********************************************************************************************************************
Real Rincflo::VolumeWeightedSum (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::VolumeWeightedSum"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    const auto& volume = m_leveldata[lev]->volume;
    MultiFab product;
    product.define(mf.boxArray(), mf.DistributionMap(), 1, 0);
    MultiFab::Copy(product, mf, comp, 0, 1, 0);
    MultiFab::Multiply(product, volume, 0, 0, 1, 0);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    return product.sum();
    #undef FUNC_NAME
}

// *********************************************************************************************************************
Real Rincflo::AreaWeightedSum (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::AreaWeightedSum"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    const auto& area = m_leveldata[lev]->area;
    MultiFab product;
    product.define(mf.boxArray(), mf.DistributionMap(), 1, 0);
    MultiFab::Copy(product, mf, comp, 0, 1, 0);
    MultiFab::Multiply(product, area, 0, 0, 1, 0);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    return product.sum();
    #undef FUNC_NAME
}

// *********************************************************************************************************************
Real Rincflo::VolumeWeightedAvg (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::VolumeWeightedAvg"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Real numerator = VolumeWeightedSum(lev, mf, comp, local);
    const auto& volume = m_leveldata[lev]->volume;
    Real denominator = volume.sum(0, false);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    return numerator / denominator;
    #undef FUNC_NAME
}

// *********************************************************************************************************************
Real Rincflo::AreaWeightedAvg (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::AreaWeightedAvg"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    Real numerator = AreaWeightedSum(lev, mf, comp, local);
    const auto& area = m_leveldata[lev]->area;
    Real denominator = area.sum(0, false);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    return numerator / denominator;
    #undef FUNC_NAME
}

// *********************************************************************************************************************
Real Rincflo::SumOverBoundary (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumOverBoundary"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto sum_func = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real sm = 0.0;
        auto func_b =
        [=, &sm] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {sm += mf_arr(i,j,k,comp);}
        };
        Loop(bx, func_b);
        return sm;
    };
    Real sum = ReduceSum(mf, flags, 0, sum_func);

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
Real Rincflo::MinOverBoundary (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MinOverBoundary"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto func_min = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_min = inf;
        auto func_b = 
        [=, &the_min] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {the_min = min(the_min, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_min;
    };
    Real the_min = ReduceMin(mf, flags, 0, func_min);

    if (!local)
        {ReduceRealMin(the_min);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_min;
}

// *********************************************************************************************************************
Real Rincflo::MaxOverBoundary (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MaxOverBoundary"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto func_max = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_max = -inf;
        auto func_b = 
        [=, &the_max] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {the_max = max(the_max, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_max;
    };
    Real the_max = ReduceMax(mf, flags, 0, func_max);

    if (!local)
        {ReduceRealMax(the_max);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_max;
}

// *********************************************************************************************************************
Real Rincflo::SumOverLiquid (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumOverLiquid"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();

    auto sum_func = 
    [comp] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real sm = 0.0;
        auto func_b =
        [=, &sm] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular()) 
                {sm += mf_arr(i,j,k,comp);}
        };
        Loop(bx, func_b);
        return sm;
    };
    Real sum = ReduceSum(mf, flags, 0, sum_func);

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
Real Rincflo::MinOverLiquid (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MinOverLiquid"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();

    auto func_min = 
    [comp] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_min = inf;
        auto func_b = 
        [=, &the_min] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular()) 
                {the_min = min(the_min, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_min;
    };
    Real the_min = ReduceMin(mf, flags, 0, func_min);

    if (!local)
        {ReduceRealMin(the_min);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_min;
}

// *********************************************************************************************************************
Real Rincflo::MaxOverLiquid (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MaxOverLiquid"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();

    auto func_max = 
    [comp] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_max = -inf;
        auto func_b = 
        [=, &the_max] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular()) 
                {the_max = max(the_max, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_max;
    };
    Real the_max = ReduceMax(mf, flags, 0, func_max);

    if (!local)
        {ReduceRealMax(the_max);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_max;
}

// *********************************************************************************************************************
Real Rincflo::SumSquareOverLiquid (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumSquareOverLiquid"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();

    auto sum_func = 
    [comp] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real sm = 0.0;
        auto func_b =
        [=, &sm] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular()) 
                {sm += mf_arr(i,j,k,comp)*mf_arr(i,j,k,comp);}
        };
        Loop(bx, func_b);
        return sm;
    };
    Real sum = ReduceSum(mf, flags, 0, sum_func);

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
Real Rincflo::CountLiquidCells (int lev, bool local) const
{
    #define FUNC_NAME "Rincflo::CountLiquidCells"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const auto& mf = m_leveldata[lev]->volume;

    auto count_func = 
    [] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        // The number of active cells in thix box
        long nc = 0;
        auto func_b =
        [=, &nc] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular()) 
                {++nc;}
        };
        Loop(bx, func_b);
        return nc;
    };
    long nc = ReduceSum(mf, flags, 0, count_func);

    if (!local)
        {ReduceLongSum(nc);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return nc;
}

// *********************************************************************************************************************
Real Rincflo::SumOverActive (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumOverActive"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto sum_func = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real sm = 0.0;
        auto func_b =
        [=, &sm] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular() || f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {sm += mf_arr(i,j,k,comp);}
        };
        Loop(bx, func_b);
        return sm;
    };
    Real sum = ReduceSum(mf, flags, 0, sum_func);

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
Real Rincflo::SumSquareOverActive (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumSquareOverActive"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto sum_func = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real sm = 0.0;
        auto func_b =
        [=, &sm] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular() || f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {sm += mf_arr(i,j,k,comp)*mf_arr(i,j,k,comp);}
        };
        Loop(bx, func_b);
        return sm;
    };
    Real sum = ReduceSum(mf, flags, 0, sum_func);

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
Real Rincflo::CountActiveCells (int lev, bool local) const
{
    #define FUNC_NAME "Rincflo::CountActiveCells"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;
    const auto& mf = m_leveldata[lev]->volume;

    auto count_func = 
    [cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        // The number of active cells in thix box
        long nc = 0;
        auto func_b =
        [=, &nc] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular() || f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {++nc;}
        };
        Loop(bx, func_b);
        return nc;
    };
    long nc = ReduceSum(mf, flags, 0, count_func);

    if (!local)
        {ReduceLongSum(nc);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return nc;
}
// *********************************************************************************************************************
Real Rincflo::MinOverActive (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MinOverActive"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto func_min = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_min = inf;
        auto func_b = 
        [=, &the_min] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular() || f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {the_min = min(the_min, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_min;
    };
    Real the_min = ReduceMin(mf, flags, 0, func_min);

    if (!local)
        {ReduceRealMin(the_min);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_min;
}

// *********************************************************************************************************************
Real Rincflo::MaxOverActive (int lev, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::MaxOverActive"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
  
    const auto& fact = EBFactory(lev);
    const auto& flags = fact.getMultiEBCellFlagFab();
    const bool cover_multiple_cuts = m_cover_multiple_cuts;

    auto func_max = 
    [comp, cover_multiple_cuts] 
    AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr, Array4<EBCellFlag const> const& f) -> Real
    {
        Real the_max = -inf;
        auto func_b = 
        [=, &the_max] (int i, int j, int k) noexcept
        {
            if (f(i,j,k).isRegular() || f(i,j,k).isBoundary(cover_multiple_cuts)) 
                {the_max = max(the_max, mf_arr(i,j,k,comp));}
        };
        Loop(bx, func_b);
        return the_max;
    };
    Real the_max = ReduceMax(mf, flags, 0, func_max);

    if (!local)
        {ReduceRealMax(the_max);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return the_max;
}

// *********************************************************************************************************************
// dir = 0,1,2 (x,y,z) ; idx = the value of the index variable to match
Real Rincflo::SumOverSlice(int dir, int idx, const MultiFab& mf, int comp, bool local) const
{
    #define FUNC_NAME "Rincflo::SumOverSlice"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    
    // Do a reduction sum operation over the slice
    Real sum {0.0};
    constexpr int nghost {0};
    // Select the lambda function based on the direction
    switch (dir)
    {
        case 0:
        {
            // Lambda function to sum over x slice
            auto func_sum_i = 
            [comp, idx] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr) -> Real
            {
                Real sm = 0.0;
                auto func_b =
                [=, &sm] (int i, int j, int k) noexcept
                {
                    if (i==idx)
                        {sm += mf_arr(i,j,k,comp);}
                };
                Loop(bx, func_b);
                return sm;
            };
            sum = ReduceSum(mf, nghost, func_sum_i);
            break;
        }
        case 1:
        {
            // Lambda function to sum over y slice
            auto func_sum_j = 
            [comp, idx] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr) -> Real
            {
                Real sm = 0.0;
                auto func_b = 
                [=, &sm] (int i, int j, int k) noexcept
                {
                    if (j==idx)
                        {sm += mf_arr(i,j,k,comp);}
                };
                Loop(bx, func_b);
                return sm;
            };
            sum = ReduceSum(mf, nghost, func_sum_j);
            break;
        }
        case 2:
        {
            // Lambda function to sum over z slice
            auto func_sum_k = 
            [comp, idx] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& mf_arr) -> Real
            {
                Real sm = 0.0;
                auto func_b = 
                [=, &sm] (int i, int j, int k) noexcept
                {
                    if (k==idx)
                        {sm += mf_arr(i,j,k,comp);}
                };
                Loop(bx, func_b);
                return sm;
            };
            sum = ReduceSum(mf, nghost, func_sum_k);
            break;
        }
        default:
            Abort("Invalid direction in Rincflo::SumOverSlice");
    }

    if (!local)
        {ReduceRealSum(sum);}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}

// *********************************************************************************************************************
// dir = 0,1,2 (x,y,z) ; lo = true (low-boundary), false (hi-boundary)
Real Rincflo::SumOverBoundaryCells (int dir, bool lo, const MultiFab& mf, int comp, bool is_nodal, bool local) const
{
    #define FUNC_NAME "Rincflo::SumOverBoundaryCells"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    
    AMREX_D_TERM(
    int Nx= Geom(0).Domain().length(0);,
    int Ny= Geom(0).Domain().length(1);,
    int Nz= Geom(0).Domain().length(2);)

    // Get the index of the slice
    int idx {-1};
    switch (dir)
    {
        case 0:
            idx = lo ? 0 : Nx-1;
            break;
        case 1:
            idx = lo ? 0 : Ny-1;
            break;
        #if (AMREX_IS_3D)
        case 2:
            idx = lo ? 0 : Nz-1;
            break;
        #endif
        default:
            Abort("Invalid direction in Rincflo::SumOverBoundaryCells");
    }

    // Adjust the index value when summing the upper bound of nodal data
    if (is_nodal && !lo)
        {++idx;}

    // Sum over the boundary cells by delegating to SumOverSlice with the suitable direction and index
    Real sum = SumOverSlice(dir, idx, mf, comp, local);

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
    return sum;
}
