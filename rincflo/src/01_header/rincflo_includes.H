// rincflo_includes.H
// Fragment of Rincflo.H
// All of the included headers and names used by Rincflo
#pragma once

// Standard libraries
#include <vector>
#include <string>
#include <fstream>
#include <format>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>

// Macros defining the dimension and EB support - this MUST come first!
// Had a bug that was extremely hard to find because prob_bc.H was included before this
#include <rincflo_macros.H>

// AMReX Libraries
#include <AMReX_ParmParse.H>
#include <AMReX_AmrCore.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_NodalProjector.H>
#include <AMReX_MacProjector.H>
#include <AMReX_Math.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_buildInfo.H>

// Embedded boundary support
#if (AMREX_USE_EB)
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBAmrUtil.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB_utils.H>
#include <AMReX_EBInterpolater.H>
#include <AMReX_EBInvInterpolater.H>
#include <EBGodunov.H>
#include <AMReX_WriteEBSurface.H>
#include <Redistribution.H>
#endif

// Incflo headers
#include <prob_bc.H>
#include <eb_if.H>
#include <Convection.H>
#include <Godunov.H>
#include <MOL.H>
#include <DiffusionScalarOp.H>
#include <DiffusionTensorOp.H>

// RFB headers
#include <DiffusionTensorOp.H>
#include <DiffusionScalarOp.H>
#include <ReactionScalarOp.H>
#include <ReactionEliquidOp.H>
#include <ReactionEsolidOp.H>
#include <cnpy.H>

// ************************************************************************************************
// Alias a few  amrex namespace names for more legible code
namespace EB2 = amrex::EB2;
namespace BCType = amrex::BCType;
namespace Gpu = amrex::Gpu;
namespace ParallelAllReduce = amrex::ParallelAllReduce;
namespace ParallelContext = amrex::ParallelContext;
namespace ParallelDescriptor = amrex::ParallelDescriptor;

// Names used from standard library and format library
using std::size_t, std::string, std::vector, std::fstream, std::ofstream, std::format;

// Take granular control over amrex names used here
using amrex::Real, amrex::Vector, amrex::IntVect, amrex::Array, amrex::Array4, 
    amrex::FabFactory, amrex::MultiFab, amrex::FabType, amrex::MFInfo, amrex::MFIter, amrex::MFItInfo,
    amrex::Box, amrex::BoxArray, amrex::FArrayBox, amrex::EBFArrayBoxFactory, amrex::IArrayBox, amrex::RealBox,
    amrex::TagBox, amrex::TagBoxArray,  amrex::VisMF, amrex::MultiFabFileFullPrefix,
    amrex::SpaceDim, amrex::Geometry, amrex::GeometryData, amrex::Orientation, 
    amrex::EBCellFlag, amrex::EBCellFlagFab, amrex::EBSupport, 
    amrex::Interpolater, amrex::InterpFromCoarseLevel, 
    amrex::cell_cons_interp, amrex::eb_cell_cons_interp, amrex::ebinv_cell_cons_interp,
    amrex::pc_interp, amrex::isnan, amrex::isinf, amrex::min, amrex::max,
    amrex::BCType::foextrap, amrex::BCType::hoextrap, amrex::BCType::ext_dir, amrex::BCType::int_dir,
    amrex::LinOpBCType, amrex::OrientationIter, amrex::Direction, amrex::PhysBCFunct, 
    amrex::BCRec, amrex::DistributionMapping, amrex::LPInfo, 
    amrex::MLMG, amrex::MLMGInv, amrex::MLLinOp, amrex::MLABecLaplacian, amrex::MLEBABecLap, amrex::MLEBABecLapInv,
    amrex::GpuArray, amrex::TilingIfNotGPU, amrex::Gpu::DeviceVector, amrex::GpuBndryFuncFab, amrex::Elixir,
    amrex::MLTensorOp, amrex::MLEBTensorOp,
    amrex::Print, amrex::AllPrint, amrex::Abort, amrex::OutStream, amrex::ParmParse,
    amrex::MacProjector, amrex::NodalProjector, amrex::convert, amrex::grow, amrex::average_cellcenter_to_face, 
    amrex::ReduceMin, amrex::ReduceMax, amrex::Loop, amrex::ParallelFor, 
    amrex::ParallelDescriptor::MyProc, amrex::ParallelDescriptor::IOProcessor, 
    amrex::ParallelDescriptor::IOProcessorNumber, amrex::ParallelDescriptor::NProcs, 
    amrex::ParallelDescriptor::second, amrex::ParallelDescriptor::Barrier, 
    amrex::ParallelDescriptor::ReduceRealMin, amrex::ParallelDescriptor::ReduceRealMax,     
    amrex::ParallelDescriptor::ReduceRealSum, amrex::ParallelDescriptor::ReduceLongSum, 
    amrex::ParallelDescriptor::ReadAndBcastFile, 
    amrex::vfrac_small, amrex::vfrac_almost_zero, amrex::vfrac_almost_one, 
    amrex::afrac_small, amrex::afrac_almost_zero, amrex::afrac_almost_one, 
    amrex::buildInfoGetGitHash, amrex::buildInfoGetBuildGitHash, amrex::buildInfoGetBuildGitName, 
    amrex::buildInfoGetBuildDate, amrex::buildInfoGetCXXName, amrex::buildInfoGetComp, amrex::buildInfoGetCompVersion,
    amrex::buildInfoGetBuildMachine, amrex::buildInfoGetAMReXDir, amrex::buildInfoGetBuildDir,
    amrex::buildInfoGetModuleName, amrex::buildInfoGetModuleVal, amrex::buildInfoGetNumModules,
    amrex::buildInfoGetLibraries, amrex::buildInfoGetLinkFlags, amrex::buildInfoGetCXXFlags, 
    amrex::buildInfoGetFcomp, amrex::buildInfoGetFcompVersion, amrex::buildInfoGetFFlags,
    cnpy::NpyArray;

// Constants
constexpr Real NaN = std::numeric_limits<Real>::quiet_NaN();