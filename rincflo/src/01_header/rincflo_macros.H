// rincflo_macros.H
// Macros related to dimension and EB support for code legibility.
#pragma once

// Set the macros AMREX_IS_2D, AMREX_IS_3D for more legible conditional compilation
#if (AMREX_SPACEDIM==2)
#define AMREX_IS_2D 1
#endif
#if (AMREX_SPACEDIM==3)
#define AMREX_IS_3D 1
#endif

// The only supported dimensions are 2D and 3D
static_assert( (AMREX_SPACEDIM==2) || (AMREX_SPACEDIM==3));

// Set the macro AMREX_EB_ONLY to include a one-line expression if/only using embedded boundaries
#if defined(AMREX_USE_EB)
#define AMREX_EB_ONLY(a) a
#else
#define AMREX_EB_ONLY(a) ((void)0)
#endif

// Set the macro AMREX_EB_DECL to include a variable followed by a comma if/only using embedded boundaries
#if defined(AMREX_USE_EB)
#define AMREX_EB_DECL(a) a,
#else
#define AMREX_EB_DECL(a)
#endif

// Set the macro AMREX_EB_TERM to include an expression if/only using embedded boundaries
#if defined(AMREX_USE_EB)
#define AMREX_EB_TERM(a) a
#else
#define AMREX_EB_TERM(a)
#endif

// Set the macro AMREX_EB_PICK to choose argument 1 when using EB, argument 2 when not
#if defined(AMREX_USE_EB)
#define AMREX_EB_PICK(a, b) a
#else
#define AMREX_EB_PICK(a, b) b
#endif

// Are we using Open MP?
#if defined(_OPENMP)
#define USE_OPENMP 1
#endif

// Are we using GPU?
#if defined(_GPU)
#define AMREX_USE_GPU 1
#endif

// Are we in debug mode?
#if defined(DEBUG_RFB)
#define DEBUG_RFB 1
#endif