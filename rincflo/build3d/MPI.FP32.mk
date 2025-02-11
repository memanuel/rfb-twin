AMREX_HOME      ?= ../../amrex
TOP             = ..
EBASE           = rfb.fp32.
USE_EB          = TRUE
PRECISION       = SINGLE
TINY_PROFILE    = FALSE
BL_NO_FORT      = TRUE
COMP            = gnu

DIM         = 3
DEBUG       = FALSE
USE_MPI     = TRUE
USE_OMP     = FALSE
USE_CUDA    = FALSE
USE_HYPRE   = TRUE
HYPRE_DIR   = ../../hypre/src/hypre

include ../src/Make.rincflo
