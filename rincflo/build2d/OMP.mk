AMREX_HOME      ?= ../../amrex
TOP             = ..
EBASE           = rfb
USE_EB          = TRUE
PRECISION       = DOUBLE
TINY_PROFILE    = FALSE
BL_NO_FORT      = TRUE
COMP            = gnu

DIM         = 2
DEBUG       = FALSE
USE_MPI     = FALSE
USE_OMP     = TRUE
USE_CUDA    = FALSE
USE_HYPRE   = TRUE
HYPRE_DIR   = ../../hypre/src/hypre

include ../src/Make.rincflo
