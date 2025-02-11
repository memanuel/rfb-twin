AMREX_HOME      ?= ../../amrex
TOP             = ..
EBASE           = rfb
USE_EB          = TRUE
PRECISION       = DOUBLE
TINY_PROFILE    = FALSE
BL_NO_FORT      = TRUE
COMP            = gnu

DIM         = 3
DEBUG       = FALSE
USE_MPI     = TRUE
USE_OMP     = FALSE
USE_CUDA    = TRUE
USE_HYPRE   = TRUE
HYPRE_DIR   = ../../hypre/src/hypre

include ../src/Make.rincflo
