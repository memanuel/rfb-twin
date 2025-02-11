AMREX_HOME      ?= ../../amrex
TOP             = ..
EBASE           = rfb
USE_EB          = TRUE
PRECISION       = DOUBLE
TINY_PROFILE    = FALSE
BL_NO_FORT      = TRUE
COMP            = gnu

DIM         = 2
DEBUG       = TRUE
USE_MPI     = TRUE
USE_OMP     = FALSE
USE_CUDA    = FALSE
USE_HYPRE   = TRUE
HYPRE_DIR   = ../../hypre/src/hypre

DEFINES 	+= -DDEBUG_RFB

include ../src/Make.rincflo
