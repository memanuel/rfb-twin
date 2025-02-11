// rincflo_copy.H
// Fragment of Rincflo.H
// All the copy methods in Rincflo
#pragma once
#include <rincflo_macros.H>

private:

// Copy level data from new to old
void CopyNewToOld_velocity      (         IntVect const& ng = IntVect{0});
void CopyNewToOld_velocity      (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_density       (         IntVect const& ng = IntVect{0});
void CopyNewToOld_density       (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_conc          (         IntVect const& ng = IntVect{0});
void CopyNewToOld_conc          (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_epotL         (         IntVect const& ng = IntVect{0});
void CopyNewToOld_epotL         (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_epotS         (         IntVect const& ng = IntVect{0});
void CopyNewToOld_epotS         (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_soc           (         IntVect const& ng = IntVect{0});
void CopyNewToOld_soc           (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_conc_adv      (         IntVect const& ng = IntVect{0});
void CopyNewToOld_conc_adv      (int lev, IntVect const& ng = IntVect{0});
void CopyNewToOld_conv_dconc_dt (         IntVect const& ng = IntVect{0});
void CopyNewToOld_conv_dconc_dt (int lev, IntVect const& ng = IntVect{0});

// Copy level data from old to new
void CopyOldToNew_velocity  (         IntVect const& ng = IntVect{0});
void CopyOldToNew_velocity  (int lev, IntVect const& ng = IntVect{0});
void CopyOldToNew_density   (         IntVect const& ng = IntVect{0});
void CopyOldToNew_density   (int lev, IntVect const& ng = IntVect{0});
void CopyOldToNew_conc      (         IntVect const& ng = IntVect{0});
void CopyOldToNew_conc      (int lev, IntVect const& ng = IntVect{0});
void CopyOldToNew_epotL     (         IntVect const& ng = IntVect{0});
void CopyOldToNew_epotL     (int lev, IntVect const& ng = IntVect{0});
void CopyOldToNew_epotS     (         IntVect const& ng = IntVect{0});
void CopyOldToNew_epotS     (int lev, IntVect const& ng = IntVect{0});
void CopyOldToNew_soc       (         IntVect const& ng = IntVect{0});
void CopyOldToNew_soc       (int lev, IntVect const& ng = IntVect{0});

// Copy from new to old in a Picard iteration
void CopyNewToOldPicard_react (IntVect const& ng = IntVect{0});
void CopyNewToOldPicard_react (int lev, IntVect const& ng = IntVect{0});

// Copy from old to new in a Picard iteration
void CopyOldToNewPicard_react (IntVect const& ng = IntVect{0});
void CopyOldToNewPicard_react (int lev, IntVect const& ng = IntVect{0});
