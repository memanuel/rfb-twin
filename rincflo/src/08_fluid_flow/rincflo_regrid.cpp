#include <rincflo.H>

// *********************************************************************************************************************
// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization. Overrides the pure virtual function in AmrCore.
void Rincflo::MakeNewLevelFromScratch(
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    #define FUNC_NAME "Rincflo::MakeNewLevelFromScratch"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    if (m_verbose > 0)
    {
        Print() << format("Making new level {:d} from scratch with {:d} boxes.\n", lev, ba.size());
        if (m_verbose > 2) 
            {Print() << "BoxArray: " << ba << std::endl;}
    }
    // DEBUG
    // Print() << format("*****Making new level {:d} from scratch with {:d} boxes.\n", lev, ba.size());

    // Set the BoxArray and DistributionMap for this level
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
    // DEBUG
    // Print() << format("Set BoxArray and DistributionMap.\n");
    // Print() << format("geom[{:d}] ", lev) << geom[lev] << "\n";
    // Print() << format("grids[{:d}] ", lev) << grids[lev] << "\n";

    // Create the embedded boundary factory for this level
    #if (AMREX_USE_EB)
    m_factory[lev] = makeEBFabFactory(geom[lev], grids[lev], dmap[lev], ngrow(), EBSupport::full);
    #else
    m_factory[lev].reset(new FArrayBoxFactory());
    #endif

    // DEBUG
    // Print() << format("Created the EB factory.\n");

    // Create the LevelData for this level and bind it to m_leveldata[lev]
    m_leveldata[lev].reset(
        new LevelData(
            grids[lev], dmap[lev], *m_factory[lev], m_nspec, nghost_state(), m_advection_type,
            is_ImplicitDiffusion(), use_tensor_correction, do_AdvectConc(),
            need_Laplacian(), need_Kappa(), need_FluxRhs(), is_Reaction(), do_Electromigration()));

    // DEBUG
    // Print() << format("Completed m_leveldata[{:d}].reset().\n", lev);

    // Set the old and new time for this level
    m_t_old[lev] = time;
    m_t_new[lev] = time;

    // Initialize the velocity and density from the restart file if applicable
    if (!is_restart()) 
        {ProblemInitFluid(lev);}

    if (m_verbose > 1) {Print() << format("Level {:d} done.\n", lev);}

    // Set the MAC projector linear operator
    #if AMREX_USE_EB
    mac_proj.reset(
        new MacProjector(
            Geom(0,finest_level),
            MLMG::Location::FaceCentroid,           // Location of mac_vec
            MLMG::Location::FaceCentroid,           // Location of beta
            MLMG::Location::CellCenter,             // Location of phi (solution variable)
            MLMG::Location::CellCenter)             // Location of divu
            );
    #else
    mac_proj.reset(new MacProjector(Geom(0,finest_level)));
    #endif
    DEBUG_PRINT(format("Completed MakeNewLevelFromScratch({:d})", lev));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Make a new level using provided BoxArray and DistributionMapping and fill with interpolated coarse level data.
// Overrides the pure virtual function in AmrCore
void Rincflo::MakeNewLevelFromCoarse (
    int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    #define FUNC_NAME "Rincflo::MakeNewLevelFromCoarse"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    if (m_verbose > 0) {Print() << format("Making new level {:d} from coarse.\n", lev);}

    // Create the embedded boundary factory for this level
    #if (AMREX_USE_EB)
    std::unique_ptr<FabFactory<FArrayBox> > fact_new = makeEBFabFactory(geom[lev], ba, dm, ngrow(), EBSupport::full);
    #else
    std::unique_ptr<FabFactory<FArrayBox> > fact_new(new FArrayBoxFactory());
    #endif

    // DEBUG
    // Print() << format("*****MakeNewLevelFromCoarse - built new EBFabFactory.\n");

    // Create the LevelData for this level using the new BoxArray, DistributionMap and factory
    std::unique_ptr<LevelData> leveldata_new
        (new LevelData(ba, dm, *fact_new, m_nspec, nghost_state(), m_advection_type,  
            is_ImplicitDiffusion(),  use_tensor_correction,  do_AdvectConc(), 
            need_Laplacian(), need_Kappa(), need_FluxRhs(), is_Reaction(), do_Electromigration()));

    // DEBUG
    // Print() << format("*****MakeNewLevelFromCoarse - ready to call fillcoarsepatch on multiple FABs.\n");

    // Populate the new level with interpolated coarse data
    fillcoarsepatch_velocity(lev, time, leveldata_new->velocity, 0);
    fillcoarsepatch_density(lev, time, leveldata_new->density, 0);
    if (m_nspec > 0) 
        {fillcoarsepatch_conc(lev, time, leveldata_new->conc, 0);}
    fillcoarsepatch_gradp(lev, time, leveldata_new->gradp, 0);
    if (is_Reaction())
    {
        fillcoarsepatch_epotL(lev, time, leveldata_new->epotL, 0);
        fillcoarsepatch_epotS(lev, time, leveldata_new->epotS, 0);
    }
    // Set the pressure on the new level to zero
    leveldata_new->pressure.setVal(0.0);

    // DEBUG
    // Print() << format("*****MakeNewLevelFromCoarse - completed calls to fillcoarsepatch.\n");

    // Move the new leveldata to m_leveldata[lev]
    m_leveldata[lev] = std::move(leveldata_new);

    // Move the new EB factory to m_factory[lev]
    m_factory[lev] = std::move(fact_new);

    // Do NOT need to set the BoxArray and DistributionMap for this level; those are done in AmrCore::regrid

    // Set the diffusion operators to nullptr
    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();

    // AmrCore::regrid does not update finest_level = new_finest until AFTER this function completes!
    // Therefore when creating the new MacProjector below we need to use the new finest level.
    int new_finest_level = max(finest_level, lev);

    // Set the MAC projector linear operator
    #if AMREX_USE_EB
    mac_proj.reset(new MacProjector(
        Geom(0, new_finest_level),
        MLMG::Location::FaceCentroid,    // Location of mac_vec
        MLMG::Location::FaceCentroid,    // Location of beta
        MLMG::Location::CellCenter,      // Location of solution variable phi
        MLMG::Location::CellCenter)      // Location of divu
    );
    #else
    mac_proj.reset(new MacProjector(Geom(0, new_finest_level)));
    #endif

    // Set the reaction operators to nullptr
    m_reaction_scalar_op.reset();
    m_reaction_eliquid_op.reset();
    m_reaction_esolid_op.reset();

    DEBUG_PRINT(format("Completed MakeNewLevelFromCoarse({:d})", lev));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Remake an existing level using provided BoxArray and DistributionMapping and fill with existing fine and coarse data.
// Overrides the pure virtual function in AmrCore
void Rincflo::RemakeLevel (int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    #define FUNC_NAME "Rincflo::RemakeLevel"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    if (m_verbose > 0) {Print() << format("Remaking level {:d}.\n", lev);}

    // Create the embedded boundary factory for this level
    #if (AMREX_USE_EB)
    std::unique_ptr<FabFactory<FArrayBox> > fact_new = makeEBFabFactory(geom[lev], ba, dm, ngrow(), EBSupport::full);
    #else
    std::unique_ptr<FabFactory<FArrayBox> > fact_new(new FArrayBoxFactory());
    #endif

    // Create the LevelData for this level using the new BoxArray, DistributionMap and factory
    std::unique_ptr<LevelData> leveldata_new
        (new LevelData(ba, dm, *fact_new, m_nspec, nghost_state(), m_advection_type,  
            is_ImplicitDiffusion(),  use_tensor_correction,  do_AdvectConc(), 
            need_Laplacian(), need_Kappa(), need_FluxRhs(), is_Reaction(), do_Electromigration()));

    // Fill the new data with the old data
    fillpatch_velocity(lev, time, leveldata_new->velocity, 0);
    fillpatch_density(lev, time, leveldata_new->density, 0);
    if (m_nspec > 0) 
        {fillpatch_conc(lev, time, leveldata_new->conc, 0);}
    fillpatch_gradp(lev, time, leveldata_new->gradp, 0);
    fillpatch_epotL(lev, time, leveldata_new->epotL, 0);
    fillpatch_epotS(lev, time, leveldata_new->epotS, 0);
    // Set the new pressure to zero    
    leveldata_new->pressure.setVal(0.0);

    // Move the new leveldata to m_leveldata[lev]
    m_leveldata[lev] = std::move(leveldata_new);

    // Move the new EB factory to m_factory[lev]
    m_factory[lev] = std::move(fact_new);

    // Do NOT need to set the BoxArray and DistributionMap for this level; those are done in AmrCore::regrid

    // Set the diffusion operators to nullptr
    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();

    // Set the MAC projector linear operator
    #if AMREX_USE_EB
    mac_proj.reset(new MacProjector(
        Geom(0,finest_level),
        MLMG::Location::FaceCentroid,    // Location of mac_vec
        MLMG::Location::FaceCentroid,    // Location of beta
        MLMG::Location::CellCenter,      // Location of phi
        MLMG::Location::CellCenter)      // Location of divu
    );
    #else
    mac_proj.reset(new MacProjector(Geom(0,finest_level)));
    #endif

    // Set the reaction operators to nullptr
    m_reaction_scalar_op.reset();
    m_reaction_eliquid_op.reset();
    m_reaction_esolid_op.reset();

    DEBUG_PRINT(format("Completed RemakeLevel({:d})\n", lev));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

// *********************************************************************************************************************
// Delete level data
// overrides the pure virtual function in AmrCore
void Rincflo::ClearLevel (int lev)
{
    #define FUNC_NAME "Rincflo::ClearLevel"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    m_leveldata[lev].reset();
    m_factory[lev].reset();
    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();
    mac_proj.reset();
    m_reaction_scalar_op.reset();
    m_reaction_eliquid_op.reset();
    m_reaction_esolid_op.reset();

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
