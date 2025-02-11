#include <rincflo.H>

//**********************************************************************************************************************
Rincflo::LevelData::LevelData (
    BoxArray const& ba,
    DistributionMapping const& dm,
    FabFactory<FArrayBox> const& fact,
    int nspec, int ng_state,
    string advection_type, bool is_ImplicitDiffusion,
    bool use_tensor_correction, bool advect_conc,
    bool need_Laplacian, bool need_Kappa, bool need_FluxRhs, 
    bool need_reaction, bool need_electromigration)
    :   
        // Arguments to MultiFab constructor: BoxArray, DistributionMapping, ncomp, ngrow, MFInfo, FabFactory

        // Calculations about the geometry: cell type, area, volume
        cell_type       (ba, dm, 1, 0, MFInfo(), fact),
        area            (ba, dm, 1, 0, MFInfo(), fact),
        volume          (ba, dm, 1, 0, MFInfo(), fact),
        area_over_volume(ba, dm, 1, 0, MFInfo(), fact),
        // cell_centroid   (ba, dm, SpaceDim, 0, MFInfo(), fact),
        // bdry_centroid   (ba, dm, SpaceDim, 0, MFInfo(), fact),
        // bdry_normal     (ba, dm, SpaceDim, 0, MFInfo(), fact),

        // Fluid flow and species concentrations
        velocity  (ba, dm, SpaceDim, ng_state, MFInfo(), fact),
        velocity_o(ba, dm, SpaceDim, ng_state, MFInfo(), fact),
        density   (ba, dm, 1       , ng_state, MFInfo(), fact),
        density_o (ba, dm, 1       , ng_state, MFInfo(), fact),
        conc      (ba, dm, nspec   , ng_state, MFInfo(), fact),
        conc_o    (ba, dm, nspec   , ng_state, MFInfo(), fact),
        conc_on   (ba, dm, nspec   , ng_state, MFInfo(), fact),
        gradp     (ba, dm, SpaceDim, 0       , MFInfo(), fact),
        mac_phi   (ba, dm, 1       , 1       , MFInfo(), fact),

        // Pressure; need to convert this to a nodal grid (not cell centered)
        pressure  (convert(ba,IntVect::TheNodeVector()), dm, 1, 0 , MFInfo(), fact),

        // Convection of quantities
        conv_dconc_dt   (ba, dm, nspec   , 0, MFInfo(), fact),      
        conv_dconc_dt_o (ba, dm, nspec   , 0, MFInfo(), fact),
        conc_adv        (ba, dm, nspec   , 0, MFInfo(), fact),
        conc_adv_o      (ba, dm, nspec   , 0, MFInfo(), fact),
        conv_dv_dt_o    (ba, dm, SpaceDim, 0, MFInfo(), fact),
        conv_drho_dt_o  (ba, dm, 1       , 0, MFInfo(), fact)
{
    // Initialize the MultiFabs required for reaction simulations when applicable
    if (need_reaction)
    {
        // Electric potential
        epotL.define    (ba, dm, 1, ng_state, MFInfo(), fact);
        epotL_o.define  (ba, dm, 1, ng_state, MFInfo(), fact);
        epotS.define    (ba, dm, 1, ng_state, MFInfo(), fact);
        epotS_o.define  (ba, dm, 1, ng_state, MFInfo(), fact);

        // Concentrations and potential during nonlinear solve
        conc_op.define  (ba, dm, nspec   , ng_state, MFInfo(), fact);
        epotL_op.define (ba, dm, 1       , ng_state, MFInfo(), fact);
        epotS_op.define (ba, dm, 1       , ng_state, MFInfo(), fact);

        // Reaction source rate
        pot_diff.define     (ba, dm, 1, 0, MFInfo(), fact);
        overpot.define      (ba, dm, 1, 0, MFInfo(), fact);
        source_mol.define   (ba, dm, 1, 0, MFInfo(), fact);
        source.define       (ba, dm, 1, 0, MFInfo(), fact);

        // Derived fields for downstream analysis and plotting
        soc.define      (ba, dm, 1, ng_state, MFInfo(), fact);
        soc_o.define    (ba, dm, 1, ng_state, MFInfo(), fact);
        current.define  (ba, dm, 1, 0       , MFInfo(), fact);
        curr_dens.define(ba, dm, 1, 0       , MFInfo(), fact);

        // Optional fields in fancy reaction models
        if(need_Kappa)
            {kappa.define(ba, dm, 1, 1, MFInfo(), fact);}
        if(need_FluxRhs)
            {flxrhs.define(ba, dm, 1, 0, MFInfo(), fact);}
        if(need_electromigration)
            {divcgpot.define(ba, dm, nspec, ng_state, MFInfo(), fact);}
    }

    // Initialize the MultiFabs required required for flow simulations depending on optional settings
    // including advection type, implicit diffusion, tensor correction, and concentration advection
    if (advection_type != "MOL") 
    {
        divtau_o.define(ba, dm, SpaceDim, 0, MFInfo(), fact);    
        if (advect_conc)
            {laps_o.define(ba, dm, nspec, 0, MFInfo(), fact);}
        if(need_Laplacian)
        {
            laps.define    (ba, dm, nspec, 0, MFInfo(), fact);
            laps_o.define  (ba, dm, nspec, 0, MFInfo(), fact);
        }
    }
    else 
    {
        conv_dv_dt.define(ba, dm, SpaceDim, 0, MFInfo(), fact);
        conv_drho_dt.define (ba, dm, 1       , 0, MFInfo(), fact);

        if (!is_ImplicitDiffusion || use_tensor_correction) 
        {
            divtau.define  (ba, dm, SpaceDim, 0, MFInfo(), fact);
            divtau_o.define(ba, dm, SpaceDim, 0, MFInfo(), fact);
        }
        if (!is_ImplicitDiffusion && advect_conc)
        {
            laps.define  (ba, dm, nspec, 0, MFInfo(), fact);
            laps_o.define(ba, dm, nspec, 0, MFInfo(), fact);
        }
        else if(need_Laplacian)
        {
            laps.define    (ba, dm, nspec, 0, MFInfo(), fact);
            laps_o.define  (ba, dm, nspec, 0, MFInfo(), fact);
        }
    }
}

//**********************************************************************************************************************
// Resize all arrays when instance of Rincflo class is constructed.
// This is only done at the very start of the simulation. 
void Rincflo::ResizeArrays ()
{
    // Time holders for fillpatch stuff
    m_t_new.resize(max_level + 1);
    m_t_old.resize(max_level + 1);
    // MultiFabs with the simulation state variables
    m_leveldata.resize(max_level+1);
    // Embedded boudnary factory
    m_factory.resize(max_level+1);
    
    // Change in velocity on each level; initialize to infinity
    m_change_vel.resize(max_level+1, std::numeric_limits<Real>::infinity());
    // Change in concentration on each level; initialize to infinity
    m_change_soc.resize(max_level+1, std::numeric_limits<Real>::infinity());
    // Change in potential on each level; for testing stagnation condition; initialize to infinity
    m_change_epot.resize(max_level+1, std::numeric_limits<Real>::infinity());

    // Minimum change in velocity on each level; for testing stagnation condition; initialize to infinity
    m_change_min_vel.resize(max_level+1, std::numeric_limits<Real>::infinity());
    // Minimum change in concentration on each level; for testing stagnation condition; initialize to infinity
    m_change_min_soc.resize(max_level+1, std::numeric_limits<Real>::infinity());
    // Minimum change in potential on each level; for testing stagnation condition; initialize to infinity
    m_change_min_epot.resize(max_level+1, std::numeric_limits<Real>::infinity());
    
    // Full history of changes in concentration; initialize each level with an empty vector of reals
    for (int lev=0; lev<= max_level; ++lev)
        {m_change_hist_soc.push_back(Vector<Real>());}
}

//**********************************************************************************************************************
Rincflo::LevelDataNumpyInput::LevelDataNumpyInput (
    BoxArray const& ba, DistributionMapping const& dm,
    FabFactory<FArrayBox> const& fact, int dim, int nt, int ng) : 

        // arguments to MultiFab constructor are:
        // BoxArray, DistributionMapping, ncomp, ngrow, info, factory

        // Geometry
        cell_type   (ba, dm, 1  , 0 , MFInfo(), fact),

        // Flow variables
        velocity    (ba, dm, dim, ng, MFInfo(), fact),
        pressure    (convert(ba,IntVect::TheNodeVector()), 
                               dm, 1  , 0 , MFInfo(), fact),
        gradp       (ba, dm, dim, 0 , MFInfo(), fact),
        // density  (ba, dm, 1  , ng, MFInfo(), fact),

        // Reaction state variables
        conc      (ba, dm, nt,  ng, MFInfo(), fact),
        epotL     (ba, dm, 1  , ng, MFInfo(), fact),
        epotS     (ba, dm, 1  , ng, MFInfo(), fact),

        // Reaction cut cells
        current     (ba, dm, 1  , 0 , MFInfo(), fact),
        overpot     (ba, dm, 1  , 0 , MFInfo(), fact)
{}

//**********************************************************************************************************************
Rincflo::LevelDataNumpyOutput::LevelDataNumpyOutput (
    BoxArray const& ba, DistributionMapping const& dm,
    FabFactory<FArrayBox> const& fact, int dim, int nt, int ng) : 

        // arguments to MultiFab constructor are:
        // BoxArray, DistributionMapping, ncomp, ngrow, info, factory

        // Geometry
        cell_type    (ba, dm, 1  , 0 , MFInfo(), fact),
        area         (ba, dm, 1  , 0 , MFInfo(), fact),
        volume       (ba, dm, 1  , 0 , MFInfo(), fact),
        // cell_centroid(ba, dm, dim, 0 , MFInfo(), fact),
        // bdry_centroid(ba, dm, dim, 0 , MFInfo(), fact),
        // bdry_normal  (ba, dm, dim, 0 , MFInfo(), fact),

        // Flow variables
        velocity    (ba, dm, dim, ng, MFInfo(), fact),
        pressure    (convert(ba,IntVect::TheNodeVector()), 
                         dm, 1  , 0 , MFInfo(), fact),
        gradp       (ba, dm, dim, 0 , MFInfo(), fact),

        // Reaction state variables
        conc      (ba, dm, nt , ng, MFInfo(), fact),
        soc         (ba, dm, 1  , ng, MFInfo(), fact),
        epotL       (ba, dm, 1  , ng, MFInfo(), fact),
        epotS       (ba, dm, 1  , ng, MFInfo(), fact),

        // Reaction cut cells
        current     (ba, dm, 1  , 0 , MFInfo(), fact),
        overpot     (ba, dm, 1  , 0 , MFInfo(), fact)
{}
