#include <rincflo.H>

namespace { const std::string level_prefix{"Level_"}; }

//**********************************************************************************************************************
void GotoNextLine(std::istream& is)
{
    constexpr std::streamsize bl_ignore_max{100000};
    is.ignore(bl_ignore_max, '\n');
}

//**********************************************************************************************************************
void Rincflo::WriteHeader(const std::string& name, bool is_checkpoint) const
{
    if(IOProcessor())
    {
        std::string HeaderFileName(name + "/Header");
        VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;

        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

        HeaderFile.open(HeaderFileName.c_str(),
                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

        if(!HeaderFile.good()) {
            amrex::FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(17);

        if(is_checkpoint) {
            HeaderFile << "Checkpoint version: 1\n";
        } else {
            HeaderFile << "HyperCLaw-V1.1\n";
        }

        HeaderFile << finest_level << "\n";

        // Time stepping controls
        HeaderFile << m_nstep << "\n";
        HeaderFile << m_cur_time << "\n";
        HeaderFile << m_dt << "\n";
        HeaderFile << m_prev_dt << "\n";
        HeaderFile << m_prev_prev_dt << "\n";
        HeaderFile << m_rstep << "\n";
        HeaderFile << m_reactant_consumption_max << "\n";
    
        // Geometry
        for(int i = 0; i < BL_SPACEDIM; ++i) {
            HeaderFile << Geom(0).ProbLo(i) << ' ';
        }
        HeaderFile << '\n';

        for(int i = 0; i < BL_SPACEDIM; ++i)
            HeaderFile << Geom(0).ProbHi(i) << ' ';
        HeaderFile << '\n';

        // BoxArray
        for(int lev = 0; lev <= finest_level; ++lev)
        {
            boxArray(lev).writeOn(HeaderFile);
            HeaderFile << '\n';
        }
    }
}

//**********************************************************************************************************************
void Rincflo::WriteCheckPointFile(CheckPointType type ) const
{
    #define FUNC_NAME "Rincflo::WriteCheckPointFile"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    std::string checkpointname;
    switch (type) 
    {
        case CheckPointType::FluidFlow:
            checkpointname = amrex::Concatenate(m_check_file, m_nstep, m_chk_min_digits);
            break;
        case CheckPointType::Reaction:
            checkpointname = amrex::Concatenate(m_react_check_file, m_rstep, m_chk_min_digits);
            break;
    }

    const string status = format("Writing checkpoint {:s}\n", checkpointname.c_str());
    // Print() << status;
    DEBUG_PRINT(status);
    amrex::PreBuildDirectorHierarchy(checkpointname, level_prefix, finest_level + 1, true);

    bool is_checkpoint = true;
    WriteHeader(checkpointname, is_checkpoint);
    WriteJobInfo(checkpointname);

    for(int lev = 0; lev <= finest_level; ++lev)
    {
        DEBUG_PRINT(format("WriteCheckPointFile() - Processing level {:d}.\n", lev));

        VisMF::Write(m_leveldata[lev]->velocity,
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "velocity"));

        VisMF::Write(m_leveldata[lev]->density,
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "density"));

        DEBUG_PRINT("WriteCheckPointFile() - Completed velocity and density.\n");

        VisMF::Write(m_leveldata[lev]->pressure,
                        amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "pressure"));

        VisMF::Write(m_leveldata[lev]->gradp,
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "gradp"));

        DEBUG_PRINT("WriteCheckPointFile() - Completed pressure and pressure gradient.\n");

        if (is_Reaction()) 
        {
            VisMF::Write(m_leveldata[lev]->conc,
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "conc"));

            VisMF::Write(m_leveldata[lev]->source,
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "source"));

            VisMF::Write(m_leveldata[lev]->overpot,
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "overpot"));
        }

        DEBUG_PRINT("WriteCheckPointFile() - Completed concentrations, source and overpot.\n");

        if(is_PotComputed()) 
        {
            VisMF::Write(m_leveldata[lev]->epotL,
                            amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "epotL"));
      
            VisMF::Write(m_leveldata[lev]->epotS,
                            amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "epotS"));

            DEBUG_PRINT("WriteCheckPointFile() - Completed epotL and epotS.\n");
        }
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

//**********************************************************************************************************************
void Rincflo::LoadCheckPointHeader()
{
    #define FUNC_NAME "Rincflo::LoadCheckPointHeader"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Main steps in LoadCheckPointHeader:
    // set up problem domain including BoxArray
    // allocate Rincflo memory (Rincflo::AllocateArrays)
    // call MakeNewLevelFromScratch

    // Problem domain
    Real prob_lo[BL_SPACEDIM];
    Real prob_hi[BL_SPACEDIM];

    // Get reference to the header file as an istringstream
    std::string File(m_restart_file + "/Header");
    Vector<char> fileCharPtr;
    ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    // Strings for reading in lines from checkpoint file
    std::string line, word;

    // Start reading from checkpoint file 

    // Title line
    std::getline(is, line);

    // Finest level
    is >> finest_level;
    GotoNextLine(is);

    // Step count
    is >> m_nstep;
    GotoNextLine(is);

    // Current time
    is >> m_cur_time;
    GotoNextLine(is);

    // Time step size
    is >> m_dt;
    GotoNextLine(is);
    is >> m_prev_dt;
    GotoNextLine(is);
    is >> m_prev_prev_dt;
    GotoNextLine(is);

    // Reaction step count
    is >> m_rstep;
    // Since we just loaded m_rstep, apply this value m_start_rsteps as well
    m_start_rsteps = m_rstep;
    GotoNextLine(is);
   
    // Last tuned value of max reactant consumption
    Real ck_reactant_consumption_max;
    is >> ck_reactant_consumption_max;
    GotoNextLine(is);
    // Set the starting value at the min of the checkpoint and input values
    m_reactant_consumption_max = std::min(m_reactant_consumption_max, ck_reactant_consumption_max);

    // Low coordinates of domain bounding box
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while(lis >> word)
            {prob_lo[i++] = std::stod(word);}
    }

    // High coordinates of domain bounding box
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while(lis >> word)
            {prob_hi[i++] = std::stod(word);}
    }

    // Set up problem domain
    RealBox rb(prob_lo, prob_hi);
    Geometry::ResetDefaultProbDomain(rb);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {SetGeometry(lev, Geometry(Geom(lev).Domain(), rb, Geom(lev).CoordInt(), Geom(lev).isPeriodic()));}

    // DEBUG
    // Print() << "LoadCheckPointHeader() - set geometry.\n";
    // Print() << format("finest_level = {:d}.\n", finest_level);

    // Set up BoxArray and DistributionMapping on levels with input data
    for(int lev = 0; lev <= finest_level; ++lev)
    {
        // Create a new BoxArray from Header for this level
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);
        // Print() << format("LoadCheckpointHeader - read BoxArray.\n");
        // Create distribution mapping
        DistributionMapping dm {ba, NProcs()};
        // Print() << format("Created DistributionMapping with {:d} processors.\n", NProcs());
        // Create a new level
        MakeNewLevelFromScratch(lev, m_cur_time, ba, dm);
        // Print() << format("LoadCheckpointHeader - completed level {:d}.\n", lev);
    }

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

//**********************************************************************************************************************
void Rincflo::LoadCheckPointFluidData ()
{
    #define FUNC_NAME "Rincflo::LoadCheckPointFluidData"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    for(int lev = 0; lev <= finest_level; ++lev)
    {
        VisMF::Read(m_leveldata[lev]->velocity,
                    amrex::MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "velocity"));
        VisMF::Read(m_leveldata[lev]->density,
                    amrex::MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "density"));
        VisMF::Read(m_leveldata[lev]->pressure,
                    amrex::MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "pressure"));
        VisMF::Read(m_leveldata[lev]->gradp,
                    amrex::MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "gradp"));
    } // for / lev

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

//**********************************************************************************************************************
void Rincflo::LoadCheckPointReactionData ()
{
    #define FUNC_NAME "Rincflo::LoadCheckPointReactionData"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    // Should we load the concentration field? Yes unless we're reinitializing all the concentrations
    bool load_conc = (!m_react_reinit_all_conc);
    // Should we load the source term? Yes when running Butler-Volmer, unless reinitializing all concentrations
    bool load_source = (m_reaction_model == ReactionModel::ButlerVolmer) && load_conc;
    // Should we load the overpotential? Yes whenever running the Nernst model, unelss reinitializing all concentrations
    bool load_overpot = (m_reaction_model == ReactionModel::Nernst) && load_conc;
    // Should we load epotL? Yes when running ButlerVolmer, unless we're reinitializing it
    bool load_epotL = (m_reaction_model == ReactionModel::ButlerVolmer) && !m_react_reinit_epotL && load_conc;
    // Should we load epotS? Yes when running ButlerVolmer, unless we're reinitializing it
    bool load_epotS = (m_reaction_model == ReactionModel::ButlerVolmer) && !m_react_reinit_epotS && load_conc;

    for(int lev = 0; lev <= finest_level; ++lev)
    {
        // Load the concentration when applicable
        if (load_conc)
            {VisMF::Read(m_leveldata[lev]->conc, MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "conc"));}
        // Load the source term when applicable
        if (load_source)
            {VisMF::Read(m_leveldata[lev]->source, MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "source"));}
        // Load overpot when applicable
        if (load_overpot)
            {VisMF::Read(m_leveldata[lev]->overpot, MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "overpot"));}
        // Load epotL when applicable
        if (load_epotL) 
        {
            VisMF::Read(m_leveldata[lev]->epotL, MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "epotL"));
            // DEBUG
            Print() << "***** Loaded epotL from checkpoint.\n";
        }
        // Load epotS when applicable
        if (load_epotS) 
            {VisMF::Read(m_leveldata[lev]->epotS, MultiFabFileFullPrefix(lev, m_restart_file, level_prefix, "epotS"));}
    } // for / lev

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

//**********************************************************************************************************************
void Rincflo::ReadCheckPointFile()
{
    #define FUNC_NAME "Rincflo::ReadCheckPointFile"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    const string status = format("Restarting from checkpoint {:s}\n", m_restart_file.c_str());
    Print() << status;
    DEBUG_PRINT(status);

    // Read the header and build the BoxArray and DistributionMapping
    LoadCheckPointHeader();
    // DEBUG
    // Print() << "Loaded checkpoint header.\n";

    // Load fluid flow data
    LoadCheckPointFluidData();
    // DEBUG
    // Print() << "Loaded fluid data.\n";

    // Load reaction data if applicable
    if (is_Reaction() && !m_react_reinit_all_conc)
        {LoadCheckPointReactionData();}

    // Print() << "Restart complete.\n";
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}

//**********************************************************************************************************************
void Rincflo::WriteJobInfo(const std::string& path) const
{
    if(IOProcessor())
    {
        // job_info file with details about the run
        std::ofstream jobInfoFile;
        std::string FullPathJobInfoFile = path;
        std::string PrettyLine =
            "===============================================================================\n";

        FullPathJobInfoFile += "/rincflo_job_info";
        jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

        // job information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Rincflo Job Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "number of MPI processes: " << NProcs() << "\n";
        #if (USE_OPENMP)
        jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
        #endif

        jobInfoFile << "\n\n";

        // build information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Build Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "build date:    " << buildInfoGetBuildDate() << "\n";
        jobInfoFile << "build machine: " << buildInfoGetBuildMachine() << "\n";
        jobInfoFile << "build dir:     " << buildInfoGetBuildDir() << "\n";
        jobInfoFile << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "COMP:          " << buildInfoGetComp() << "\n";
        jobInfoFile << "COMP version:  " << buildInfoGetCompVersion() << "\n";
        jobInfoFile << "FCOMP:         " << buildInfoGetFcomp() << "\n";
        jobInfoFile << "FCOMP version: " << buildInfoGetFcompVersion() << "\n";

        jobInfoFile << "\n";

        const char* githash1 = buildInfoGetGitHash(1);
        const char* githash2 = buildInfoGetGitHash(2);
        if(std::strlen(githash1) > 0)
        {
            jobInfoFile << "Rincflo git hash: " << githash1 << "\n";
        }
        if(std::strlen(githash2) > 0)
        {
            jobInfoFile << "AMReX git hash: " << githash2 << "\n";
        }

        jobInfoFile << "\n\n";

        // grid information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Grid Information\n";
        jobInfoFile << PrettyLine;

        for(int i = 0; i <= finest_level; i++)
        {
            jobInfoFile << " level: " << i << "\n";
            jobInfoFile << "   number of boxes = " << grids[i].size() << "\n";
            jobInfoFile << "   maximum zones   = ";
            for(int dir = 0; dir < BL_SPACEDIM; dir++)
            {
                jobInfoFile << geom[i].Domain().length(dir) << " ";
            }
            jobInfoFile << "\n\n";
        }

        jobInfoFile << "\n\n";

        // runtime parameters
        jobInfoFile << PrettyLine;
        jobInfoFile << " Inputs File Parameters\n";
        jobInfoFile << PrettyLine;

        ParmParse::dumpTable(jobInfoFile, true);

        jobInfoFile.close();
    }
}

//**********************************************************************************************************************
int Rincflo::CountPlotComponents()
{
    // Accumulate all the possible plot components
    int ncomp {0};

    // Cell type
    if (m_plt_cell_type) {++ncomp;}
    // Embedded boundary fields
    #if (AMREX_USE_EB)
    // Cut cell area fraction
    if(m_plt_afrac) {++ncomp;}
    // Cut cell volume fraction
    if(m_plt_vfrac) {++ncomp;}
    // Cut cell surface area
    if(m_plt_area) {++ncomp;}
    // Cell volume
    if(m_plt_volume) {++ncomp;}
    #endif

    // Basic fluid flow outputs: velocity and pressure
    // Velocity (magnitude)
    if (m_plt_velmag) {++ncomp;}
    // Velocity components
    AMREX_D_TERM(
    if (m_plt_velx) {++ncomp;},
    if (m_plt_vely) {++ncomp;},
    if (m_plt_velz) {++ncomp;})

    // Pressure
    if(m_plt_p) {++ncomp;}

    // Advanced fluid outputs
    // Laplacian tracers
    m_plt_laps &= need_Laplacian();
    if (m_plt_laps) {ncomp += m_nspec;}
    // Convective term
    if (m_plt_cnvt) {ncomp += m_nspec;}
    // Density
    if (m_plt_rho) {++ncomp;}

    // Pressure gradient components
    AMREX_D_TERM(
    if (m_plt_gpx) {++ncomp;},
    if (m_plt_gpy) {++ncomp;},
    if (m_plt_gpz) {++ncomp;})

    // Vorticity
    if(m_plt_vort) {++ncomp;}

    // Flags to summarize capabilities of the simulation run
    bool is_nernst = (m_reaction_model == ReactionModel::Nernst);
    bool is_special = (m_reaction_model == ReactionModel::SpecialRedox);
    bool is_BV = (m_reaction_model == ReactionModel::ButlerVolmer);
    bool is_advanced =  is_nernst || is_special || is_BV;

    // Need 2 or more species for SOC and a source term
    m_plt_soc &= (is_Reaction() && (m_nspec>1));
    m_plt_src &= (is_Reaction() && (m_nspec>1));
    // These fields are only available in more advanced models
    m_plt_overpot   = m_plt_overpot && is_advanced;
    m_plt_potdiff   = m_plt_potdiff && is_advanced;
    m_plt_curr      = m_plt_curr && is_advanced;
    m_plt_curr_dens = m_plt_curr_dens && is_advanced;
    m_plt_epotL     = m_plt_epotL && is_BV;
    m_plt_epotS     = m_plt_epotS && is_BV;
    m_plt_flxrhs    = m_plt_flxrhs && m_calc_flux_epotL;
    m_plt_kappaL    = m_plt_kappaL && is_BV;
    m_plt_divcgpot  = m_plt_divcgpot && is_BV && do_Electromigration();

    // Species concentrations
    if (m_plt_conc) {ncomp += m_nspec;}
    // State of charge: c1/(c0+c1) only available with 2 or more species
    if(m_plt_soc) {++ncomp;}

    // Electrochemistry and reaction kinetics
    // Electric field (liquid)
    if(m_plt_epotL) {ncomp++;}
    // Electric field (solid)
    if(m_plt_epotS) {ncomp++;}
    // Potential difference
    if(m_plt_potdiff) {++ncomp;}
    // Overpotential
    if(m_plt_overpot) {++ncomp;}
    // Current
    if(m_plt_curr) {++ncomp;}
    // Current density
    if(m_plt_curr_dens) {++ncomp;}
    // Source term (for c0 / reduced species)
    if(m_plt_src) {++ncomp;}

    // Advanced reaction outputs
    // kappaL
    if(m_plt_kappaL) {++ncomp;}
    // flxrhs
    if(m_plt_flxrhs) {++ncomp;}
    // divcgpot
    if(m_plt_divcgpot) {ncomp+=m_nspec;}
    
    return ncomp;
}

//**********************************************************************************************************************
void Rincflo::WritePlotFile(CheckPointType type, int step )
{
    #define FUNC_NAME "Rincflo::WritePlotFile"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);
    // const string status = format("Entering WritePlotFile({:d}, {:d}))\n", static_cast<int>(type), step);

    // Set number of ghost cells
    const int ng = has_eb() ? 2 : 1;
    // #if (AMREX_USE_EB)
    // const int ng = (EBFactory(0).isAllRegular()) ? 1 : 2;
    // #else
    // const int ng = 1;
    // #endif

    // Patch velocity and gradp
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
        fillpatch_velocity(lev, m_cur_time, m_leveldata[lev]->velocity, ng);
        fillpatch_gradp(lev, m_cur_time, m_leveldata[lev]->gradp, 0);
    }

    // Set name of plotfile and print status message
    std::string plotfilename;
    switch (type) 
    {
        case CheckPointType::FluidFlow:
            plotfilename = amrex::Concatenate(m_plot_file, step, m_chk_min_digits);
            break;
        case CheckPointType::Reaction:
            plotfilename = amrex::Concatenate(m_react_plot_file, step, m_chk_min_digits);
            break;
        default:
            plotfilename = amrex::Concatenate("errorplt", step, m_chk_min_digits);
    }
    Print() << 
        format("\nWriting plotfile {:s} at step {:d} (time={:f}).\n", plotfilename.c_str(), step, m_cur_time);

    // The number of components we are going to plot
    int ncomp = CountPlotComponents();
    DEBUG_PRINT(format("Counted {:d} plot components.\n", ncomp));
    
    // Multifab for the fields we want to plot
    Vector<MultiFab> mf(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) 
        {mf[lev].define(grids[lev], dmap[lev], ncomp, 0, MFInfo(), Factory(lev));}

    // Variable names for the plot
    Vector<std::string> pltscaVarsName;
    int icomp = 0;
    // Number of components copied at a time is usually one except for concentrations
    constexpr int numcomp {1};

    // Compute derived reaction fields when applicable
    if (is_Reaction()) 
        {CalcDerivedReact();}

    // ********************************
    // Embedded boundary
    // ********************************
    #if (AMREX_USE_EB)
    // Plot flags (region type)
    if (m_plt_cell_type) 
    {
        for (int lev = 0; lev <= finest_level; ++lev)
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->cell_type, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("cell_type");
        ++icomp;
    }

    // Plot afrac (embedded boundary fraction)
    if (m_plt_afrac)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], EBFactory(lev).getBndryArea().ToMultiFab(0.0,0.0), 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("afrac");
        ++icomp;
    }

    // Plot volume fraction (liquid)
    if (m_plt_vfrac) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], EBFactory(lev).getVolFrac(), 0, icomp, numcomp, 0);}
      pltscaVarsName.push_back("vfrac");
      ++icomp;
    }
    #endif

    // Plot area (active surface area)
    if (m_plt_area)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->area, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("area");
        ++icomp;
    }

    // Plot volume of liquid
    if (m_plt_volume)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->volume, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("volume");
        ++icomp;
    }

    // ********************************
    // Basic fluid flow
    // ********************************
    // Plot velocity magnitude
    if (m_plt_velmag) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab velmag(mf[lev], amrex::make_alias, icomp, 1);
            CalcVelMag(lev, velmag, m_leveldata[lev]->velocity);
        }
        pltscaVarsName.push_back("velmag");
        ++icomp;
    }

    // Plot velocity x component
    if (m_plt_velx)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->velocity, 0, icomp, 1, 0);}
        pltscaVarsName.push_back("velx");
        ++icomp;
    }

    // Plot velocity y component
    if (m_plt_vely)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->velocity, 1, icomp, 1, 0);}
        pltscaVarsName.push_back("vely");
        ++icomp;
    }

    // Plot velocity z component
    #if (AMREX_IS_3D)
    if (m_plt_velz) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->velocity, 2, icomp, 1, 0);}
        pltscaVarsName.push_back("velz");
        ++icomp;
    }
    #endif

    // Plot pressure
    if (m_plt_p) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            const auto& pressure = m_leveldata[lev]->pressure;
            MultiFab pressure_adj(pressure.boxArray(), pressure.DistributionMap(), 1, 0);
            CalcAdjustedPressure(lev, pressure_adj, pressure, true);
            amrex::average_node_to_cellcenter(mf[lev], icomp, pressure_adj, 0, numcomp);
        }
        pltscaVarsName.push_back("pressure");
        ++icomp;
    }

    // Plot pressure gradient p_x
    if (m_plt_gpx) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab::Copy(mf[lev], m_leveldata[lev]->gradp, 0, icomp, numcomp, 0);
            // add the background pressure to gradp - this is NOT included in the solution of the Poisson equation!
            mf[lev].plus(m_gp0[0], icomp, numcomp, 0);
        }
        pltscaVarsName.push_back("gradpx");
        ++icomp;
    }

    // Plot pressure gradient p_y
    if (m_plt_gpy) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab::Copy(mf[lev], m_leveldata[lev]->gradp, 1, icomp, numcomp, 0);
            // add the background pressure to gradp - this is NOT included in the solution of the Poisson equation!
            mf[lev].plus(m_gp0[1], icomp, numcomp, 0);
        }
        pltscaVarsName.push_back("gradpy");
        ++icomp;
    }

    // Plot pressure gradient p_z
    #if (AMREX_IS_3D)
    if (m_plt_gpz) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab::Copy(mf[lev], m_leveldata[lev]->gradp, 2, icomp, numcomp, 0);
            // add the background pressure to gradp - this is NOT included in the solution of the Poisson equation!
            mf[lev].plus(m_gp0[1], icomp, numcomp, 0);
        }
        pltscaVarsName.push_back("gradpz");
        ++icomp;
    }
    #endif

    // ********************************
    // Basic reaction
    // ********************************
    // Plot state of charge
    if (m_plt_soc) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->soc, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("soc");
        ++icomp;
    }

    // Plot concentrations
    if (m_plt_conc) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->conc, 0, icomp, m_nspec, 0);}
        for (int i = 0; i < m_nspec; ++i) 
            {pltscaVarsName.push_back("conc"+std::to_string(i));}
        icomp += m_nspec;
    }
    
    // ********************************
    // Electrochemistry and reaction kinetics
    // ********************************
    // Plot potential in solid
    if (m_plt_epotS)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->epotS, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("epotS");
        ++icomp;
    }

    // Plot potential in liquid
    if (m_plt_epotL)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->epotL, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("epotL");
        ++icomp;
      }

    // Plot potential difference
    if (m_plt_potdiff) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->pot_diff, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("pot_diff");
        ++icomp;
    }

    // Plot overpotential
    if (m_plt_overpot) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->overpot, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("overpot");
        ++icomp;
    }

    // Plot current
    if (m_plt_curr) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->current, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("current");
        ++icomp;
    }

    // Plot current density
    if (m_plt_curr_dens) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->curr_dens, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("curr_dens");
        ++icomp;
    }

    // Plot source term (millimolarity change per second)
    if (m_plt_src)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->source, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("src");
        ++icomp;
    }

    // Plot conductivity in liquid
    if (m_plt_kappaL) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->kappa, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("kL");
        ++icomp;
    }

    // Plot divergence of potential
    if (m_plt_divcgpot) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->divcgpot, 0, icomp, m_nspec, 0);}
        for (int i = 0; i < m_nspec; ++i) 
            {pltscaVarsName.push_back("divcgpot"+std::to_string(i));}
        icomp += m_nspec;
    }
   
    // PLot flux on right side
    if (m_plt_flxrhs) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->flxrhs, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("flx");
        ++icomp;
    }

    // ********************************
    // Advanced fluid flow
    // ********************************
    // Plot laplacian tracers
    if (m_plt_laps) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->laps, 0, icomp, m_nspec, 0);}
        for (int i = 0; i < m_nspec; ++i) 
            {pltscaVarsName.push_back("laps"+std::to_string(i));}
        icomp += m_nspec;
    }

    // Plot convection
    if (m_plt_cnvt)
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->conv_dconc_dt, 0, icomp, m_nspec, 0);}
        for (int i = 0; i < m_nspec; ++i) 
            {pltscaVarsName.push_back("cnvt"+std::to_string(i));}
        icomp += m_nspec;
    }

    // Plot density
    if (m_plt_rho) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
            {MultiFab::Copy(mf[lev], m_leveldata[lev]->density, 0, icomp, numcomp, 0);}
        pltscaVarsName.push_back("density");
        ++icomp;
   
        #if (AMREX_USE_EB)
        for (int lev = 0; lev <= finest_level; ++lev) 
            {EB_set_covered(mf[lev], 0.0);}
        #endif
    }
    
    // Plot vorticity
    if (m_plt_vort) 
    {
        for (int lev = 0; lev <= finest_level; ++lev) 
        {
            MultiFab vort(mf[lev], amrex::make_alias, icomp, 1);
            CalcVorticity(lev, m_cur_time, vort, m_leveldata[lev]->velocity);
        }
        pltscaVarsName.push_back("vort");
        ++icomp;
    }

    // Check that we've plotted the right number of components...    
    AMREX_ALWAYS_ASSERT(ncomp == static_cast<int>(pltscaVarsName.size()));

    // This needs to be defined in order to use amrex::WriteMultiLevelPlotfile, 
    // but will never change unless we use subcycling. 
    // If we do use subcycling, this should be a Rincflo class member. 
    Vector<int> istep(finest_level + 1, step);

    // Write the plotfile
    amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, GetVecOfConstPtrs(mf), 
                                   pltscaVarsName, Geom(), m_cur_time, istep, refRatio());
    WriteJobInfo(plotfilename);

    DEBUG_PRINT(format("Completed WritePlotFile({:d}).\n", step));
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME

} // function WritePlotFile
