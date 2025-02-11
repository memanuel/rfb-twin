#include <rincflo.H>

// tag cells for refinement
// overrides the pure virtual function in AmrCore
void Rincflo::ErrorEst (int lev, TagBoxArray& tags, Real time, int ngrow)
{
    #define FUNC_NAME "Rincflo::ErrorEst"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    static bool first = true;
    
    static Vector<Real> rhoerr_v;
    static Vector<Real> AMREX_D_DECL(velxerr_v, velyerr_v, velzerr_v), dervelerr_v;
    static Vector<Real> conc1err_v, conc2err_v, conc1errb_v, conc2errb_v, derconcerr_v;
    static Vector<Real> socerr_v;
    
    if (first) 
    {
        first = false;
        ParmParse pp("rincflo");
    
        pp.query("do_rho_ref", m_do_rho_ref);
        if(m_do_rho_ref>0)
        {
            pp.queryarr("rhoerr", rhoerr_v);
            if (rhoerr_v.size() > 0) 
            {
                Real last = rhoerr_v.back();
                rhoerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on RHO" << std::endl;
        }

        pp.query("do_conc1_ref", m_do_conc1_ref);
        if(m_do_conc1_ref>0)
        {
            pp.queryarr("conc1err", conc1err_v);
            if (conc1err_v.size() > 0) 
            {
                Real last = conc1err_v.back();
                conc1err_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on CONC_1 (above)" << std::endl;
        }
    
        pp.query("do_conc1_ref_below", m_do_conc1_ref_below);
        if(m_do_conc1_ref_below>0)
        {
            pp.queryarr("conc1errb", conc1errb_v);
            if (conc1errb_v.size() > 0) 
            {
                Real last = conc1errb_v.back();
                conc1errb_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on CONC_1 (below)" << std::endl;
        }

        pp.query("do_conc2_ref", m_do_conc2_ref);
        if(m_do_conc2_ref>0)
        {
            pp.queryarr("conc2err", conc2err_v);
            if (conc2err_v.size() > 0) 
            {
                Real last = conc2err_v.back();
                conc2err_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on CONC_2 (above)" << std::endl;
        }

        pp.query("do_conc2_ref_below", m_do_conc2_ref_below);
        if(m_do_conc2_ref_below>0)
        {
            pp.queryarr("conc2errb", conc2errb_v);
            if (conc2errb_v.size() > 0) 
            {
                Real last = conc2errb_v.back();
                conc2errb_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on CONC_2 (below)" << std::endl;
        }

        pp.query("do_velx_ref", m_do_velx_ref);
        if(m_do_velx_ref>0)
        {
            pp.queryarr("velxerr", velxerr_v);
            if (velxerr_v.size() > 0) 
            {
                Real last = velxerr_v.back();
                velxerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on VEL_X" << std::endl;
        }

        pp.query("do_vely_ref", m_do_vely_ref);
        if(m_do_vely_ref>0)
        {
            pp.queryarr("velyerr", velxerr_v);
            if (velyerr_v.size() > 0) 
            {
                Real last = velyerr_v.back();
                velyerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on VEL_Y" << std::endl;
        }

        #if (AMREX_IS_3D)
        pp.query("do_velz_ref", m_do_velz_ref);
        if(m_do_velz_ref>0)
        {
            pp.queryarr("velzerr", velzerr_v);
            if (velzerr_v.size() > 0) 
            {
                Real last = velzerr_v.back();
                velzerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on VEL_Z" << std::endl;
        }
        #endif
    
        pp.query("do_dervel_ref", m_do_dervel_ref);
        if(m_do_dervel_ref>0)
        {
            pp.query("exclude_z_dir_for_dervel_ref", m_exclude_z_dir_for_dervel_ref);
            pp.queryarr("dervelerr", dervelerr_v);
            if (dervelerr_v.size() > 0) 
            {
                Real last = dervelerr_v.back();
                dervelerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on DER VEL" << std::endl;
        }

        pp.query("do_derconc_ref", m_do_derconc_ref);
        if(m_do_derconc_ref>0)
        {
            pp.queryarr("dertraerr", derconcerr_v);
            if (derconcerr_v.size() > 0) 
            {
                Real last = derconcerr_v.back();
                derconcerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on DER CONC" << std::endl;
        }

        pp.query("do_soc_ref", m_do_soc_ref);
        if(m_do_soc_ref>0)
        {
            pp.queryarr("socerr", socerr_v);
            if (socerr_v.size() > 0) 
            {
                Real last = socerr_v.back();
                socerr_v.resize(max_level+1, last);
            }
            Print() << "SET REFINING on SOC" << std::endl;
        }

        // Print refining information
        if (m_verbose > 0)
        {
            Print() << "##### REFINING INFO ##### " << std::endl;
            Print() << "## rho_ref= " << m_do_rho_ref << std::endl;
            if (m_do_rho_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<rhoerr_v.size();i++) 
                    {Print() << rhoerr_v[i] << " ";}
                Print() << std::endl;
            }
            Print() << "## velx_ref= " << m_do_velx_ref << std::endl;
            if (m_do_velx_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<velxerr_v.size();i++) 
                    {Print() << velxerr_v[i] << " ";}
                Print() << std::endl;
            }
            Print() << "## vely_ref= " << m_do_vely_ref << std::endl;
            if (m_do_vely_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<velyerr_v.size();i++) 
                    {Print() << velyerr_v[i] << " ";}
                Print() << std::endl;
            }
            #if (AMREX_IS_3D)
            Print() << "## velz_ref= " << m_do_velz_ref << std::endl;
            if (m_do_velz_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<velzerr_v.size();i++) 
                    {Print() << velzerr_v[i] << " ";}
                Print() << std::endl;
            }
            #endif

            Print() << "## dervel_ref= " << m_do_dervel_ref << std::endl;
            if (m_do_dervel_ref > 0) 
            {
                if(m_exclude_z_dir_for_dervel_ref) {  Print() << "# neglecting z-direction!\n" ;}
                Print() << "# values: " ;
                for(int i=0;i<dervelerr_v.size();i++) 
                    {Print() << dervelerr_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "## spedcies1_ref= " << m_do_conc1_ref << std::endl;
            if (m_do_conc1_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<conc1err_v.size();i++) 
                    {Print() << conc1err_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "## conc1_ref_below= " << m_do_conc1_ref_below << std::endl;
            if (m_do_conc1_ref_below > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<conc1errb_v.size();i++) 
                    {Print() << conc1errb_v[i] << " ";}
                Print() << std::endl;
            }
            Print() << "## conc2_ref= " << m_do_conc2_ref << std::endl;
            if (m_do_conc2_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<conc2err_v.size();i++) 
                    {Print() << conc2err_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "## conc2_ref_below= " << m_do_conc2_ref_below << std::endl;
            if (m_do_conc2_ref_below > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<conc2errb_v.size();i++) 
                    {Print() << conc2errb_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "## derconc_ref= " << m_do_derconc_ref << std::endl;
            if (m_do_derconc_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<derconcerr_v.size();i++) 
                    {Print() << derconcerr_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "## soc_ref= " << m_do_soc_ref << std::endl;
            if (m_do_soc_ref > 0) 
            {
                Print() << "# values: " ;
                for(int i=0;i<socerr_v.size();i++)
                    {Print() << socerr_v[i] << " ";}
                Print() << std::endl;
            }

            Print() << "#########################" << std::endl;
        }
    }

    //decide if we need to tag depending on flow or reaction part (and level)
    bool tag_rho = lev < rhoerr_v.size();
    bool tag_conc1 = lev < conc1err_v.size();
    bool tag_conc2 = lev < conc2err_v.size();
    bool tag_conc1b = lev < conc1errb_v.size();
    bool tag_conc2b = lev < conc2errb_v.size();
    AMREX_D_TERM(
    bool tag_velx = lev < velxerr_v.size();,
    bool tag_vely = lev < velyerr_v.size();,
    bool tag_velz = lev < velzerr_v.size();)
    bool tag_dervel = lev < dervelerr_v.size();
    bool tag_dertra = lev < derconcerr_v.size();
    bool tag_soc = lev < socerr_v.size();

    if( m_rstep < 1 ) 
    {  
        //flow part
        if( m_do_rho_ref==0 or m_do_rho_ref==2 ) {tag_rho=false;}
        if( m_do_conc1_ref==0 or m_do_conc1_ref==2 ) {tag_conc1=false;}
        if( m_do_conc2_ref==0 or m_do_conc2_ref==2 ) {tag_conc2=false;}
        if( m_do_conc1_ref_below==0 or m_do_conc1_ref_below==2 ) {tag_conc1b=false;}
        if( m_do_conc2_ref_below==0 or m_do_conc2_ref_below==2 ) {tag_conc2b=false;}
        if( m_do_velx_ref==0 or m_do_velx_ref==2 ) {tag_velx=false;}
        if( m_do_vely_ref==0 or m_do_vely_ref==2 ) {tag_vely=false;}
        #if (AMREX_IS_3D)
        if( m_do_velz_ref==0 or m_do_velz_ref==2 ) {tag_velz=false;}
        #endif
        if( m_do_dervel_ref==0 or m_do_dervel_ref==2 ) {tag_dervel=false;}
        if( m_do_derconc_ref==0 or m_do_derconc_ref==2 ) {tag_dertra=false;}
        if( m_do_soc_ref==0 or m_do_soc_ref==2 ) {tag_soc=false;}
    }
    else 
    {  
        //reaction part
        if( m_do_rho_ref<2 ) {tag_rho=false;}
        if( m_do_conc1_ref<2 ) {tag_conc1=false;}
        if( m_do_conc2_ref<2 ) {tag_conc2=false;}
        if( m_do_conc1_ref_below<2 ) {tag_conc1b=false;}
        if( m_do_conc2_ref_below<2 ) {tag_conc2b=false;}
        if( m_do_velx_ref<2 ) {tag_velx=false;}
        if( m_do_vely_ref<2 ) {tag_vely=false;}
        #if (AMREX_IS_3D)
        if( m_do_velz_ref<2 ) {tag_velz=false;}
        #endif
        if( m_do_dervel_ref<2 ) {tag_dervel=false;}
        if( m_do_derconc_ref<2 ) {tag_dertra=false;}
        if( m_do_soc_ref<2 ) {tag_soc=false;}
    }

    //for the derivatives we need 1,2 ghost cell(s)... ??
    if (tag_dervel) 
        {fillpatch_velocity(lev, time, m_leveldata[lev]->velocity, 2);}
    if (tag_dertra) 
        {fillpatch_conc(lev, time, m_leveldata[lev]->conc, 2);}

    const auto   tagval = TagBox::SET;

    #if (AMREX_USE_EB)
    auto const& factory = EBFactory(lev);
    auto const& flags_mf = factory.getMultiEBCellFlagFab();
    #endif
    
    AMREX_D_TERM(
    const Real dx = Geom(lev).CellSize(0);,
    const Real dy = Geom(lev).CellSize(1);,
    const Real dz = Geom(lev).CellSize(2);)
    
    if (tag_rho) 
    {
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(m_leveldata[lev]->density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& tag = tags.array(mfi);
            Array4<Real const> const& rho = m_leveldata[lev]->density.const_array(mfi);
            Real rhoerr = tag_rho ? rhoerr_v[lev]: std::numeric_limits<Real>::max();
            const EBCellFlagFab& flags = flags_mf[mfi];
            const auto& flag = flags.const_array();
      
            ParallelFor(bx,
            [tag_rho,rhoerr,tagval,rho,tag,flag]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (tag_rho and rho(i,j,k) > rhoerr) 
                    {tag(i,j,k) = tagval;}
            });
        }
    }

    if (tag_conc1 or tag_conc2 or tag_dertra  ) 
    {
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(m_leveldata[lev]->conc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& tag = tags.array(mfi);
            Array4<Real const> const& trac = m_leveldata[lev]->conc.const_array(mfi);
            Real traerr = tag_conc1 ? conc1err_v[lev]: std::numeric_limits<Real>::max();
            Real tra2err = tag_conc2 ? conc2err_v[lev] : std::numeric_limits<Real>::max();
            Real traerrb = tag_conc1b ? conc1errb_v[lev]: std::numeric_limits<Real>::max();
            Real tra2errb = tag_conc2b ? conc2errb_v[lev] : std::numeric_limits<Real>::max();
            Real dertraerr = tag_dertra ? derconcerr_v[lev] : std::numeric_limits<Real>::max();
            int ncomp = m_nspec;
            const EBCellFlagFab& flags = flags_mf[mfi];
            const auto& flag = flags.const_array();
            ParallelFor(bx,
            [   AMREX_D_DECL(dx,dy,dz),ncomp,tag_conc1,tag_conc2,tag_conc1b,tag_conc2b,tag_dertra,
                traerr,tra2err,traerrb,tra2errb,dertraerr,tagval,trac,tag,flag]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if(tag_conc1 and trac(i,j,k,0) > traerr) 
                    {tag(i,j,k) = tagval;}
                if (tag_conc2 and trac(i,j,k,1) > tra2err) 
                    {tag(i,j,k) = tagval;}
                if(tag_conc1b and trac(i,j,k,0) < traerrb) 
                    {tag(i,j,k) = tagval;}
                if (tag_conc2b and trac(i,j,k,1) < tra2errb) 
                    {tag(i,j,k) = tagval;}
                if(tag_dertra) 
                {
                    if(!flag(i,j,k).isCovered() ) 
                    {
                        for(int n=0; n< ncomp ; n++) 
                        {
                            //perhaps to improve to next-next-neigh (+-2,0,0) and similar
                            Real derx=0.0,dery=0.0;
                            #if (AMREX_IS_3D)
                            Real derz=0.0;
                            #endif
                            Real eps=0.01;
                            Real uxx=0.0,uxm=0.0,uxp=0.0,avu=0.0;
                            Real vyy=0.0,vym=0.0,vyp=0.0,avv=0.0;
                            #if (AMREX_IS_3D)
                            Real wzz=0.0,wzm=0.0,wzp=0.0,avw=0.0;
                            #endif
                                   
                            if (!flag(i+1,j,k).isCovered() && !flag(i-1,j,k).isCovered()) 
                            { 
                                //no covered neighs
                                uxp = std::abs(trac(i+1,j,k,n) - trac(i,j,k,n));
                                uxm = std::abs(trac(i-1,j,k,n) - trac(i,j,k,n));
                                uxx = std::abs(trac(i-1,j,k,n) - 2*trac(i,j,k,n) + trac(i+1,j,k,n));
                                avu = std::abs(trac(i-1,j,k,n)) + 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i+1,j,k,n));
                            }  
                            else if (!flag(i+1,j,k).isCovered()) 
                            { 
                                //covered left (if possible fish right for second der)
                                uxp = std::abs(trac(i+1,j,k,n) - trac(i,j,k,n));
                                if (!flag(i+2,j,k).isCovered())
                                    uxx = std::abs(trac(i,j,k,n) - 2*trac(i+1,j,k,n) + trac(i+2,j,k,n));
                                    avu = 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i+1,j,k,n));
                            }
                            else if (!flag(i-1,j,k).isCovered()) 
                            { 
                                //covered right (if possible fish left for second der)
                                uxp = std::abs(trac(i-1,j,k,n) - trac(i,j,k,n));
                                if (!flag(i-2,j,k).isCovered())
                                uxx = std::abs(trac(i-2,j,k,n) - 2*trac(i-1,j,k,n) + trac(i,j,k,n));
                                avu = std::abs(trac(i-1,j,k,n)) + 2*std::abs(trac(i,j,k,n));
                            }

                            if (!flag(i,j+1,k).isCovered() && !flag(i,j-1,k).isCovered()) 
                            { 
                                //no covered neighs
                                vyp = std::abs(trac(i,j+1,k,n) - trac(i,j,k,n));
                                vym = std::abs(trac(i,j-1,k,n) - trac(i,j,k,n));
                                vyy = std::abs(trac(i,j-1,k,n) - 2*trac(i,j,k,n) + trac(i,j+1,k,n));
                                avv = std::abs(trac(i,j-1,k,n)) + 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i,j+1,k,n));
                            }  
                            else if (!flag(i,j+1,k).isCovered()) 
                            { 
                                //covered left (if possible fish right for second der)
                                vyp = std::abs(trac(i,j+1,k,n) - trac(i,j,k,n));
                                if (!flag(i,j+2,k).isCovered())
                                    vyy = std::abs(trac(i,j,k,n) - 2*trac(i,j+1,k,n) + trac(i,j+2,k,n));
                                avv = 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i,j+1,k,n));
                            }
                            else if (!flag(i,j-1,k).isCovered()) 
                            { 
                                //covered right (if possible fish left for second der)
                                vyp = std::abs(trac(i,j-1,k,n) - trac(i,j,k,n));
                                if (!flag(i,j-2,k).isCovered())
                                    vyy = std::abs(trac(i,j-2,k,n) - 2*trac(i,j-1,k,n) + trac(i,j,k,n));
                                avv = std::abs(trac(i,j-1,k,n)) + 2*std::abs(trac(i,j,k,n));
                            }
                            #if (AMREX_IS_3D)
                            if (!flag(i,j,k+1).isCovered() && !flag(i,j,k-1).isCovered()) 
                            { 
                                //no covered neighs
                                wzp = std::abs(trac(i,j,k+1,n) - trac(i,j,k,n));
                                wzm = std::abs(trac(i,j,k-1,n) - trac(i,j,k,n));
                                wzz = std::abs(trac(i,j,k-1,n) - 2*trac(i,j,k,n) + trac(i,j,k+1,n));
                                avw = std::abs(trac(i,j,k-1,n)) + 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i,j,k+1,n));
                            }  
                            else if (!flag(i,j,k+1).isCovered()) 
                            { 
                                //covered left (if possible fish right for second der)
                                wzp = std::abs(trac(i,j,k+1,n) - trac(i,j,k,n));
                                if (!flag(i,j,k+2).isCovered())
                                    wzz = std::abs(trac(i,j,k,n) - 2*trac(i,j,k+1,n) + trac(i,j,k+2,n));
                                avw = 2*std::abs(trac(i,j,k,n)) + std::abs(trac(i,j,k+1,n));
                            }
                            else if (!flag(i,j,k-1).isCovered()) 
                            { 
                                //covered right (if possible fish left for second der)
                                wzp = std::abs(trac(i,j,k-1,n) - trac(i,j,k,n));
                                if (!flag(i,j,k-2).isCovered())
                                    wzz = std::abs(trac(i,j,k-2,n) - 2*trac(i,j,k-1,n) + trac(i,j,k,n));
                                avw = std::abs(trac(i,j,k-1,n)) + 2*std::abs(trac(i,j,k,n));
                            }
                            #endif
                            AMREX_D_TERM(
                            Real denx=0.25*avu;,
                            Real deny=0.25*avv;,
                            Real denz=0.25*avw;)
                            Real smallnumber=1e-8;
                            if(denx > smallnumber)
                                {derx= uxx / denx * dx;}
                            if(deny > smallnumber)
                                {dery= vyy / deny * dy;}
                            #if (AMREX_IS_3D)
                            if(denz > smallnumber)
                                {derz= wzz / denz * dz;}
                            #endif         
                            if (max(AMREX_D_DECL(derx,dery,derz)) >= dertraerr) 
                                {tag(i,j,k) = tagval;}
                        }
                    }
                }
            });
        }
    }

    #if (AMREX_IS_3D)
    if (tag_velx or tag_vely or tag_velz or tag_dervel ) {
    #else
    if (tag_velx or tag_vely or tag_dervel ) 
    {
        #endif
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(m_leveldata[lev]->velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& tag = tags.array(mfi);
            Array4<Real const> const& vel = m_leveldata[lev]->velocity.const_array(mfi);
            AMREX_D_TERM(
            Real velxerr = tag_velx ? velxerr_v[lev]: std::numeric_limits<Real>::max();,
            Real velyerr = tag_vely ? velyerr_v[lev]: std::numeric_limits<Real>::max();,
            Real velzerr = tag_velz ? velzerr_v[lev]: std::numeric_limits<Real>::max();)
            Real dervelerr = tag_dervel ? dervelerr_v[lev] : std::numeric_limits<Real>::max();
            int exclude_z_dir = m_exclude_z_dir_for_dervel_ref;
            const EBCellFlagFab& flags = flags_mf[mfi];
            const auto& flag = flags.const_array();
            ParallelFor(bx,
            [   AMREX_D_DECL(dx,dy,dz),exclude_z_dir,AMREX_D_DECL(tag_velx,tag_vely,tag_velz),tag_dervel,
                AMREX_D_DECL(velxerr,velyerr,velzerr),dervelerr,tagval,vel,tag,flag]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (tag_velx and std::abs(vel(i,j,k,0)) > velxerr) 
                    {tag(i,j,k) = tagval;}
                if (tag_vely and std::abs(vel(i,j,k,1)) > velyerr) 
                    {tag(i,j,k) = tagval;}
                #if (AMREX_IS_3D)
                if (tag_velz and std::abs(vel(i,j,k,2)) > velzerr) 
                    {tag(i,j,k) = tagval;}
                #endif
                if (tag_dervel) 
                {
                    if(!flag(i,j,k).isCovered() ) 
                    {
                        //perhaps to improve to next-next-neigh (+-2,0,0) and similar (~second-order)
                        Real derxx=0.0,derxy=0.0;
                        Real deryx=0.0,deryy=0.0;
                        Real uxx=0.0,uyy=0.0;
                        Real vxx=0.0,vyy=0.0;
                        Real aux=0.0,auy=0.0;
                        Real avx=0.0,avy=0.0;
                        #if (AMREX_IS_3D)
                        Real derxz=0.0,deryz=0.0;
                        Real derzx=0.0,derzy=0.0,derzz=0.0;
                        Real uzz=0.0,vzz=0.0;
                        Real wxx=0.0,wyy=0.0,wzz=0.0;
                        Real auz=0.0,avz=0.0;
                        Real awx=0.0,awy=0.0,awz=0.0;
                        #endif                   
                                   
                        if (!flag(i+1,j,k).isCovered() && !flag(i-1,j,k).isCovered()) 
                        { 
                            //no covered neighs
                            AMREX_D_TERM(
                            uxx = std::abs(vel(i-1,j,k,0) - 2*vel(i,j,k,0) + vel(i+1,j,k,0));,
                            vxx = std::abs(vel(i-1,j,k,1) - 2*vel(i,j,k,1) + vel(i+1,j,k,1));,
                            wxx = std::abs(vel(i-1,j,k,2) - 2*vel(i,j,k,2) + vel(i+1,j,k,2));)
                            AMREX_D_TERM(
                            aux = std::abs(vel(i-1,j,k,0)) + 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i+1,j,k,0));,
                            avx = std::abs(vel(i-1,j,k,1)) + 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i+1,j,k,1));,
                            awx = std::abs(vel(i-1,j,k,2)) + 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i+1,j,k,2));)
                        }  
                        else if (!flag(i+1,j,k).isCovered()) 
                        { 
                            //covered left (if possible fish right for second der)
                            if (!flag(i+2,j,k).isCovered()) 
                            {
                                AMREX_D_TERM( 
                                uxx = std::abs(vel(i,j,k,0) - 2*vel(i+1,j,k,0) + vel(i+2,j,k,0));,
                                vxx = std::abs(vel(i,j,k,1) - 2*vel(i+1,j,k,1) + vel(i+2,j,k,1));,
                                wxx = std::abs(vel(i,j,k,2) - 2*vel(i+1,j,k,2) + vel(i+2,j,k,2));)
                            }
                            AMREX_D_TERM(
                            aux = 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i+1,j,k,0));,
                            avx = 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i+1,j,k,1));,
                            awx = 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i+1,j,k,2));)
                        }
                        else if (!flag(i-1,j,k).isCovered()) 
                        { 
                            //covered right (if possible fish left for second der)
                            if (!flag(i-2,j,k).isCovered()) 
                            {
                                AMREX_D_TERM(
                                uxx = std::abs(vel(i-2,j,k,0) - 2*vel(i-1,j,k,0) + vel(i,j,k,0));,
                                vxx = std::abs(vel(i-2,j,k,1) - 2*vel(i-1,j,k,1) + vel(i,j,k,1));,
                                wxx = std::abs(vel(i-2,j,k,2) - 2*vel(i-1,j,k,2) + vel(i,j,k,2));)
                            }
                                AMREX_D_TERM(
                                aux = std::abs(vel(i-1,j,k,0)) + 2*std::abs(vel(i,j,k,0));,
                                avx = std::abs(vel(i-1,j,k,1)) + 2*std::abs(vel(i,j,k,1));,
                                awx = std::abs(vel(i-1,j,k,2)) + 2*std::abs(vel(i,j,k,2));)
                        }
                        
                        if (!flag(i,j+1,k).isCovered() && !flag(i,j-1,k).isCovered()) 
                        { 
                            //no covered neighs
                            AMREX_D_TERM(
                            uyy = std::abs(vel(i,j-1,k,0) - 2*vel(i,j,k,0) + vel(i,j+1,k,0));,
                            vyy = std::abs(vel(i,j-1,k,1) - 2*vel(i,j,k,1) + vel(i,j+1,k,1));,
                            wyy = std::abs(vel(i,j-1,k,2) - 2*vel(i,j,k,2) + vel(i,j+1,k,2));)
                            AMREX_D_TERM(
                            auy = std::abs(vel(i,j-1,k,0)) + 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i,j+1,k,0));,
                            avy = std::abs(vel(i,j-1,k,1)) + 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i,j+1,k,1));,
                            awy = std::abs(vel(i,j-1,k,2)) + 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i,j+1,k,2));)
                        }  
                        else if (!flag(i,j+1,k).isCovered()) 
                        { 
                            //covered left (if possible fish right for second der)
                            if (!flag(i,j+2,k).isCovered()) 
                            {
                                AMREX_D_TERM(
                                uyy = std::abs(vel(i,j,k,0) - 2*vel(i,j+1,k,0) + vel(i,j+2,k,0));,
                                vyy = std::abs(vel(i,j,k,1) - 2*vel(i,j+1,k,1) + vel(i,j+2,k,1));,
                                wyy = std::abs(vel(i,j,k,2) - 2*vel(i,j+1,k,2) + vel(i,j+2,k,2));)
                            }
                                AMREX_D_TERM(
                                auy = 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i,j+1,k,0));,
                                avy = 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i,j+1,k,1));,
                                awy = 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i,j+1,k,2));)
                        }
                        else if (!flag(i,j-1,k).isCovered()) 
                        { 
                            //covered right (if possible fish left for second der)
                            if (!flag(i,j-2,k).isCovered()) 
                            {
                                AMREX_D_TERM(
                                uyy = std::abs(vel(i,j-2,k,0) - 2*vel(i,j-1,k,0) + vel(i,j,k,0));,
                                vyy = std::abs(vel(i,j-2,k,1) - 2*vel(i,j-1,k,1) + vel(i,j,k,1));,
                                wyy = std::abs(vel(i,j-2,k,2) - 2*vel(i,j-1,k,2) + vel(i,j,k,2));)
                            }
                                AMREX_D_TERM(
                                auy = std::abs(vel(i,j-1,k,0)) + 2*std::abs(vel(i,j,k,0));,
                                avy = std::abs(vel(i,j-1,k,1)) + 2*std::abs(vel(i,j,k,1));,
                                awy = std::abs(vel(i,j-1,k,2)) + 2*std::abs(vel(i,j,k,2));)
                        }
                
                        #if (AMREX_IS_3D)
                        if (!flag(i,j,k+1).isCovered() && !flag(i,j,k-1).isCovered()) 
                        { 
                            //no covered neighs
                            uzz = std::abs(vel(i,j,k-1,0) - 2*vel(i,j,k,0) + vel(i,j,k+1,0));
                            vzz = std::abs(vel(i,j,k-1,1) - 2*vel(i,j,k,1) + vel(i,j,k+1,1));
                            wzz = std::abs(vel(i,j,k-1,2) - 2*vel(i,j,k,2) + vel(i,j,k+1,2));
                            auz = std::abs(vel(i,j,k-1,0)) + 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i,j,k+1,0));
                            avz = std::abs(vel(i,j,k-1,1)) + 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i,j,k+1,1));
                            awz = std::abs(vel(i,j,k-1,2)) + 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i,j,k+1,2));
                        }  
                        else if (!flag(i,j,k+1).isCovered()) 
                        { 
                            //covered left (if possible fish right for second der)
                            if (!flag(i,j,k+2).isCovered()) 
                            {
                                uzz = std::abs(vel(i,j,k,0) - 2*vel(i,j,k+1,0) + vel(i,j,k+2,0));
                                vzz = std::abs(vel(i,j,k,1) - 2*vel(i,j,k+1,1) + vel(i,j,k+2,1));
                                wzz = std::abs(vel(i,j,k,2) - 2*vel(i,j,k+1,2) + vel(i,j,k+2,2));
                            }
                            auz = 2*std::abs(vel(i,j,k,0)) + std::abs(vel(i,j,k+1,0));
                            avz = 2*std::abs(vel(i,j,k,1)) + std::abs(vel(i,j,k+1,1));
                            awz = 2*std::abs(vel(i,j,k,2)) + std::abs(vel(i,j,k+1,2));
                        }
                        else if (!flag(i,j,k-1).isCovered()) 
                        { 
                            //covered right (if possible fish left for second der)
                            if (!flag(i,j,k-2).isCovered()) 
                            {
                                uzz = std::abs(vel(i,j,k-2,0) - 2*vel(i,j,k-1,0) + vel(i,j,k,0));
                                vzz = std::abs(vel(i,j,k-2,1) - 2*vel(i,j,k-1,1) + vel(i,j,k,1));
                                wzz = std::abs(vel(i,j,k-2,2) - 2*vel(i,j,k-1,2) + vel(i,j,k,2));
                            }
                            auz = std::abs(vel(i,j,k-1,0)) + 2*std::abs(vel(i,j,k,0));
                            avz = std::abs(vel(i,j,k-1,1)) + 2*std::abs(vel(i,j,k,1));
                            awz = std::abs(vel(i,j,k-1,2)) + 2*std::abs(vel(i,j,k,2));
                        }
                        #endif

                        //actually different number if 2D... 0.5??
                        Real denxx=0.25*aux;
                        Real denxy=0.25*auy;
                        Real denyx=0.25*avx;
                        Real denyy=0.25*avy;
                        #if (AMREX_IS_3D)
                        Real denxz=0.25*auz;
                        Real denyz=0.25*avz;
                        Real denzx=0.25*awx;
                        Real denzy=0.25*awy;
                        Real denzz=0.25*awz;
                        #endif
                        Real smallnumber=1e-8;
                        if(denxx > smallnumber)  derxx= uxx / denxx * dx;
                        if(denxy > smallnumber)  derxy= uyy / denxy * dy;
                        if(denyx > smallnumber)  deryx= vxx / denyx * dx;
                        if(denyy > smallnumber)  deryy= vyy / denyy * dy;
                        #if (AMREX_IS_3D)
                        if(denxz > smallnumber)  derxz= uzz / denxz * dz;
                        if(denyz > smallnumber)  deryz= vzz / denyz * dz;
                        if(denzx > smallnumber)  derzx= wxx / denzx * dx;
                        if(denzy > smallnumber)  derzy= wyy / denzy * dy;
                        if(denzz > smallnumber)  derzz= wzz / denzz * dz;
                        #endif
                        Real maxder=0.0;
                        #if (AMREX_IS_3D)
                        maxder=max(derxx,derxy,deryx,deryy,derzx,derzy);
                        if(!exclude_z_dir) 
                            {maxder=max(maxder,derxz,deryz,derzz);}
                        #else
                        maxder=max(derxx,derxy,deryx,deryy);
                        #endif
                        if (maxder >= dervelerr) 
                            {tag(i,j,k) = tagval;}
                    }
                }
            });
        }
    }

    if (tag_soc) 
    {
        //first we need to compute soc
        // MultiFab socmf;
        // socmf.define(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        // ComputeStateOfCharge(lev, socmf, m_leveldata[lev]->conc);
        auto const& socmf = m_leveldata[lev]->soc;
        #if (USE_OPENMP)
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(m_leveldata[lev]->density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& tag = tags.array(mfi);
            Array4<Real const> const& soc = socmf.const_array(mfi);
            Real socerr = tag_soc ? socerr_v[lev]: std::numeric_limits<Real>::max();
            const EBCellFlagFab& flags = flags_mf[mfi];
            const auto& flag = flags.const_array();
      
            ParallelFor(bx,
            [tag_soc,socerr,tagval,soc,tag,flag]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (tag_soc and soc(i,j,k) > socerr) 
                    {tag(i,j,k) = tagval;}
            });
        }
    }
    
    #if (AMREX_USE_EB)
    // Refine on cut cells
    if (m_refine_cutcells)  
        {amrex::TagCutCells(tags, m_leveldata[lev]->velocity);}
    #endif

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
