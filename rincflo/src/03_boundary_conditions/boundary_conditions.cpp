#include <rincflo.H>

// *********************************************************************************************************************
void Rincflo::InitBoundaryConditions ()
{
    // Process the one boundary condition
    auto f = [this] (std::string const& bcid, Orientation ori)
    {
        // default
        m_bc_density[ori] = 1.0;
        AMREX_D_TERM(
        m_bc_velocity[ori][0] = 0.0;,
        m_bc_velocity[ori][1] = 0.0;,
        m_bc_velocity[ori][2] = 0.0;)
        m_bc_conc[ori].resize(m_nspec,0.0);
        m_bc_epotL[ori]=0.0; 
        m_bc_epotS[ori]=0.0;

        ParmParse pp(bcid);
        std::string bc_type_in = "null";
        pp.query("type", bc_type_in);
        std::string bc_type = amrex::toLower(bc_type_in);

        if (bc_type == "pressure_inflow" || bc_type == "pi")
        {
            m_bc_type[ori] = BC::pressure_inflow;
            pp.get("pressure", m_bc_pressure[ori]);
            // assuming dirichlet for conc...
            pp.queryarr("conc", m_bc_conc[ori], 0, m_nspec);
            // Report the boundary condition
            Print() << format("{:s} set to pressure inflow.\n", bcid.c_str());
            Print() << format("pressure = {:f} Pa.\n", m_bc_pressure[ori]);
            Print() << format("conc = [");
            for (int i=0; i<m_nspec; i++)
            {
                string pad = i<m_nspec-1 ? ", " : "].\n"; 
                Print() << format("{:16.12f}{:s}", m_bc_conc[ori][i], pad);
            }
        }
        else if (bc_type == "pressure_outflow" or bc_type == "po")
        {
            m_bc_type[ori] = BC::pressure_outflow;
            pp.get("pressure", m_bc_pressure[ori]);
            // Report the boundary condition
            Print() << format("{:s} set to pressure outflow, presure = {:f} Pa.\n", bcid.c_str(), m_bc_pressure[ori]);
        }
        else if (bc_type == "mass_inflow" or bc_type == "mi")
        {
            m_bc_type[ori] = BC::mass_inflow;
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                for (int i=0; i < SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }

            pp.query("density", m_bc_density[ori]);
            pp.queryarr("conc", m_bc_conc[ori], 0, m_nspec);
            // Report the boundary condition
            Print() << format("{:s} set to mass inflow.\n", bcid.c_str());
            Print() << format("{:s} conc = [", bcid.c_str());
            for (int i=0; i<m_nspec; i++)
            {
                string pad = i<m_nspec-1 ? ", " : "].\n"; 
                Print() << format("{:12.9f}{:s}", m_bc_conc[ori][i], pad);
            }
        }
        else if (bc_type == "no_slip_wall" or bc_type == "nsw")
        {
            m_bc_type[ori] = BC::no_slip_wall;
            Print() << format("{:s} set to no-slip wall.\n", bcid.c_str());

            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential components of a specified velocity field -- 
            // the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i < SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "slip_wall" or bc_type == "sw")
        {
            m_bc_type[ori] = BC::slip_wall;
            Print() << format("{:s} set to slip wall.\n", bcid.c_str());

            // These values are set by default above - note that we only actually use the zero value 
            // for the normal direction; the tangential components are set to be first order extrap 
            // m_bc_velocity[ori] = {0.0, 0.0, 0.0};
        }
        else if (bc_type == "charging_wall_pot" or bc_type == "cwp")
        {
            m_bc_type[ori] = BC::charging_wall_pot;
            pp.query("epotL", m_bc_epotL[ori]);
            pp.query("epotS", m_bc_epotS[ori]);
            Print() << format("{:s} set to charging (no-slip) wall, fix potential, value for epotS= {:f}, value for epotL = {:f}.\n", 
            bcid.c_str(), m_bc_epotS[ori], m_bc_epotL[ori]);
      
            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential components of a specified velocity field 
            // -- the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i < SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "charging_wall_cur" || bc_type == "cwc")
        {
            m_bc_type[ori] = BC::charging_wall_cur;
            pp.query("epotS", m_bc_epotS[ori]);
            pp.query("epotL", m_bc_epotL[ori]);

            Print() << format("{:s} set to charging (no-slip) wall, fix current, value for (der)epotS={:f}, value for (der)epotL={:f}.\n", 
            bcid.c_str(), m_bc_epotS[ori], m_bc_epotL[ori]);
      
            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential component sof a specified velocity field 
            // -- the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i < SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "mix_wall_pots" || bc_type == "mwps")
        {
            m_bc_type[ori] = BC::mix_wall_potS;
            pp.query("epotS", m_bc_epotS[ori]);
            Print() << format(
                "{:s} set to mix-charging (no-slip) wall, homog. Neum for liquid, fix potential for solid, value epotS= {:f}.\n", 
                bcid.c_str(), m_bc_epotS[ori]);
      
            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential componentsof a specified velocity field 
            // -- the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i<SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "mix_wall_potl" || bc_type == "mwpl")
        {
            m_bc_type[ori] = BC::mix_wall_potL;
            pp.query("epotL", m_bc_epotL[ori]);

            Print() << bcid.c_str() <<" set to mix-charging (no-slip) wall, homog.Neum for solid, fix potential for liquid, value epotL= "
                << m_bc_epotL[ori] << std::endl ;

            if(m_nspec>2) 
                {pp.query("protons", m_bc_conc[ori][2]);}
        
            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential componentsof a specified velocity field 
            // -- the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                    v[ori.coordDir()] = 0.0;
                    for (int i=0; i<SpaceDim; i++)
                        {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "mix_wall_potl_currs" || bc_type == "mwplcs" || bc_type == "mwcspl")
        {
            m_bc_type[ori] = BC::mix_wall_potL_currS;
            pp.query("epotL", m_bc_epotL[ori]);
            pp.query("epotS", m_bc_epotS[ori]);
            Print() << format(
                "{:s} set to mix-charging (no-slip) wall, fix potential for liquid, value epotL={:f} fix current for solid, value epotS= {:f}.\n", 
                bcid.c_str(), m_bc_epotL[ori], m_bc_epotS[ori]);
      
            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential components of a specified velocity field 
            // -- the wall is not allowed to move in the normal direction
            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i<SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else if (bc_type == "mix_wall_currl_pots" or bc_type == "mwclps" or bc_type == "mwpscl")
        {
            m_bc_type[ori] = BC::mix_wall_currL_potS;
            pp.query("epotL", m_bc_epotL[ori]);
            pp.query("epotS", m_bc_epotS[ori]);

            Print() << format(
                "{:s} set to mix-charging (no-slip) wall, fix current for liquid, value epotL= {:f} fix potential for solid, value epotS= {:f}.\n", 
                bcid.c_str(), m_bc_epotL[ori], m_bc_epotS[ori]);

            // Note that m_bc_velocity defaults to 0 above so we are ok if queryarr finds nothing 
            // Here we make sure that we only use the tangential componentsof a specified velocity field 
            // -- the wall is not allowedto move in the normal direction

            vector<Real> v;
            if (pp.queryarr("velocity", v, 0, SpaceDim)) 
            {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i<SpaceDim; i++)
                    {m_bc_velocity[ori][i] = v[i];}
            }
        }
        else
            {m_bc_type[ori] = BC::undefined;}

        if (geom[0].isPeriodic(ori.coordDir())) 
        {
            if (m_bc_type[ori] == BC::undefined) 
                {m_bc_type[ori] = BC::periodic;} 
            else 
                {Abort("Wrong BC type for periodic boundary");}
        }
    };
    AMREX_D_TERM(
    f("xlo", Orientation(Direction::x,Orientation::low));,
    f("ylo", Orientation(Direction::y,Orientation::low));,
    f("zlo", Orientation(Direction::z,Orientation::low));)
    AMREX_D_TERM(
    f("xhi", Orientation(Direction::x,Orientation::high));,
    f("yhi", Orientation(Direction::y,Orientation::high));,
    f("zhi", Orientation(Direction::z,Orientation::high));)
  
    // BC for concentration
    if (m_nspec > 0) 
    {
        Vector<Real> h_data(m_nspec*SpaceDim*2);
        Real* hp = h_data.data();
        for (auto const& v : m_bc_conc) 
        {
            for (auto x : v) 
            {*(hp++) = x;}
        }

        m_bc_conc_raii.resize(m_nspec*SpaceDim*2);
        Real* p = m_bc_conc_raii.data();
        #if (AMREX_USE_GPU)
        Gpu::htod_memcpy
        #else
        std::memcpy
        #endif
        (p, h_data.data(), sizeof(Real)*h_data.size());

        for (int i = 0; i < SpaceDim*2; ++i) 
        {
            m_bc_conc_d[i] = p;
            p += m_nspec;
        }
    }

    // BC for velocity
    {
        m_bcrec_velocity.resize(SpaceDim);
        for (OrientationIter oit; oit; ++oit) 
        {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::pressure_inflow || bct == BC::pressure_outflow)
            {
                if (side == Orientation::low) 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setLo(dir, foextrap);,
                    m_bcrec_velocity[1].setLo(dir, foextrap);,
                    m_bcrec_velocity[2].setLo(dir, foextrap);)
                } 
                else 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setHi(dir, foextrap);,
                    m_bcrec_velocity[1].setHi(dir, foextrap);,
                    m_bcrec_velocity[2].setHi(dir, foextrap);)
                }
            }
            else if (   bct == BC::mass_inflow || bct == BC::no_slip_wall || bct == BC::charging_wall_pot || 
                        bct == BC::charging_wall_cur || bct == BC::mix_wall_potS || bct == BC::mix_wall_potL || 
                        bct == BC::mix_wall_potL_currS || bct == BC::mix_wall_currL_potS)
            {
                if (side == Orientation::low) 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setLo(dir, ext_dir);,
                    m_bcrec_velocity[1].setLo(dir, ext_dir);,
                    m_bcrec_velocity[2].setLo(dir, ext_dir);)
                } 
                else 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setHi(dir, ext_dir);,
                    m_bcrec_velocity[1].setHi(dir, ext_dir);,
                    m_bcrec_velocity[2].setHi(dir, ext_dir);)
                }
            }
            else if (bct == BC::slip_wall)
            {
                if (side == Orientation::low) 
                {
                    // Tangential directions have hoextrap
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setLo(dir, hoextrap);,
                    m_bcrec_velocity[1].setLo(dir, hoextrap);,
                    m_bcrec_velocity[2].setLo(dir, hoextrap);)

                    // Only normal direction has ext_dir
                    m_bcrec_velocity[dir].setLo(dir, ext_dir);
                } 
                else 
                {
                    // Tangential directions have hoextrap
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setHi(dir, hoextrap);,
                    m_bcrec_velocity[1].setHi(dir, hoextrap);,
                    m_bcrec_velocity[2].setHi(dir, hoextrap);)

                    // Only normal direction has ext_dir
                    m_bcrec_velocity[dir].setHi(dir, ext_dir);
                }
            }
            else if (bct == BC::periodic)
            {
                if (side == Orientation::low) 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setLo(dir, int_dir);,
                    m_bcrec_velocity[1].setLo(dir, int_dir);,
                    m_bcrec_velocity[2].setLo(dir, int_dir);)
                } 
                else 
                {
                    AMREX_D_TERM(
                    m_bcrec_velocity[0].setHi(dir, int_dir);,
                    m_bcrec_velocity[1].setHi(dir, int_dir);,
                    m_bcrec_velocity[2].setHi(dir, int_dir);)
                }
            }
        }
        m_bcrec_velocity_d.resize(SpaceDim);
        #if (AMREX_USE_GPU)
        Gpu::htod_memcpy
        #else
        std::memcpy
        #endif
        (m_bcrec_velocity_d.data(), m_bcrec_velocity.data(), sizeof(BCRec)*SpaceDim);
    }

  // BC for density
  {
    m_bcrec_density.resize(1);
    for (OrientationIter oit; oit; ++oit) {
      Orientation ori = oit();
      int dir = ori.coordDir();
      Orientation::Side side = ori.faceDir();
      auto const bct = m_bc_type[ori];
      if (bct == BC::pressure_inflow  or
      bct == BC::pressure_outflow or
      bct == BC::slip_wall        or
      bct == BC::no_slip_wall     or
      bct == BC::charging_wall_pot or
      bct == BC::charging_wall_cur or
      bct == BC::mix_wall_potS or
      bct == BC::mix_wall_potL or
      bct == BC::mix_wall_potL_currS or
      bct == BC::mix_wall_currL_potS)
    {
      if (side == Orientation::low) {
        m_bcrec_density[0].setLo(dir, foextrap);
      } else {
        m_bcrec_density[0].setHi(dir, foextrap);
      }
    }
      else if (bct == BC::mass_inflow)
    {
      if (side == Orientation::low) {
        m_bcrec_density[0].setLo(dir, ext_dir);
      } else {
        m_bcrec_density[0].setHi(dir, ext_dir);
      }
    }
      else if (bct == BC::periodic)
    {
      if (side == Orientation::low) {
        m_bcrec_density[0].setLo(dir, int_dir);
      } else {
        m_bcrec_density[0].setHi(dir, int_dir);
      }
    }
    }
    m_bcrec_density_d.resize(1);
    #if (AMREX_USE_GPU)
    Gpu::htod_memcpy
    #else
    std::memcpy
    #endif
      (m_bcrec_density_d.data(), m_bcrec_density.data(), sizeof(BCRec));
  }

    // BC for concentration
    if (m_nspec > 0)
    {
        m_bcrec_conc.resize(m_nspec);
        for (OrientationIter oit; oit; ++oit) 
        {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if(bct == BC::mix_wall_potL)
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setLo(dir, foextrap);}
                    if(m_fix_protons_mem && m_nspec>2)
                        {m_bcrec_conc[2].setLo(dir, ext_dir);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setHi(dir, foextrap);}
                    if(m_fix_protons_mem && m_nspec>2)
                        {m_bcrec_conc[2].setHi(dir, ext_dir);}
                }
            }
            else if (bct == BC::pressure_outflow ||
                bct == BC::slip_wall        ||
                bct == BC::no_slip_wall     ||
                bct == BC::charging_wall_pot ||
                bct == BC::charging_wall_cur ||
                bct == BC::mix_wall_potS ||
                bct == BC::mix_wall_potL_currS ||
                bct == BC::mix_wall_currL_potS )
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setLo(dir, foextrap);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setHi(dir, foextrap);}
                }
            }
            else if (bct == BC::pressure_inflow  || bct == BC::mass_inflow)
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setLo(dir, ext_dir);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setHi(dir, ext_dir);}
                }
            }
            else if (bct == BC::periodic)
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setLo(dir, int_dir);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_conc) 
                        {b.setHi(dir, int_dir);}
                }
            }
        }
        m_bcrec_conc_d.resize(m_nspec);
        #if (AMREX_USE_GPU)
        Gpu::htod_memcpy
        #else
        std::memcpy
        #endif
            (m_bcrec_conc_d.data(), m_bcrec_conc.data(), sizeof(BCRec)*m_nspec);
    }

    // BC for force
    {
        const int ncomp = max(m_nspec, SpaceDim);
        m_bcrec_force.resize(ncomp);
        for (OrientationIter oit; oit; ++oit) 
        {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::periodic)
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_force) 
                        {b.setLo(dir, int_dir);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_force) 
                        {b.setHi(dir, int_dir);}
                }
            }
            else
            {
                if (side == Orientation::low) 
                {
                    for (auto& b : m_bcrec_force) 
                        {b.setLo(dir, foextrap);}
                } 
                else 
                {
                    for (auto& b : m_bcrec_force) 
                        {b.setHi(dir, foextrap);}
                }
            }
        }
        m_bcrec_force_d.resize(ncomp);
        #if (AMREX_USE_GPU)
        Gpu::htod_memcpy
        #else
        std::memcpy
        #endif
            (m_bcrec_force_d.data(), m_bcrec_force.data(), sizeof(BCRec)*ncomp);
    }

    // Read the membrane voltage when it's applicable
    // ParmParse ppr("rincflo");
    // ppr.query("V_membrane", m_V_membrane);

    // BC for epotL
    {
        m_bcrec_epotL.resize(1);
        for (OrientationIter oit; oit; ++oit) 
        {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::pressure_inflow  ||
                bct == BC::pressure_outflow ||
                bct == BC::slip_wall        ||
                bct == BC::no_slip_wall     ||
                bct == BC::mass_inflow ||
                bct == BC::mix_wall_potS )
            {
                if (side == Orientation::low) 
                    {m_bcrec_epotL[0].setLo(dir, foextrap);} 
                else 
                    {m_bcrec_epotL[0].setHi(dir, foextrap);}
            }
                else if (bct == BC::charging_wall_pot ||
                    bct == BC::charging_wall_cur ||
                    bct == BC::mix_wall_potL ||
                    bct == BC::mix_wall_potL_currS ||
                    bct == BC::mix_wall_currL_potS)
                {
                    if (side == Orientation::low) 
                        {m_bcrec_epotL[0].setLo(dir, ext_dir);} 
                    else 
                        {m_bcrec_epotL[0].setHi(dir, ext_dir);
                }
            }
            else if (bct == BC::periodic)
            {
                if (side == Orientation::low) 
                    {m_bcrec_epotL[0].setLo(dir, int_dir);} 
                else 
                    {m_bcrec_epotL[0].setHi(dir, int_dir);}
            }
        }
        m_bcrec_epotL_d.resize(1);
        #if (AMREX_USE_GPU)
        Gpu::htod_memcpy
        #else
        std::memcpy
        #endif
            (m_bcrec_epotL_d.data(), m_bcrec_epotL.data(), sizeof(BCRec));
    }

    // BC for epotS
    {
        m_bcrec_epotS.resize(1);
        for (OrientationIter oit; oit; ++oit) 
        {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::pressure_inflow  ||
                bct == BC::pressure_outflow ||
                bct == BC::slip_wall        ||
                bct == BC::no_slip_wall     ||
                bct == BC::mass_inflow      ||
                bct == BC::mix_wall_potL)
            {
                if (side == Orientation::low) 
                    {m_bcrec_epotS[0].setLo(dir, foextrap);} 
                else 
                    {m_bcrec_epotS[0].setHi(dir, foextrap);}
                }
                else if (bct == BC::charging_wall_pot ||
                bct == BC::charging_wall_cur ||
                bct == BC::mix_wall_potS ||
                bct == BC::mix_wall_potL_currS ||
                bct == BC::mix_wall_currL_potS)
                {
                    if (side == Orientation::low) 
                        {m_bcrec_epotS[0].setLo(dir, ext_dir);} 
                    else 
                        {m_bcrec_epotS[0].setHi(dir, ext_dir);}
                }
                else if (bct == BC::periodic)
                {
                    if (side == Orientation::low) 
                        {m_bcrec_epotS[0].setLo(dir, int_dir);} 
                    else 
                        {m_bcrec_epotS[0].setHi(dir, int_dir);}
                }
            }
            m_bcrec_epotS_d.resize(1);
            #if (AMREX_USE_GPU)
            Gpu::htod_memcpy
            #else
            std::memcpy
            #endif
                (m_bcrec_epotS_d.data(), m_bcrec_epotS.data(), sizeof(BCRec));
    }
}
