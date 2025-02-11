#include <rincflo.H>

void
Rincflo::ComputeDivTau(
    Vector<MultiFab*> const& divtau, 
    Vector<MultiFab const*> const& vel,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& eta)
{
    if (use_tensor_correction) 
    {
        get_diffusion_tensor_op()->ComputeDivTau(divtau, vel, density, eta);
        AMREX_EB_ONLY(EB_set_covered(*divtau[0], 0.0));

        Vector<MultiFab*> divtau_scal; 
        divtau_scal.push_back(new MultiFab(grids[0], dmap[0], divtau[0]->nComp(), divtau[0]->nGrow(),MFInfo(),*m_factory[0]));
        divtau_scal[0]->setVal(0.);

        get_diffusion_scalar_op()->ComputeDivTau({divtau_scal}, vel, density, eta);
        AMREX_EB_ONLY(EB_set_covered(*divtau_scal[0], 0.0));

        // Define divtau to be (divtau_full - divtau_separate)
        if (m_verbose > 0)
            {Print() << " ... Defining divtau as the difference between tensor and scalar versions\n";}

        // Print() << "X-comp: Norm of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(0) << " " << divtau_scal[0]->norm0(0) << std::endl;
        // Print() << "Y-comp: Norm of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(1) << " " << divtau_scal[0]->norm0(1) << std::endl;
        // Print() << "Z-comp: Norm of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(2) << " " << divtau_scal[0]->norm0(2) << std::endl;

        divtau[0]->Saxpy(*divtau[0], -1.0, *divtau_scal[0], 0, 0, SpaceDim, 0);

        // Print() << "X-comp: Norm of difference of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(0) << std::endl;
        // Print() << "Y-comp: Norm of difference of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(1) << std::endl;
        // Print() << "Z-comp: Norm of difference of tensor apply vs scalar apply " << 
        //                    divtau[0]->norm0(2) << std::endl;

    } 
    else if (use_tensor_solve) 
        {get_diffusion_tensor_op()->ComputeDivTau(divtau, vel, density, eta);} 
    else 
        {get_diffusion_scalar_op()->ComputeDivTau(divtau, vel, density, eta);}
}

// *********************************************************************************************************************
void
Rincflo::ComputeLaplacian(
    Vector<MultiFab*> const& laps, 
    Vector<MultiFab const*> const& scalar,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& eta)
{
    get_diffusion_scalar_op()->ComputeLaplacian(laps, scalar, density, eta);
}

// *********************************************************************************************************************
void
Rincflo::DiffuseScalar(
    Vector<MultiFab*> const& scalar,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt_diff)
{
    get_diffusion_scalar_op()->DiffuseScalar(scalar, density, eta, dt_diff);
}

// *********************************************************************************************************************
void
Rincflo::DiffuseVelocity(
    Vector<MultiFab*> const& vel,
    Vector<MultiFab*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt_diff)
{
    if (use_tensor_correction) 
    {
        //        Print() << " \n ... diffuse components separately but with tensor terms added explicitly... " << std::endl;
        get_diffusion_scalar_op()->diffuse_vel_components(vel, density, eta, dt_diff);
    } 
    else if (use_tensor_solve) 
        {get_diffusion_tensor_op()->DiffuseVelocity(vel, density, eta, dt_diff);} 
    else 
        {get_diffusion_scalar_op()->diffuse_vel_components(vel, density, eta, dt_diff);}
}

// *********************************************************************************************************************
DiffusionTensorOp*
Rincflo::get_diffusion_tensor_op ()
{
    if (!m_diffusion_tensor_op) m_diffusion_tensor_op.reset(new DiffusionTensorOp(this));
    return m_diffusion_tensor_op.get();
}

// *********************************************************************************************************************
DiffusionScalarOp*
Rincflo::get_diffusion_scalar_op ()
{
    if (!m_diffusion_scalar_op) m_diffusion_scalar_op.reset(new DiffusionScalarOp(this));
    return m_diffusion_scalar_op.get();
}

// *********************************************************************************************************************
Vector<Array<LinOpBCType,SpaceDim> >
Rincflo::get_diffuse_tensor_bc (Orientation::Side side) const noexcept
{
    Vector<Array<LinOpBCType,SpaceDim>> r(SpaceDim);
    for (int dir = 0; dir < SpaceDim; ++dir) 
    {
        if (Geom(0).isPeriodic(dir)) 
        {
            AMREX_D_TERM( 
            r[0][dir] = LinOpBCType::Periodic;,
            r[1][dir] = LinOpBCType::Periodic;,
            r[2][dir] = LinOpBCType::Periodic;)
        } 
        else 
        {
            auto bc = m_bc_type[Orientation(dir,side)];
            switch (bc)
            {
                case BC::pressure_inflow:
                case BC::pressure_outflow:
                {
                    // All three components are Neumann
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Neumann;,
                    r[1][dir] = LinOpBCType::Neumann;,
                    r[2][dir] = LinOpBCType::Neumann;)
                    break;
                }
                case BC::mass_inflow:
                case BC::no_slip_wall:
                case BC::charging_wall_pot:
                case BC::charging_wall_cur:
                case BC::mix_wall_potS:
                case BC::mix_wall_potL:
                case BC::mix_wall_potL_currS:
                case BC::mix_wall_currL_potS:
                {
                    // All three components are Dirichlet
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Dirichlet;,
                    r[1][dir] = LinOpBCType::Dirichlet;,
                    r[2][dir] = LinOpBCType::Dirichlet;)
                    break;
                }
                case BC::slip_wall:
                {
                    // Tangential components are Neumann
                    // Normal     component  is  Dirichlet
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Neumann;,
                    r[1][dir] = LinOpBCType::Neumann;,
                    r[2][dir] = LinOpBCType::Neumann;)
                    r[dir][dir] = LinOpBCType::Dirichlet;
                    break;
                }
                default:
                    Abort("get_diffuse_tensor_bc: undefined BC type");
            };
        }
    }
    return r;
}

// *********************************************************************************************************************
Array<LinOpBCType,SpaceDim>
Rincflo::get_DiffuseVelocity_bc (Orientation::Side side, int comp) const noexcept
{
    Vector<Array<LinOpBCType,SpaceDim>> r(SpaceDim);
    for (int dir = 0; dir < SpaceDim; ++dir) 
    {
        if (Geom(0).isPeriodic(dir)) 
        {
            AMREX_D_TERM(
            r[0][dir] = LinOpBCType::Periodic;,
            r[1][dir] = LinOpBCType::Periodic;,
            r[2][dir] = LinOpBCType::Periodic;)
        } 
        else 
        {
            auto bc = m_bc_type[Orientation(dir,side)];
            switch (bc)
            {
                case BC::pressure_inflow:
                case BC::pressure_outflow:
                {
                    // All three components are Neumann
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Neumann;,
                    r[1][dir] = LinOpBCType::Neumann;,
                    r[2][dir] = LinOpBCType::Neumann;)
                    break;
                }
                case BC::mass_inflow:
                case BC::no_slip_wall:
                case BC::charging_wall_pot:
                case BC::charging_wall_cur:
                case BC::mix_wall_potS:
                case BC::mix_wall_potL:
                case BC::mix_wall_potL_currS:
                case BC::mix_wall_currL_potS:
                {
                    // All three components are Dirichlet
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Dirichlet;,
                    r[1][dir] = LinOpBCType::Dirichlet;,
                    r[2][dir] = LinOpBCType::Dirichlet;)
                    break;
                }
                case BC::slip_wall:
                {
                    // Tangential components are Neumann
                    // Normal     component  is  Dirichlet
                    AMREX_D_TERM(
                    r[0][dir] = LinOpBCType::Neumann;,
                    r[1][dir] = LinOpBCType::Neumann;,
                    r[2][dir] = LinOpBCType::Neumann;)

                    r[dir][dir] = LinOpBCType::Dirichlet;
                    break;
                }
                default:
                    Abort("get_DiffuseVelocity_bc: undefined BC type");
            };
        }
    }
    return r[comp];
}

// *********************************************************************************************************************
Array<LinOpBCType,SpaceDim>
Rincflo::get_DiffuseScalar_bc (Orientation::Side side) const noexcept
{
    Array<LinOpBCType,SpaceDim> r;
    for (int dir = 0; dir < SpaceDim; ++dir) 
    {
        if (Geom(0).isPeriodic(dir)) 
            {r[dir] = LinOpBCType::Periodic;} 
        else 
        {
            auto bc = m_bc_type[Orientation(dir,side)];
            switch (bc)
            {
                case BC::pressure_outflow:
                case BC::slip_wall:
                case BC::no_slip_wall:
                case BC::charging_wall_pot:
                case BC::charging_wall_cur:
                case BC::mix_wall_potS:
                case BC::mix_wall_potL:
                case BC::mix_wall_potL_currS:
                case BC::mix_wall_currL_potS:
                {
                    r[dir] = LinOpBCType::Neumann;
                    break;
                }
                case BC::pressure_inflow:
                case BC::mass_inflow:
                {
                    r[dir] = LinOpBCType::Dirichlet;
                    break;
                }
                default:
                    Abort("get_DiffuseScalar_bc: undefined BC type");
            };
        }
    }
    return r;
}

// *********************************************************************************************************************
Array<MultiFab,SpaceDim>
Rincflo::average_velocity_eta_to_faces (int lev, MultiFab const& cc_eta) const
{
    const auto& ba = cc_eta.boxArray();
    const auto& dm = cc_eta.DistributionMap();
    const auto& fact = cc_eta.Factory();
    Array<MultiFab,SpaceDim> r{
        AMREX_D_DECL(
        MultiFab(convert(ba,IntVect::TheDimensionVector(0)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(1)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(2)), dm, 1, 0, MFInfo(), fact))};
        #if (AMREX_USE_EB)
        // Note we use the conc bc's here only to know when the bc is ext_dir (this should be the same for conc and eta)
        EB_interp_CellCentroid_to_FaceCentroid (cc_eta, GetArrOfPtrs(r), 0, 0, 1, geom[lev], get_conc_bcrec());
        #else
        amrex::average_cellcenter_to_face(GetArrOfPtrs(r), cc_eta, Geom(lev));
        #endif
        FixupEtaOnDomainFaces(lev, r, cc_eta);
        return r;
    }

// *********************************************************************************************************************
Array<MultiFab,SpaceDim>
Rincflo::AverageScalarEtaToFaces (int lev, int comp, MultiFab const& cc_eta) const
{
    const auto& ba = cc_eta.boxArray();
    const auto& dm = cc_eta.DistributionMap();
    const auto& fact = cc_eta.Factory();
    MultiFab cc(cc_eta, amrex::make_alias, comp, 1);
    Array<MultiFab,SpaceDim> r{
        AMREX_D_DECL(
        MultiFab(convert(ba,IntVect::TheDimensionVector(0)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(1)), dm, 1, 0, MFInfo(), fact),
        MultiFab(convert(ba,IntVect::TheDimensionVector(2)), dm, 1, 0, MFInfo(), fact))};
        #if (AMREX_USE_EB)
        EB_interp_CellCentroid_to_FaceCentroid (cc, GetArrOfPtrs(r), 0, 0, 1, geom[lev],  get_conc_bcrec());
        #else
        amrex::average_cellcenter_to_face(GetArrOfPtrs(r), cc, Geom(lev));
        #endif
        FixupEtaOnDomainFaces(lev, r, cc);
        return r;
}

// *********************************************************************************************************************
void
Rincflo::FixupEtaOnDomainFaces (int lev, Array<MultiFab,SpaceDim>& fc, MultiFab const& cc) const
{
    const Geometry& gm = Geom(lev);
    const Box& domain = gm.Domain();
    MFItInfo mfi_info{};
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
    #if (USE_OPENMP)
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(cc,mfi_info); mfi.isValid(); ++mfi) 
    {
        Box const& bx = mfi.validbox();
        Array4<Real const> const& cca = cc.const_array(mfi);

        int idim = 0;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                };
                ParallelFor(amrex::bdryLo(bx, idim), func);
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i-1,j,k);
                };
                ParallelFor(amrex::bdryHi(bx, idim), func);
            }
        }

        idim = 1;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                };
                ParallelFor(amrex::bdryLo(bx, idim), func);
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j-1,k);
                };
                ParallelFor(amrex::bdryHi(bx, idim), func);
            }
        }

        #if (AMREX_IS_3D)
        idim = 2;
        if (!gm.isPeriodic(idim)) 
        {
            Array4<Real> const& fca = fc[idim].array(mfi);
            if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k);
                };
                ParallelFor(amrex::bdryLo(bx, idim), func);
            }
            if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
            {
                auto func =
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fca(i,j,k) = cca(i,j,k-1);
                };
                ParallelFor(amrex::bdryHi(bx, idim), func);
            }
        }
        #endif
    }
}
