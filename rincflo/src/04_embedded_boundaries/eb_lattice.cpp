#include <rincflo.H>

/********************************************************************************
 * Function to read-in from files and make EB lattice                           *
 ********************************************************************************/
void Rincflo::MakeEB_lattice()
{
    #define FUNC_NAME "Rincflo::MakeEB_lattice"
    BL_PROFILE(FUNC_NAME);
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    FILE *infp;
    char line[1000],*itrash;
    int Nobj;
  
    ParmParse pp("amr");
    int amr_max_lev;
    pp.query("max_level", amr_max_lev);

    ParmParse pp2("geo_eb");
    int required_coarsening_level = amr_max_lev;
    pp2.query("required_coarsening_level", required_coarsening_level);
    // The required_corsening_level MUST be at least the same as the amr_max_lev
    required_coarsening_level = std::max(amr_max_lev, required_coarsening_level);
    int max_coarsening_level=20;
    pp2.query("max_coarsening_level", max_coarsening_level);

    // Status message for EB lattice generation
    Print() << format("Rincflo::MakeEB_lattice - amr_max_lev={:d}, required_coarsening_level={:d}, max_coarsening_level={:d}\n",
        amr_max_lev, required_coarsening_level, max_coarsening_level);

    if ((required_coarsening_level < amr_max_lev ) && m_build_coarse_level_by_coarsening)
    {
        Print() << format("Rincflo::MakeEB_lattice - Error! amr_max={:d}, req={:d}, max_coarse={:d}\n", 
            amr_max_lev, required_coarsening_level, max_coarsening_level);
    }

    // Get information from inputs file.                               
    bool fluid_inside=false;
    pp2.query("fluid_inside", fluid_inside);
    std::string inputfilename;
    pp2.query("inputfilename", inputfilename);

    // If the inputfilename is empty, then we have no EB geometry to build
    if (inputfilename.empty())
    {
        Print() << "No inputfilename specified.  No EB geometry to build.\n";
        return;
    }

    // Get the extension of the filename to infer the style of input file
    // Allowed extensions are .cub, .sph, .rod, .elrod
    std::filesystem::path filepath {inputfilename};
    auto ftype = filepath.extension().string();
    auto ftype_allowed = std::set<std::string> {".cub", ".sph", ".rod", ".elrod"};
    // Check that this is an allowed file type
    if (ftype_allowed.find(ftype) == ftype_allowed.end())
        {Abort("error in inputfilename.  Must be one of .cub, .sph, .rod, .elrod.\n");}
    // Open the file and check it's readable
    infp = fopen(inputfilename.c_str(), "r");
    if(infp==NULL) 
        {Abort("error in opening inputfile for lattice geometry");}

    // inputfile format:
    // Nobj
    // Lx Ly Lz (or other format for box info... not checked anyway)
    // color x y z s Mxx Mxy Mxz Myx Myy Myz Mzx Mzy Mzz  (if .cub)
    // color x y z D                                      (if .sph)
    // color x y z ux uy uz D L                           (if .rod)
    // color x y z ux uy uz D L sx sy sz                  (if .elrod)
    // For 2D geometries, can put placeholder values in for z

    itrash = fgets(line, sizeof(line), infp);
    if(itrash == NULL)
        {Abort("error in reading");}
    sscanf(line,"%d\n",&Nobj);
    // neglect info on box (not checking if ok with simulation box)
    itrash = fgets(line,sizeof(line),infp);
    if(itrash == NULL) 
        {Abort("error in reading");}
    DEBUG_PRINT(std::format("Input file {:s} has type {:s} with {:d} objects.\n", inputfilename, ftype, Nobj));

    if (ftype == ".cub")
    {
        double x,y,z,xA,yA,zA,xB,yB,zB,lxx,lyy,lzz;
        Vector<EB2::BoxIF> vobjs;
    
        for(int i=0;i<Nobj;i++)
        {
            // read & make single object
            itrash=fgets(line,sizeof(line),infp);
            if(itrash==NULL) 
                {Abort("error in reading .cub data.");}
            sscanf(line,"%*c %lf %lf %lf %*f %lf %*f %*f %*f %lf %*f %*f %*f %lf\n",&x,&y,&z,&lxx,&lyy,&lzz);
            xA=x-0.5*lxx;    yA=y-0.5*lyy;    zA=z-0.5*lzz;
            xB=x+0.5*lxx;    yB=y+0.5*lyy;    zB=z+0.5*lzz;
    
            EB2::BoxIF prismbase({AMREX_D_DECL(xA,yA,zA)}, {AMREX_D_DECL(xB,yB,zB)}, fluid_inside);
            vobjs.push_back(prismbase);
        }
    
        UnionListIF<EB2::BoxIF> allobjs(vobjs);
        // Generate GeometryShop
        auto gshop = EB2::makeShop(allobjs);
        // Build index space
        EB2::Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level);    
    } //*.cub

    else if (ftype== ".sph")
    {
        double x, y, z, diam, radius;
        Vector<EB2::SphereIF> vobjs;
        for(int i=0; i<Nobj; i++)
        {
            itrash=fgets(line,sizeof(line),infp);
            if(itrash==NULL)
                {Abort("error in reading .sph data.");}
            sscanf(line,"%*c %lf %lf %lf %lf\n", &x, &y, &z, &diam);
            radius = 0.5*diam;

            EB2::SphereIF sph(radius, {AMREX_D_DECL(x,y,z)}, fluid_inside);
            vobjs.push_back(sph);
        }
        // Collection of all the sphere objects
        UnionListIF<EB2::SphereIF> allobjs(vobjs);
        // Generate GeometryShop
        auto gshop = EB2::makeShop(allobjs);
        // Build index space
        Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level);    
    } // *.sph

    else if (ftype== ".rod")    
    {
        // Description of the current cylinder being processed
        double x, y, z, ux, uy, uz, diam, radius, length;
        Vector<EB2::CylinderIF> vobjs;
        for(int i=0; i<Nobj; ++i)
        {
            // read & make single object
            itrash=fgets(line,sizeof(line),infp);
            if(itrash==NULL)
                {Abort("error in reading .rod data.");}
            sscanf(line,"%*c %lf %lf %lf %lf %lf %lf %lf %lf\n", &x, &y, &z, &ux, &uy, &uz, &diam, &length);
            radius=0.5*diam;
            int dir=-1;

            // Only axis-aligned cylinders currently supported, i.e. exactly one of the three components of the direction vector is nonzero
            int n_u_nonzero = 0;
            if (ux != 0.0) {n_u_nonzero += 1;}
            if (uy != 0.0) {n_u_nonzero += 1;}
            if (uz != 0.0) {n_u_nonzero += 1;}
            if (n_u_nonzero != 1)
                {Abort("error in reading .rod data.  Only axis-aligned cylinders currently supported.");}

            // Pick the non-zero coordinate axis, disregarding the sign
            if (ux != 0.0) {dir = 0;}
            if (uy != 0.0) {dir = 1;}
            if (uz != 0.0) {dir = 2;}

            // Build the cylinder and add it to the list of objects
            EB2::CylinderIF cyl(radius, length, dir, {AMREX_D_DECL(x,y,z)}, fluid_inside);
            vobjs.push_back(cyl);

            // Report details of each cylinder built if verbosity is high enough
            if (m_verbose > 1)
            {
                Print() << format("EB2::CylinderIF: i={:3d}, radius={:8f}, length={:f}, dir={:d}, center=({:10f},{:10f},{:10f}), fluid_inside={:d}\n",
                    i, radius, length, dir, x, y, z, fluid_inside);
            }

        }
        if (m_verbose)
            {Print() << format("Completed reading of {:d} cylinder objects.\n", Nobj);}
        // Collection of all the cylinder objects
        UnionListIF<EB2::CylinderIF> allobjs(vobjs);
        if (m_verbose)
        {
            Print() << format("Built {:d} cylinder objects. Runtime {:.0f} sec.\n",             
            Nobj, ParallelDescriptor::second());
        }

        // Generate GeometryShop
        auto gshop = EB2::makeShop(allobjs);
        if (m_verbose)
            {Print() << format("Completed EB2::EB2::makeShop(allobjs). Runtime {:.0f} sec.\n", second());}

        // Build index space
        Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level, m_ngrow, m_build_coarse_level_by_coarsening);

        if (m_verbose)
        {
            Print() << format("Completed build of index space with Build(). Runtime {:.0f} sec.\n", 
                ParallelDescriptor::second());
        }
    } //*.rod
  
    else if (ftype == ".elrod")    
    {
        double x,y,z,ux,uy,uz,diam,radius,length,sx,sy,sz;
        Vector<EB2::EllipCylIF> vobjs;

        for(int i=0;i<Nobj;i++)
        {
            // read & make single object
            itrash=fgets(line,sizeof(line),infp);
            if(itrash==NULL) 
                {Abort("error in reading");}
            sscanf(line,"%*c %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",&x,&y,&z,&ux,&uy,&uz,&diam,&length,&sx,&sy,&sz);
            radius=0.5*diam;
            int dir=-1;
            if(ux>uy and ux>uz)
                {dir=0;}
            else if(uy>ux and uy>uz)
                {dir=1;}
            else if(uz>ux and uz>uy)
                {dir=2;}

            EB2::EllipCylIF elcyl ({AMREX_D_DECL(radius*sx,radius*sy,radius*sz)}, length, dir, {AMREX_D_DECL(x,y,z)}, fluid_inside);
            vobjs.push_back(elcyl);
        }

        UnionListIF<EB2::EllipCylIF> allobjs(vobjs);
        // Generate GeometryShop
        auto gshop = EB2::makeShop(allobjs);
        // Build index space
        Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level);    

    } //*.elrod

    fclose(infp);  
    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
