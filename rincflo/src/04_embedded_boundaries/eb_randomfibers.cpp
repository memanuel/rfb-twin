#include <rincflo.H>

/********************************************************************************
 *                                                                              *
 * Function to read-in from files and make fibers  EB                           *
 *                                                                              *
 ********************************************************************************/
void Rincflo::MakeEB_random_fibers()
{
    FILE *infp;
    char line[1000], *itrash;
    int Nrods;

    ParmParse pp("amr");
    int amr_max_lev;
    pp.query("max_level", amr_max_lev);

    ParmParse pp2("geo_eb");
    int required_coarsening_level=amr_max_lev;
    pp2.query("req_lev", required_coarsening_level);
    int max_coarsening_level=20;
    pp2.query("max_lev", max_coarsening_level);

    if (required_coarsening_level < amr_max_lev )    
        {Print() << "ERROR?? amr_max= " << amr_max_lev << " req= " << required_coarsening_level << " max_coarse= " << max_coarsening_level << std::endl;}

    // Get information from inputs file.                               
    bool fluid_inside=false;
    pp2.query("fluid_inside", fluid_inside);
    std::string inputfilename;
    pp2.query("inputfilename", inputfilename);

    if(inputfilename == "initfibers.rod") {}
    else {Abort("error in inputfilename");}
  
    infp = fopen(inputfilename.c_str(), "r");
    if(infp==NULL) 
        {Abort("error in opening file inputfile (e.g. initfibers.rod)");}
    //inputfile format:
    //Nrods
    //Lx Ly Lz (or other forma for box info... not checked anyway)
    //color x y z ux uy uz D L                           (if .rod)


    itrash = fgets(line, sizeof(line), infp);
    if (itrash==NULL) 
        {Abort("error in reading");}
    sscanf(line,"%d\n",&Nrods);
    itrash = fgets(line,sizeof(line),infp); //neglect info on box (not checking if ok with simulation box)
    if(itrash==NULL) 
        {Abort("error in reading");}

    // spherocylinders
    if (inputfilename== "initfibers.rod")    
    {
        double x,y,z,ux,uy,uz,diam,radius,length;
        double xsA,ysA,zsA,xsB,ysB,zsB;

        #if (AMREX_IS_2D)
        Vector< EB2::UnionIF< EB2::TranslationIF<EB2::RotationIF<EB2::CylinderIF>>, EB2::SphereIF, EB2::SphereIF > > vobjs;
        #else
        Vector< EB2::UnionIF< EB2::TranslationIF<EB2::RotationIF<EB2::RotationIF<EB2::CylinderIF>>>, EB2::SphereIF, EB2::SphereIF > > vobjs;
        #endif
        
        for(int i=0;i<Nrods;i++)
        {
            // read & make single object
            itrash=fgets(line,sizeof(line),infp);
            if(itrash==NULL) Abort("error in reading");
            sscanf(line,"%*c %lf %lf %lf %lf %lf %lf %lf %lf\n",&x,&y,&z,&ux,&uy,&uz,&diam,&length);
            radius=0.5*diam;
            xsA=x+0.5*ux*length;      ysA=y+0.5*uy*length;      zsA=z+0.5*uz*length;
            xsB=x-0.5*ux*length;      ysB=y-0.5*uy*length;      zsB=z-0.5*uz*length;

            //Left-handed rotation system in AMReX
            #if (AMREX_IS_2D)
            double angle;
            int dir=0;
            if(uy>0) angle=acos(ux/sqrt(ux*ux+uy*uy));
            else  angle=-acos(ux/sqrt(ux*ux+uy*uy));
            
            EB2::CylinderIF cylbase(radius, length, 0, {AMREX_D_DECL(0,0,0)}, fluid_inside);
            auto firstrot = rotate(cylbase,angle,dir);
            auto cyl = translate(firstrot,{AMREX_D_DECL(x,y,z)});
            EB2::SphereIF sphA(radius, {AMREX_D_DECL(xsA,ysA,zsA)}, fluid_inside);
            EB2::SphereIF sphB(radius, {AMREX_D_DECL(xsB,ysB,zsB)}, fluid_inside);
            auto scyl = EB2::makeUnion(cyl,sphA,sphB);
            vobjs.push_back(scyl);
            #else
            double angle1, angle2;
            int dir1, dir2;
            dir1=1;
            dir2=2;
            if(uz>=1.0)  angle1=0.0;
            else if (uz<=-1.0) angle1=M_PI;
            else  angle1=acos(uz);
            if (ux*ux+uy*uy<1e-20) angle2=0.0;
            else if(uy>0) angle2=acos(ux/sqrt(ux*ux+uy*uy));
            else  angle2=-acos(ux/sqrt(ux*ux+uy*uy));
            
            EB2::CylinderIF cylbase(radius, length, 2, {0,0,0}, fluid_inside);
            auto firstrot = rotate(cylbase,angle1,dir1);
            auto secondrot = rotate(firstrot,angle2,dir2);
            auto cyl = translate(secondrot,{x,y,z});
            EB2::SphereIF sphA(radius, {xsA,ysA,zsA}, fluid_inside);
            EB2::SphereIF sphB(radius, {xsB,ysB,zsB}, fluid_inside);
            auto scyl = EB2::makeUnion(cyl,sphA,sphB);
            vobjs.push_back(scyl);
            #endif
        }

        #if (AMREX_IS_2D)
        UnionListIF<EB2::UnionIF< EB2::TranslationIF<EB2::RotationIF<EB2::CylinderIF>>, EB2::SphereIF, EB2::SphereIF >> allobjs(vobjs);
        #else
        UnionListIF<EB2::UnionIF< EB2::TranslationIF<EB2::RotationIF<EB2::RotationIF<EB2::CylinderIF>>>, EB2::SphereIF, EB2::SphereIF >> allobjs(vobjs);
        #endif

        // Generate GeometryShop
        auto gshop = EB2::makeShop(allobjs);
        // Build index space
        EB2::Build(gshop, geom.back(), required_coarsening_level, max_coarsening_level);
    }   // if inputfilename == initfibers.rod
    
    fclose(infp);  
}
