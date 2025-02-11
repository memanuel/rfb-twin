#include <rincflo.H>

// Alias the file system
namespace fs = std::filesystem;

void writeBuildInfo();

int main(int argc, char* argv[])
{
    // check to see if it contains --describe
    if (argc >= 2) 
    {
        for (auto i = 1; i < argc; i++) 
        {
            if (std::string(argv[i]) == "--describe") 
            {
                writeBuildInfo();
                return 0;
            }
        }
    }

    amrex::Initialize(argc, argv);
    { 
        // These braces are necessary to ensure amrex::Finalize() can be called without explicitly
        // deleting all the Rincflo member MultiFabs
        BL_PROFILE("main()");

        // Issue an error if input file is not given 
        if(argc < 2) 
            {Abort("Input file must be given as command-line argument.");}

        // Write out the Rincflo git hash (the AMReX git hash is already written)
        const char* githash_rincflo = buildInfoGetGitHash(1);
        Print() << format("Rincflo git hash: {}\n", githash_rincflo);
        // Write out the hostname
        char hostname[HOST_NAME_MAX];
        int result = gethostname(hostname, HOST_NAME_MAX);
        if (!result)
            {Print() << format("Hostname: {:s}\n", hostname);}
        // Write out the time as an ISO string
        time_t now = time(0);
        std::tm* localTime = std::localtime(&now);
        std::stringstream ss;
        ss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S"); 
        Print() << format("Time: {:s}\n", ss.str());

        // Start timing the program
        Real start_time = second();
       
        // Create directory for profiling files if missing
        fs::create_directory("profile");

        // Default constructor. Note inheritance: Rincflo : AmrCore : AmrMesh.
        Rincflo rf;

        // Initialize data, parameters, arrays and derived internals
        rf.InitData();

        // Summarize the geometry and grid structure
        rf.AnalyzeGeometry();

        // Load simulation data from numpy checkpoint if requested
        if (rf.do_ReadNumpy())
            {rf.ReadNumpy();}

        // Time when initialization ends
        Real init_end_time = second();

        // Evolve system to final time
        if(rf.do_flow()) 
            {rf.Evolve();}

        // Time when evolve (fluid flow) ends
        Real evolve_end_time = second();
        
        // Electrochemistry reaction
        if(rf.do_reaction()) 
            {rf.React();}

        // Time when reaction ends
        Real react_end_time = second();

        // Take max over the MPI workers    
        ReduceRealMax(start_time, IOProcessorNumber());
        ReduceRealMax(init_end_time, IOProcessorNumber());
        ReduceRealMax(evolve_end_time, IOProcessorNumber());
        ReduceRealMax(react_end_time, IOProcessorNumber());

        // Calculate time for initialization, fluid flow (evolve) and reaction
        Real init_time = init_end_time - start_time;
        Real evolve_time = evolve_end_time - init_end_time;
        Real react_time = react_end_time - evolve_end_time;

        // Print timing results for main simulation block
        Print() << "********************************************************************************\n";
        Print() << format("Time spent in InitData():    {:10.2f} sec\n", init_time);
        Print() << format("Time spent in Evolve():      {:10.2f} sec\n", evolve_time);
        Print() << format("Time spent in React():       {:10.2f} sec\n", react_time);

        // Write out numpy arrays if requested
        if(rf.do_WriteNumpy())
        {
            rf.WriteNumpyAll();
            Real numpy_time = second() - react_end_time;
            Print() << format("Time spent in WriteNumpyAll():  {:10.2f} sec\n", numpy_time);
        }

        // Perform the analysis to the console if requested
        if(rf.do_Analysis()) 
            {rf.Analysis();}

    } // End braces to separate scope from amrex::Finalize
    Print() << "********************************************************************************\n";
    amrex::Finalize();
}
