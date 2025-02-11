#include <rincflo.H>

// Methods for debug printing
#if (DEBUG_RFB)

// *********************************************************************************************************************
void Rincflo::DebugPrintSetup()
{
    // Create directory for debug log files if necessary
    std::filesystem::create_directory("debug");

    // Function to build filename from processor number
    auto fname_func = [](int proc)-> string 
        {return format("debug/proc_{:02d}.log", proc);};

    // Empty out the debug log files
    for (int proc=0; proc < debug_log_size; ++proc)
    {
        // const string fname = format("debug/proc_{:02d}.log", proc);
        const string fname = fname_func(proc);
        auto fs = ofstream(fname, std::ios_base::trunc);
        fs.close();
    }

    // Populate fs_debug with filestream handles to debug log files
    int proc_max = std::min(debug_log_size, ParallelDescriptor::NProcs());
    for (int proc=0; proc < proc_max; ++proc)
    {
        const string fname = fname_func(proc);
        fs_debug.push_back(ofstream(fname, std::ios_base::app));
    }
}

// *********************************************************************************************************************
// Print a debug message to a debug log file corresponding to its MPI rank, e.g. debug_01.txt
void Rincflo::DebugPrint(const string &msg) const
{
    const int proc {MyProc()};
    if (proc < debug_log_size)
    {
        auto& fs = fs_debug[proc];
        string padding = string(4*stack_depth, ' ');
        Print(proc, fs) << padding << msg;
    }
}

// *********************************************************************************************************************
// Print a debug message to a debug log file corresponding to its MPI rank; message is the source file and line number.
void Rincflo::DebugPrintLocation(const char* file, int line) const
{
    const int proc {MyProc()};
    if (proc < debug_log_size)
    {
        auto& fs = fs_debug[proc];
        Print(proc, fs) << format(" ({:s} / line {:d})\n", file, line);
    }
}

// *********************************************************************************************************************
// Print a specialized debug message to indicate entry to a function
void Rincflo::DebugFunctionEntry(const char* function_name) const
{
    const string msg = format("Entering function {:s}", function_name);
    DebugPrint(msg);
    ++stack_depth;
}

// *********************************************************************************************************************
// Print a specialized debug message to indicate exit from a function
void Rincflo::DebugFunctionExit(const char* function_name) const
{
    const string msg = format("Exiting function {:s}", function_name);
    --stack_depth;
    DebugPrint(msg);
}

#endif
