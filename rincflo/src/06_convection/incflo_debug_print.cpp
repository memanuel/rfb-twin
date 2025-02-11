#include <Convection.H>

// Methods for debug printing
#if (DEBUG_RFB)
// Print a debug message to a debug log file corresponding to its MPI rank, e.g. debug_01.txt
void convection::DebugPrint(const string &msg)
{
    const int proc {MyProc()};
    if (proc < debug_log_size)
    {
        const string fname = format("debug/proc_{:02d}.log", proc);        
        auto fs = std::fstream(fname, std::ios_base::app);
        Print(proc, fs) << msg;
        fs.close();
    }
}
#endif
