// rincflo_utilities.H
// Fragment of Rincflo.H
// All the utility methods in Rincflo
#pragma once
#include <rincflo_macros.H>

private:

void WriteHeader (const std::string& name, bool is_checkpoint) const;
void WriteJobInfo (const std::string& dir) const;

// Write a checkpoint file to disk
void WriteCheckPointFile (CheckPointType type) const;

// Count the number of components to be plotted
int CountPlotComponents();

// Write out the plot file to disk
void WritePlotFile (CheckPointType type, int step);

// Read a checkpoint file from disk
void ReadCheckPointFile ();

// Load the header of a checkpoint file from disk
void LoadCheckPointHeader ();

// Load the fluid flow data from an amrex checkpoint file into existing MultiFabs
void LoadCheckPointFluidData ();

// Load the reaction data from an amrex checkpoint file into existing MultiFabs
void LoadCheckPointReactionData ();

#if (AMREX_USE_EB)
void WriteMyEBSurface ();
#endif

void PrintMaxValues (Real time);
void PrintMaxVal (int lev);
void PrintMinValues (Real time);
void PrintMinVal (int lev);

// Print a row of stars to the console
void PrintStars() const
    {Print() << "********************************************************************************\n";}

// *********************************************************************************************************************
// Methods for debug printing
#if (DEBUG_RFB)
// Initialize for debug printing
void DebugPrintSetup();

// Print a debug message to a debug log file corresponding to its MPI rank, e.g. debug_01.txt
void DebugPrint(const string &msg) const;

// Print a debug message to a debug log file corresponding to its MPI rank; message is the source file and line number.
void DebugPrintLocation(const char* file, int line) const;

// Print a specialized debug message to indicate entry to a function
void DebugFunctionEntry(const char* function_name) const;

// Print a specialized debug message to indicate exit from a function
void DebugFunctionExit(const char* function_name) const;

#endif

// Define macros for debug printing that disappear when not in debug mode.
// (Paranoid and don't want to rely on empty function optimization in g++ with -O3)
// Two flavors of debug print macros
// (1) print just the message
// (2) print the message plus the source and line number indented on the following line
// Also macros to indicate entry and exit of a function.
#if (DEBUG_RFB)

#define DEBUG_PRINT(msg)                DebugPrint(msg);

#define DEBUG_PRINT_LOC()               DebugPrintLocation(__FILE__, __LINE__);

#define DEBUG_FUNC_ENTRY(func_name)     {\
    DebugFunctionEntry(func_name);\
    DEBUG_PRINT_LOC()\
}

#define DEBUG_FUNC_EXIT(func_name)     {\
    DebugFunctionExit(func_name);\
    DEBUG_PRINT_LOC()\
}

#else

#define DEBUG_PRINT(msg)

#define DEBUG_PRINT_LOC()

#define DEBUG_FUNC_ENTRY(func_name)

#define DEBUG_FUNC_EXIT(func_name)

#endif
