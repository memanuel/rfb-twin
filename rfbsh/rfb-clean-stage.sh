#!/bin/bash

# Clean up a directory of rincflo simulation outputs
rm -rf chk???????
rm -rf plt???????
rm -rf reactchk???????
rm -rf reactplt???????

# Clean up old versions if present
rm -rf chk???????.old*
rm -rf plt???????.old*
rm -rf reactchk???????.old*
rm -rf reactplt???????.old*
rm -rf errorplt*

# Clean up numpy directory contents; leave empty directory in place
rm -rf numpy/*

# Clean up plot directory contents; leave empty directory in place
rm -rf plots/*.png

# Clean up symlinks to checkpoints and plotfiles for the last step run
rm -rf chk_last
rm -rf plt_last
rm -rf reactchk_last
rm -rf reactplt_last

# Clean up backtrace error messages
rm -f Backtrace*

# Clean up debug logs
rm -f debug/proc_??.log

# Clean up profiling directory
rm -f profile/*

# Clean up inputs_run files
rm -rf inputs?D_run*.cfg
