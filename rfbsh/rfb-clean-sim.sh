#!/bin/bash

# Clean up all the stages in this simulation
rm -rf stage??

# Clean up the symlinks for checkpoints, plots and numpy; empty contents without deleting directory
rm -rf chk/* reactchk/*
rm -rf plt/* reactplt/*
rm -rf numpy/*
rm -rf plots/*

# Remove symlinks to last checkpoint, plotfile, numpy
rm -rf chk_last reactchk_last
rm -rf plt_last reactplt_last
rm -rf numpy_last
