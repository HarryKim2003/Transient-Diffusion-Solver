
# 2B URA Project: High-Performance Porous Media Diffusion Solver in Julia

**Author:** Harry Kim  
**Advisor:** Prof. Jeff Gostick, University of Waterloo  
**GPU Support:** ✅ NVIDIA CUDA  (AMD ROCm in progress)   
**Threads:** Multi-threaded CPU fallback  
**Focus:** Scientific computing, numerical simulation, GPU acceleration, physical modeling

## Notes on Running the Program: 

This program uses Threads, so please allow your program to use more threads.

I developed on Windows. To do so, open up cmd, "set JULIA_NUM_THREADS=4", and restart Julia. 
Alternatively, in VScode, open up the Command Pallette (command/ctrl + shift + p),
Open up "Preferences: Open Settings (UI), search julia numThreads, change the value as desired. 

Currently, I'm using 12 threads on my Desktop, and 4 on my Laptop. 

The following are the list of packages used. Be sure to import them before running main.jl.

OrdinaryDiffEq
DifferentialEquations
SparseArrays
LinearAlgebra
Plots
ColorSchemes
Statistics
Random
KrylovKit
Base.Threads
LsqFit
Tortuosity
Tortuosity: tortuosity, vec_to_grid
CUDSS
LinearSolve
DiffEqGPU

Cheers. 
