
# 2B URA Project: High-Performance Porous Media Diffusion Solver in Julia

**Author:** Harry Kim  
**Advisor:** Prof. Jeff Gostick, University of Waterloo  
**GPU Support:** NVIDIA CUDA  (AMD ROCm in progress)   
**Threads:** Multi-threaded CPU fallback  
**Focus:** Scientific computing, numerical simulation, GPU acceleration, physical modeling



 Overview

This project implements a **parallelized transient diffusion solver** to determine **effective diffusivity (D_eff)** and **tortuosity** of porous materials, using 2D image-based domain masks with dynamically placed obstacles.

The simulation:

- Solves the **transient diffusion PDE** on masked 2D domains using `OrdinaryDiffEq.jl`
- Computes **D_eff** via curve fitting to analytical solutions at virtual probes
- Supports **GPU acceleration** with both NVIDIA CUDA and (soon) AMD ROCm backends
- Enables **column-wise profiling**, porosity analysis, and tortuosity estimation
- Is built for **modularity, scalability**, and high-performance research simulatio




## ⚙️ Features

| Feature               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Analytical + Numerical Hybrid | Curve-fits transient profiles to derive `D_eff` from simulation             |
| GPU Acceleration   | Optional CUDA/ROCm backend using `CuSparseMatrixCSR` or `ROCSPARSEMatrixCSR` |
| Threaded CPU Fallback | Uses multithreaded `KenCarp4()` solver when GPU unavailable                  |
| Sparse Matrix Construction | Custom diffusion stencil on masked domains                                 |
| Visualization      | Integrated heatmaps, transient plots, and `D_eff` profiles                    |
| Modular Architecture | Clean separation into `simulations.jl`, `analysis.jl`, and `utils.jl`            |
| Porosity & Tortuosity | Auto-calculates porosity and tortuosity based on the evolving concentration field |





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
