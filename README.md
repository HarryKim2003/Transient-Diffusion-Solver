
# 2B URA Project: High-Performance Porous Media Diffusion Solver in Julia

**Author:** Harry Kim  
**Supervisor:** Prof. Jeff Gostick, University of Waterloo  
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




## Features

| Feature               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Analytical + Numerical Hybrid | Curve-fits transient profiles to derive `D_eff` from simulation             |
| GPU Acceleration   | Optional CUDA/ROCm backend using `CuSparseMatrixCSR` or `ROCSPARSEMatrixCSR` |
| Threaded CPU Fallback | Uses multithreaded `KenCarp4()` solver when GPU unavailable                  |
| Sparse Matrix Construction | Custom diffusion stencil on masked domains                                 |
| Visualization      | Integrated heatmaps, transient plots, and `D_eff` profiles                    |
| Modular Architecture | Clean separation into `simulations.jl`, `analysis.jl`, and `utils.jl`            |
| Porosity & Tortuosity | Auto-calculates porosity and tortuosity based on the evolving concentration field |



## Why It Matters

This simulation pipeline supports research in materials such as **fuel cell membranes**, **battery electrodes**, and **nanoporous materials**, where accurate diffusivity prediction is crucial.

Built in **Julia** with a focus on **performance**, **scientific accuracy**, and **software maintainability**, this project reflects the principles expected of real-world production simulations at scale.

Generally, as Julia is designed to be a great choice for scientific research and numeric computation, it was paramount to have a framework that supported transient diffusion calculations in this language. 



## Technical Highlights

- Thread-safe CPU simulation (`KenCarp4`, `KrylovJL_GMRES`)
- Sparse matrix assembly optimized for memory locality
- GPU-compatible masking via broadcast-safe `CuArray` logic
- Automatic recovery of `D_eff` from 2D field using `LsqFit.jl`
- Visualization with `Plots.jl`, `ColorSchemes.jl`, and analytical overlays
- Designed for easy extension to real images (e.g. segmented micro-CT)

---

## Project Structure

- main.jl # Entry point and simulation controller
- simulations.jl # GPU/CPU solvers and matrix assembly
- analysis.jl # Curve fitting, D_eff profiling, tortuosity calc
- utils.jl # Mask generation and visualizations


## Sample Output

- Final concentration heatmaps (with pore masking)
- Virtual probe concentration vs. time plots with analytical fits
- Scatter plots of `D_eff` vs X-position
- Histograms of pore-level `D_eff` distributions

## Skills Demonstrated

| Category            | Tools / Topics |
|---------------------|----------------|
| Performance         | CUDA.jl, SparseArrays, multithreading |
| Scientific Computing| Transient PDEs, ODE solvers, LsqFit |
| Software Engineering| Modular design, performance profiling, error handling |
| Visualization       | Plots.jl, ColorSchemes.jl |
| Research Application| Porous media, D_eff, tortuosity, fuel cell modeling |



## Notes on Running the Program: 

This program uses Threads, so please allow your program to use more threads.

I developed on Windows. To do so, open up cmd, "set JULIA_NUM_THREADS=4", and restart Julia. 
Alternatively, in VScode, open up the Command Pallette (command/ctrl + shift + p),
Open up "Preferences: Open Settings (UI), search julia numThreads, change the value as desired. 

Currently, I'm using 12 threads on my Desktop, and 4 on my Laptop. 

The following are the list of packages used. Be sure to import them before running main.jl.

- OrdinaryDiffEq
- DifferentialEquations
- SparseArrays
- LinearAlgebra
- Plots
- ColorSchemes
- Statistics
- Random
- KrylovKit
- Base.Threads
- LsqFit
- Tortuosity
- Tortuosity: tortuosity, vec_to_grid
- CUDSS
- LinearSolve
- DiffEqGPU

Cheers. 
