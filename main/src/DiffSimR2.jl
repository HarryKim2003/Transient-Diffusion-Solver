# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


#!!!! IMPORTANT !!!!
# This program uses Threads, so please allow your program to use more threads.
# (Thank you ece 252)

# I developed on Windows. To do so, open up cmd, "set JULIA_NUM_THREADS=4", and restart Julia. 
# Alternatively, in VScode, open up the Command Pallette (command/ctrl + shift + p),
# Open up "Preferences: Open Settings (UI), search julia numThreads, change the value as desired. 

# Currently, I'm using 12 threads on my Desktop, and 4 on my Laptop. 
# There must be better ways to make this faster... 

# Step 1: Add spheres then masks as "dead areas" in order to simulate the porous material. (Working, D_eff profile graph is a little sus...)
# Step 2: Make sure sparse arrays are being used (Good!)


# Step 3: Start using GPU 


# Step 4: Make extract_and_plot_Deff_map NOT o(n^3) time... 

#To do for Thurs:
# Fix Masked concentration 0 being more attractive logic
# Fix D_Eff profile error (Check D_Eff Profile Masking Issue on GPT for notes)

using OrdinaryDiffEq
using BenchmarkTools
using DifferentialEquations
using SparseArrays #Note: Make sure sparse arrays are being used 
using LinearAlgebra
using Plots
using ColorSchemes
using Statistics
using Random
using KrylovKit
using Base.Threads #systems and concurrency ECE 252 instant usage here we go 

using LsqFit #found out on sunday evenign: LsqFit is bad according to reddit 
# https://www.reddit.com/r/Julia/comments/19e1qp3/goodness_of_fit_parameters_using_lsqfitjl/
#maybe use seomething else? 


println("hello world")

##### GLOBAL VARIABLES ##### 

#Part 1: Building the bulk; open air experiment 
gr();

#Grid Settings
N = 40; #Number of grid points in x + y direction. #increased from 40 for accuracy... 
L = 0.01 # domain length in meters (1cm)
dx = L / N # grid spacing in meters
D = 2.09488e-5 #Bulk diffusivity of oxygen in air (m^2/s)

#Time settings 
tspan = (0.0, 5.0) #simuates 0 to 5 second
save_times = range(tspan[1], tspan[2], length=300)

# Boundary conditions. DO NOT CHANGE THESE FOR NOW, CODE WILL BREAK 
C_left = 1.0
C_right = 0.0

##### GLOBAL VARIABLES END ##### 


##### "HELPER" FUNCTIONS SECTIONS #####
function make_analytical_model(x; terms=100)
    return (t, p) -> begin
        D_eff = p[1]
        @inbounds [analytical_concentration(ti, D_eff, x; terms=terms) for ti in t]
    end
end

function fit_Deff(sim_times::AbstractVector, sim_concs::AbstractVector, x::Float64;
    p0=[1e-5], clip_low=0.05, clip_high=0.9, terms=100)
    # Normalize
    sim_concs = (sim_concs .- C_right) ./ (C_left - C_right)
    maxC = maximum(sim_concs)

    # Clip time range
    idx_start = findfirst(c -> c > clip_low * maxC, sim_concs)
    idx_stop = findfirst(c -> c > clip_high * maxC, sim_concs)
    idx_start = isnothing(idx_start) ? 1 : idx_start
    idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

    # Model
    model = (t, p) -> [analytical_concentration(ti, p[1], x; terms=terms) for ti in t]

    # Fit
    fit = curve_fit(model, sim_times[idx_start:idx_stop], sim_concs[idx_start:idx_stop], p0)
    return fit.param[1]  # Return D_eff
end


function analytical_concentration(t, D_eff, x; terms=100)
    sum = 0.0
    for n in 1:terms
        sum = sum + (C_left / n) * sin(n * pi * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2)
    end

    return (C_left - (C_left * (x / L)) - (2 / pi) * sum)
end

# Wrapper for curve fitting
function model_wrapper(p, tvec)
    D_eff = p[1]
    x = 0.5 * L
    return [analytical_concentration(t, D_eff, x) for t in tvec]
end

function generate_mask(N; sphere_radius=5, num_spheres=10)
    mask = ones(Float64, N, N)
    rng = MersenneTwister(1234)  # for reproducibility
    for _ in 1:num_spheres
        x_c = rand(rng, sphere_radius+1:N-sphere_radius)
        y_c = rand(rng, sphere_radius+1:N-sphere_radius)
        for i in 1:N, j in 1:N
            if (i - x_c)^2 + (j - y_c)^2 ≤ sphere_radius^2
                mask[i, j] = 0.0
            end
        end
    end
    return mask
end


##### "HELPER" FUNCTION SECTION END #####


# Matrix builder function for the 2D transient diffusion equation.
function build_diffusion_matrix(N, dx, D, mask)
    N2 = N * N
    A = spzeros(Float64, N2, N2)

    for i in 1:N, j in 1:N
        if mask[i, j] == 0.0
            continue  # skip solid cell
        end

        idx = (j - 1) * N + i
        A[idx, idx] = 0.0  # Will be updated below

        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 ≤ ni ≤ N && 1 ≤ nj ≤ N && mask[ni, nj] == 1.0
                nidx = (nj - 1) * N + ni
                A[idx, nidx] = 1
                A[idx, idx] -= 1
            end
        end
    end

    A .*= (D / dx^2)
    u0 = zeros(N2)
    return A, u0
end

#Buiding the 2d transient diffusion equation.  
function transient_equation(N, dx, D; mask=ones(N, N))
    A, u0 = build_diffusion_matrix(N, dx, D, mask)

    function f!(du, u, p, t)
        u[1:N:end] .= C_left
        u[N:N:end] .= C_right
        mul!(du, A, u)
        du[1:N:end] .= 0.0
        du[N:N:end] .= 0.0

        # zero out diffusion in masked (solid) regions
        for i in 1:N, j in 1:N
            if mask[i, j] == 0.0
                idx = (j - 1) * N + i
                du[idx] = 0.0
            end
        end
    end

    prob = ODEProblem(f!, u0, tspan)
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05)
    sim_times = sol.t

    # Visualization
    println("Simulation complete. Plotting final concentration...") #for testing.(thankyou copilot)

    final_C = reshape(sol[end], N, N)
    final_C_norm = (final_C .- C_right) ./ (C_left - C_right)
    reversed_rainbow = reverse(ColorSchemes.rainbow.colors)

    p = heatmap(final_C_norm',
        title="Masked Concentration",
        yflip=true,
        colorbar=true,
        c=reversed_rainbow)
    display(p)

    fit_multiple_virtual_pores(sol, N, dx, L, sim_times, C_left, C_right)

    return sol, sol.t
end




#Fits pores to model -> Returns fitted D_Eff (Guesses Deffs, see which one fits closest to C) 
function fit_multiple_virtual_pores(sol, N, dx, L, sim_times, C_left, C_right)

    println("🔁 Fitting virtual pores using $(nthreads()) threads...")

    # Select virtual pore positions (you can change these)
    x_positions = [0.25 * L, 0.5 * L, 0.75 * L]
    col_indices = [Int(round(x / dx)) for x in x_positions]

    # Precompute solution tensor
    U_mat = hcat(sol.u...)
    U = reshape(U_mat, N, N, :)  # U[i, j, t]

    # Visual stuff
    colors = [:cyan, :green, :blue]
    markers = [:star, :utriangle, :cross]
    labels = ["0.25L", "0.5L", "0.75L"]

    # Output storage (thread-safe)
    D_eff_results = Vector{Union{Float64,Missing}}(undef, length(col_indices))
    D_eff_results .= missing

    p = plot(title="Transient Diffusion Fit at Virtual Pores",
        xlabel="Time [s]", ylabel="Concentration")

    Threads.@threads for thread_i in 1:length(col_indices)
        col = col_indices[thread_i]
        try
            col_idxs = [(j - 1) * N + col for j in 1:N]  # all rows at column col
            sim_concs = [mean(u[col_idxs]) for u in sol.u]
            D_fit = fit_Deff(sim_times, sim_concs, col * dx)
            D_eff_results[thread_i] = D_fit

            # For visualizing (not thread-safe — defer plotting)
        catch e
            @warn "Fitting failed at col = $col" exception = (e, catch_backtrace())
            D_eff_results[thread_i] = missing
        end
    end

    # === Plotting happens AFTER threading ===
    for (i, col) in enumerate(col_indices)
        col_idxs = [(j - 1) * N + col for j in 1:N]
        sim_concs = [mean(u[col_idxs]) for u in sol.u]
        sim_concs = (sim_concs .- C_right) ./ (C_left - C_right)

        D_fit = D_eff_results[i]
        if D_fit === missing
            continue
        end

        println("x = ", labels[i], ", Recovered D_eff ≈ ", D_fit)

        model = make_analytical_model(col * dx)
        plot!(p, sim_times, sim_concs,
            label="x = $(labels[i]) concentration", lw=2,
            marker=markers[i], color=colors[i])
        plot!(p, sim_times, model(sim_times, [D_fit]),
            label="x = $(labels[i]) fitted plot", lw=2,
            linestyle=:dash, color=colors[i])
    end

    display(p)
end


# Fits C at every (i,i), returning one Deff. 
# Steady state should be VERY close to true D. 
function extract_and_plot_Deff_map(sol, N, dx, L, sim_times)
    println("📊 Fitting all virtual pores using $(nthreads()) threads...")

    # Precompute solution tensor
    U_mat = hcat(sol.u...)           # Stack all solution vectors (N^2 × time)
    U = reshape(U_mat, N, N, :)      # (i, j, t)

    # Thread-safe preallocated output
    d_eff_array = Vector{Union{Float64,Missing}}(undef, (N - 2) * N)
    d_eff_array .= missing

    Threads.@threads for i in 2:N-1
        for j in 1:N
            idx_global = (i - 2) * N + j
            sim_concs = vec(U[i, j, :])

            try
                D_fit = fit_Deff(sim_times, sim_concs, i * dx) #Still o(n^3)... 
                d_eff_array[idx_global] = D_fit
            catch
                d_eff_array[idx_global] = missing
            end
        end
    end

    # === Sequential post-processing ===

    d_eff_profile = Float64[]
    x_arr = Float64[]

    for i in 2:N-1
        start_idx = (i - 2) * N + 1
        stop_idx = start_idx + N - 1
        col_vals = d_eff_array[start_idx:stop_idx]
        mean_D = mean(skipmissing(col_vals))
        push!(d_eff_profile, mean_D)
        push!(x_arr, i * dx)
    end

    # Plot: D_eff Profile vs X
    plot(x_arr, d_eff_profile,
        seriestype=:scatter, label="Mean D_eff per column",
        xlabel="x [m]", ylabel="D_eff", ylims=(2.0e-5, 3.0e-5), title="D_eff Profile vs X")

    # Optional: Histogram
    # histogram(skipmissing(d_eff_array), bins=100,
    #     title="D_eff Distribution", xlabel="D_eff", ylabel="Count", legend=false)
end

# transient_equation(N, dx, D);
mask = generate_mask(N, sphere_radius=3, num_spheres=5)
sol, sim_times = transient_equation(N, dx, D; mask=mask)
extract_and_plot_Deff_map(sol, N, dx, L, sim_times)