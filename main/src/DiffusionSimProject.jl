# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


# Step 1: Add spheres then masks as "dead areas" in order to simulate the porous material.
# Step 2: Make sure sparse arrays are being used 
# Step 3: STart using GPU 

using OrdinaryDiffEq
using BenchmarkTools
using DifferentialEquations
using SparseArrays #Note: Make sure sparse arrays are being used 
using LinearAlgebra
using Plots
using ColorSchemes
using Statistics
using LsqFit #found out on sunday evenign: LsqFit is bad according to reddit 
# https://www.reddit.com/r/Julia/comments/19e1qp3/goodness_of_fit_parameters_using_lsqfitjl/
#maybe use seomething else? 


using KrylovKit

println("hello world")


#Part 1: Building the bulk; open air experiment 
gr();

#Grid Settings
N = 100; #Number of grid points in x + y direction. #increased from 40 for accuracy... 
L = 0.01 # domain length in meters (1cm)
dx = L / N # grid spacing in meters
D = 2.09488e-5 #Bulk diffusivity of oxygen in air (m^2/s)

#For part 2 
# pore_diam = spacing * 0.75
# throat_diam = pore_diam * 0.5
# throat_len = spacing - pore_diam / 2 - pore_diam / 2
# A_throat = π * (throat_diam / 2)^2
# g = D * A_throat / throat_len  # [m³/s]

#BOUNDARY CONDITIONS: NOTE: CHANGE THESE FOR DIFFERENT BOUNDARY CONDITIONS!!
#For now, the code is set to have a left boundary condition of 1.0 and a right boundary condition of 0.0
#Might want to optimize for differnet options in the future.

C_left = 1.0
C_right = 0.0

#Time settings 
tspan = (0.0, 5.0) #simuates 0 to 5 second
save_times = range(tspan[1], tspan[2], length=300)

# Matrix builder function for the 2D transient diffusion equation.
function build_diffusion_matrix(N, dx, D)

    # Create the sparse matrix A
    N2 = N * N # Total number of grid points
    A = spzeros(Float64, N2, N2) # Initialize a sparse matrix of size N^2 x N^2

    # Fill the sparse matrix A with the finite difference coefficients
    for i in 1:N
        for j in 1:N
            idx = (j - 1) * N + i

            if i == 1 || i == N #Boundary conditions 
                A[idx, idx] = 1.0  #sets these as 1.0 to represent fixed boundary conditions
                continue
            end

            #initializing boundary diffusion logic
            if i > 1
                A[idx, idx-1] = 1  # Left
            end
            if i < N
                A[idx, idx+1] = 1  # Right
            end
            if j > 1
                A[idx, idx-N] = 1  # Down
            end
            if j < N
                A[idx, idx+N] = 1  # Up
            end
            A[idx, idx] = -(i > 1) - (i < N) - (j > 1) - (j < N)
        end
    end
    A .*= (D / dx^2) # Scale the matrix by D/dx^2

    u0 = zeros(N2) # Initial condition: all zeros (no concentration)

    return A, u0
end

#Buiding the 2d transient diffusion equation.  
function transient_equation(N, dx, D)
    A, u0 = build_diffusion_matrix(N, dx, D)

    # Create the ODE problem

    function f!(du, u, p, t)
        # Enforce boundary values *before* applying A
        u[1:N:end] .= C_left
        u[N:N:end] .= C_right

        # Then apply the diffusion operator
        mul!(du, A, u)  # Compute du = A * u (diffusion) (doesn't allocate new memory apparently :o)

        # After computing du = A*u, fix the BCs
        du[1:N:end] .= 0.0
        du[N:N:end] .= 0.0
    end

    prob = ODEProblem(f!, u0, tspan)
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05)

    col = div(N, 2)
    idxs = [(j - 1) * N + col for j in 1:N]  # all rows at center column
    sim_concs = [mean(u[idxs]) for u in sol.u] #Basically, average conc at each column.
    #Probably unneccessary since the heat map final is uniform on the cols...v 


    maxC = maximum(sim_concs)
    idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
    idx_stop = isnothing(idx_stop) ? length(sim_concs) : max(idx_stop, 20) #Stop at 90%.

    if isnothing(idx_stop)
        idx_stop = length(sim_concs)
    end

    sim_times = sol.t


    # Plot the solution
    println("Simulation complete. Plotting final concentration...") #for testing.(thankyou copilot)
    final_u = sol[end]
    final_C = reshape(final_u, N, N) # Reshape to 2D grid
    final_C_norm = (final_C .- C_right) ./ (C_left - C_right) #Normalize 
    reversed_rainbow = reverse(ColorSchemes.rainbow.colors)

    p = heatmap(final_C_norm',
        title="Final Concentration Distribution",
        xlabel="X", ylabel="Y",
        colorbar=true,
        c=reversed_rainbow,
        yflip=true)
    display(p)
    gui()

    fit_multiple_virtual_pores(sol, N, dx, L, sim_times)
    return sol, sim_times
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



function fit_multiple_virtual_pores(sol, N, dx, L, sim_times)
    # Select virtual pore positions along the x-axis (normalized locations)
    x_positions = [0.25 * L, 0.5 * L, 0.75 * L] #NOTE: change this if you want to 
    col_indices = [Int(round(x / dx)) for x in x_positions]

    colors = [:cyan, :green, :blue]              # Different colors for each pore
    markers = [:star, :utriangle, :cross]        # Different marker styles
    labels = ["0.25L", "0.5L", "0.75L"]           # Human-readable labels

    # Create a single figure to plot all curves
    p = plot(title="Transient Diffusion Fit at Virtual Pores",
        xlabel="Time [s]", ylabel="Concentration")

    for (i, col) in enumerate(col_indices)
        col_idxs = [(j - 1) * N + col for j in 1:N]  # all rows at column col
        sim_concs = [mean(u[col_idxs]) for u in sol.u] #again, not sure if this is needed. Avg concentration for column at each X

        # Normalize
        sim_concs = (sim_concs .- C_right) ./ (C_left - C_right)


        # Clip to 90% rise
        maxC = maximum(sim_concs)
        idx_start = findfirst(x -> x > 0.05 * maxC, sim_concs)
        idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
        idx_start = isnothing(idx_start) ? 1 : idx_start
        idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

        # Fit analytical model to simulation data at this pore
        p0 = [1e-5]
        model(t, p) = [analytical_concentration(ti, p[1], col * dx) for ti in t]
        fit = curve_fit(model, sim_times[idx_start:idx_stop], sim_concs[idx_start:idx_stop], p0)

        #NOTE: Apparently, curve_fit and other lsqfit methods in general are really bad.
        # The logistic decay instead of growth (expected (?)) is almsot certainly due to 
        # fitting methods / numerical overshooting 

        #Possible problem: Fitting is using 40 terms, conc is using 39 


        println("x = ", labels[i], ", Recovered D_eff ≈ ", fit.param[1])

        # Plot simulation + fit on the same figure
        plot!(p, sim_times, sim_concs,
            label="x = $(labels[i]) concentration", lw=2,
            marker=markers[i], color=colors[i])
        plot!(p, sim_times, model(sim_times, fit.param),
            label="x = $(labels[i]) fitted plot", lw=2,
            linestyle=:dash, color=colors[i])
    end

    display(p)
end

function extract_and_plot_Deff_map(sol, N, dx, L, sim_times)
    println("📊 Fitting all virtual pores...")

    d_eff_array = Float64[]
    d_eff_profile = Float64[]
    x_arr = Float64[]
    d_eff_buffer = Float64[]
    pore_ctr = 0


    for i in 2:N-1  # x-direction (skip boundaries)
        for j in 1:N  # y-direction (all rows)
            idx = (j - 1) * N + i
            sim_concs = [u[idx] for u in sol.u]

            # Fit only until 90% rise
            maxC = maximum(sim_concs)
            idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
            idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

            p0 = [1e-5]
            model(t, p) = [analytical_concentration(ti, p[1], i * dx) for ti in t]

            try
                fit = curve_fit(model, sim_times[1:idx_stop], sim_concs[1:idx_stop], p0)
                D_fit = fit.param[1]
                push!(d_eff_array, D_fit)
                push!(d_eff_buffer, D_fit)
            catch
                push!(d_eff_array, NaN)
                push!(d_eff_buffer, NaN)
            end

            pore_ctr += 1
            if pore_ctr % N == 9
                push!(d_eff_profile, mean(skipmissing(d_eff_buffer)))
                push!(x_arr, i * dx)
                empty!(d_eff_buffer)
            end
        end
    end


    # D_eff Profile vs X
    plot(x_arr, d_eff_profile,
        seriestype=:scatter, label="Mean D_eff per column",
        xlabel="x [m]", ylabel="D_eff", ylims=(2.0e-5, 3.0e-5), title="D_eff Profile vs X")
    # Histogram
    # d_vals = filter(!isnan, d_eff_array)
    # histogram(d_vals, bins=100, title="D_eff Distribution",
    #     xlabel="D_eff", ylabel="Count", label="", legend=false, xlims=(2.0e-5, 3.0e-5))
end

# transient_equation(N, dx, D);
sol, sim_times = transient_equation(N, dx, D);
extract_and_plot_Deff_map(sol, N, dx, L, sim_times)



