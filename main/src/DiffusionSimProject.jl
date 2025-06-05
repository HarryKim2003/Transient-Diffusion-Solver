# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


using OrdinaryDiffEq
using BenchmarkTools
using DifferentialEquations
using SparseArrays
using LinearAlgebra
using Plots 
using ColorSchemes
using Statistics
using LsqFit
println("hello world")

#Part 1: Building the bulk; open air experiment 
gr();

#Grid Settings
N = 40; #Number of grid points in x + y direction. #increased from 40 for accuracy... 
L = 0.01 # domain length in meters (1cm)
dx = L/N # grid spacing in meters
D = 2.09488e-5 #Bulk diffusivity of oxygen in air (m^2/s)

#Time settings 
tspan = (0.0, 5.0) #simuates 0 to 5 second

# Matrix builder function for the 2D transient diffusion equation.
function build_diffusion_matrix(N, dx, D)

    # Create the sparse matrix A
    N2 = N * N # Total number of grid points
    A = spzeros(Float64, N2, N2); # Initialize a sparse matrix of size N^2 x N^2

    # Fill the sparse matrix A with the finite difference coefficients
    for i in 1:N
        for j in 1:N
            idx = (j - 1) * N + i;

            if i == 1 || i == N #Boundary conditions 
                A[idx, idx] = 1.0;  #sets these as 1.0 to represent fixed boundary conditions
                continue;
            end;

            #initializing boundary diffusion logic
            if i > 1
                A[idx, idx - 1] = 1  # Left
            end
            if i < N
                A[idx, idx + 1] = 1  # Right
            end
            if j > 1
                A[idx, idx - N] = 1  # Down
            end
            if j < N
                A[idx, idx + N] = 1  # Up
            end
            A[idx, idx] = - (i > 1) - (i < N) - (j > 1) - (j < N)
        end;
    end;
    A .*= (D / dx^2); # Scale the matrix by D/dx^2

    u0 = zeros(N2); # Initial condition: all zeros (no concentration)
    for j in 1:N
        for i in 1:N 
            idx = (j - 1) * N + i; #flattened index
            if i == 1 #Left boundary condition
                u0[idx] = 1.0; #Note: CHANGE THIS FOR DIFFERENT BOUNDARY CONDITIONS!! 
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
                #I think Prof Jeff wanted 1:0, 1:1, etc...      
            elseif i == N  #Right boundary condition
                u0[idx] = 0.0; #Note: CHANGE THIS FOR DIFFERENT BOUNDARY CONDITIONS!! 
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
            else
                u0[idx] = 0.0; #initially all zeros.
            end;
        end;
    end;

    return A, u0
end


#Buiding the 2d transient diffusion equation.  
function transient_equation(N, dx, D)
    A, u0 = build_diffusion_matrix(N, dx, D)

    # Create the ODE problem
    # f(du, u, p, t) = mul!(du, A, u); #du = A * u

    function f!(du, u, p, t)
        mul!(du, A, u)
        du[1:N:end] .= 0.0  # Left boundary (fixed to C = 1)
        du[N:N:end] .= 0.0  # Right boundary (fixed to C = 0)
    end


    prob = ODEProblem(f!, u0, tspan);
    solvers = [TRBDF2(), Rosenbrock23(), KenCarp4(), Rodas5()]
    # solver_names = ["TRBDF2", "Rosenbrock23", "KenCarp4", "Rodas5"]
    solver_names = ["KenCarp4"]
    save_times = range(0.0, 5.0, length=300)

    # for (i, solver) in enumerate(solvers)
    #     println("🔧 Trying solver: ", solver_names[i])

    #     @time sol = solve(prob, solver; saveat=save_times, abstol=1e-8, reltol=1e-8)

    #     # Extract D_eff values at x = 0.25L, 0.5L, 0.75L
    #     x_positions = [0.25 * L, 0.5 * L, 0.75 * L]
    #     col_indices = [Int(round(x / dx)) for x in x_positions]
    #     row = div(N, 2)

    #     println("📈 Recovered D_eff values:")
    #     for (j, col) in enumerate(col_indices)
    #         idx = (row - 1) * N + col
    #         sim_concs = [u[idx] for u in sol.u]

    #         maxC = maximum(sim_concs)
    #         idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
    #         idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

    #         p0 = [1e-5]
    #         model(t, p) = [analytical_concentration(ti, p[1], col * dx) for ti in t]
    #         fit = curve_fit(model, save_times[1:idx_stop], sim_concs[1:idx_stop], p0)

    #         println("x = $(x_positions[j]/L)L, D_eff ≈ $(fit.param[1])")
    #     end

    #     println("─────────────────────────────────────")
    # end

    sol = solve(prob, KenCarp4(); saveat=0.05); #Save sample every 0.05 seconds
    # sol = solve(prob, Rodas5(); saveat=range(0, stop=5.0, length=300))

    
    row = div(N, 2);
    col = div(N, 2);
    idxs = [(j - 1) * N + col for j in 1:N]  # all rows at center column

    sim_concs = [mean(u[idxs]) for u in sol.u]
    # sim_concs = (sim_concs .- minimum(sim_concs)) ./ (maximum(sim_concs) - minimum(sim_concs))

    maxC = maximum(sim_concs)
    idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)

    if isnothing(idx_stop)
        idx_stop = length(sim_concs)
    end

    sim_times = sol.t

    # Normalize simulation concentrations to [0, 1]

    # Plot the solution
    println("Simulation complete. Plotting final concentration...") #for testing.(thankyou copilot)
    final_u = sol[end];
    final_C = reshape(final_u, N, N) # Reshape to 2D grid
    final_C_norm = (final_C .- minimum(final_C)) ./ (maximum(final_C) - minimum(final_C)) # Normalize to [0, 1]
 
    #############
    # avg_dC_dx = mean(final_C_norm[:, 2] .- final_C_norm[:, 1]) / dx  # average gradient near inlet
    # J = -avg_dC_dx  # flux at left wall (Fick's Law)
    # delta_C = 1.0   # inlet concentration - outlet concentration
    # D_eff_est = J * L / delta_C

    # println("Estimated D_eff from final profile: ", D_eff_est)
    #############

    reversed_rainbow = reverse(ColorSchemes.rainbow.colors)

    p = heatmap(final_C_norm',
                title="Final Concentration Distribution",
                xlabel="X", ylabel="Y",
                colorbar=true,
                c=reversed_rainbow,
                yflip=true)    
    display(p);
    gui()

    # fit_D_eff(sim_times, sim_concs, idx_stop) #For the middle pore, creates a graph... WIll extend to multi later 
    fit_multiple_virtual_pores(sol, N, dx, L, sim_times)
end

# function analytical_concentration(t, D_eff, x; C_L = 1.0, L=0.01, terms = 100)
#     sum = 0.0;
#     for n in 1:terms
#         sum += (1/n)*sin(n * π * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2);
#     end;
#     return C_L - (2*C_L/ π) * sum;
# end;

function analytical_concentration(t, D_eff, x; C_L = 1.0, L=0.01, terms = 100) #Python was sum_num = 100, but increasing for accuracy...
    sum = 0.0
    for n in 1:terms
        sum += (1/n) * sin(n * π * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2)
    end
    return C_L * (1 - x / L) - (2 * C_L / π) * sum
end


function analytical_steady_state_concentration(x; L=0.01, C_L=1.0, sum_num=100)
    sum_value = 0.0
    for n in 1:sum_num
        sum_value += (1 / n) * sin(n * π * x / L)
    end
    C = C_L - (C_L * x / L) - (2 * C_L / π) * sum_value
    return C
end

# Wrapper for curve fitting
function model_wrapper(p, tvec)
    D_eff = p[1]
    x = 0.5 * L
    return [analytical_concentration(t, D_eff, x) for t in tvec]
end


function fit_D_eff(sim_times, sim_concs, idx_stop)
    # sim_times = vec(sim_times)
    # sim_concs = vec(sim_concs)

    # Initial guess
    p0 = [1e-5]

    # Define model function for LsqFit
    model(t, p) = [analytical_concentration(ti, p[1], 0.5 * L) for ti in t]

    # Perform curve fitting
    fit = curve_fit(model, sim_times[1:idx_stop], sim_concs[1:idx_stop], p0)
    D_eff_fit = fit.param[1]
    println("Recovered D_eff ≈ ", D_eff_fit)

    # Plot
    plot(sim_times, sim_concs, label="Simulation", lw=2)
    plot!(sim_times, model(sim_times, fit.param), label="Fit", lw=2, linestyle=:dash)
    xlabel!("Time [s]")
    ylabel!("Concentration")
    title!("Analytical Fit at Center Point")
end

function fit_multiple_virtual_pores(sol, N, dx, L, sim_times)
    # Select virtual pore positions along the x-axis (normalized locations)
    x_positions = [0.25 * L, 0.5 * L, 0.75 * L]
    col_indices = [Int(round(x / dx)) for x in x_positions]
    row = div(N, 2)  # Middle row

    colors = [:cyan, :green, :blue]              # Different colors for each pore
    markers = [:star, :utriangle, :cross]        # Different marker styles
    labels = ["0.25L", "0.5L", "0.75L"]           # Human-readable labels

    # Create a single figure to plot all curves
    p = plot(title="Transient Diffusion Fit at Virtual Pores",
             xlabel="Time [s]", ylabel="Concentration")

    for (i, col) in enumerate(col_indices)
        idx = (row - 1) * N + col                      # Convert (i,j) to flat index
        sim_concs = [u[idx] for u in sol.u]           # Concentration at this pore over time

        # Clip to 90% rise
        maxC = maximum(sim_concs)
        idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
        idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

        # Fit analytical model to simulation data at this pore
        p0 = [1e-5]
        model(t, p) = [analytical_concentration(ti, p[1], col * dx) for ti in t]
        fit = curve_fit(model, sim_times[1:idx_stop], sim_concs[1:idx_stop], p0)
        println("x = ", labels[i], ", Recovered D_eff ≈ ", fit.param[1])

        # Plot simulation + fit on the same figure
        plot!(p, sim_times, sim_concs,
              label="x = $(labels[i]) concentration", lw=2,
              marker=markers[i], color=colors[i])
        plot!(p, sim_times, model(sim_times, fit.param),
              label="x = $(labels[i]) fitted plot", lw=2,
              linestyle=:dash, color=colors[i])
    end

    display(p)  # Show the combined plot
end




transient_equation(N, dx, D);