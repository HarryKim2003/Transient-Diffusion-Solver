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
using KrylovKit

println("hello world")

#Part 1: Building the bulk; open air experiment 
gr();

#Grid Settings
N = 40; #Number of grid points in x + y direction. #increased from 40 for accuracy... 
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

tspan = (0.0, 10.0)

save_times = range(tspan[1], tspan[2], length=300)
function build_pore_network_matrix(pore_num::Int, g::Float64)
    N = pore_num
    N2 = N * N
    A = spzeros(Float64, N2, N2)
    u0 = zeros(N2)

    for j in 1:N
        for i in 1:N
            idx = (j - 1) * N + i

            if i == 1  # Left boundary
                u0[idx] = C_left
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
            elseif i == N  # Right boundary
                u0[idx] = C_right
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
            else  # Internal nodes
                u0[idx] = 0.0
                if i > 1
                    A[idx, idx-1] = g
                end
                if i < N
                    A[idx, idx+1] = g
                end
                if j > 1
                    A[idx, idx-N] = g
                end
                if j < N
                    A[idx, idx+N] = g
                end
                A[idx, idx] = -((i > 1) + (i < N) + (j > 1) + (j < N)) * g
            end
        end
    end

    return A, u0
end

function analytical_concentration(t, D_eff, x)
    C_l = 1.0
    L = 0.01
    sum = 0.0
    for n in 1:100
        sum += (1 / n) * sin(n * π * x / L) * exp(-D_eff * n^2 * π^2 * t / L^2)
    end
    return C_l * (1 - x / L) - (2 * C_l / π) * sum
end

function fit_selected_pores(sol, selected_pores::Vector{Int}, sim_times, dx, L, pore_num)
    for pore in selected_pores
        sim_concs = [u[pore] for u in sol.u]
        maxC = maximum(sim_concs)

        idx_start = findfirst(x -> x > 0.05 * maxC, sim_concs)
        idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
        idx_start = isnothing(idx_start) ? 1 : idx_start
        idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

        p0 = [1e-5]
        i = (pore - 1) % pore_num + 1
        x_pos = (i - 1) * dx


        model(t, p) = [analytical_concentration(ti, p[1], x_pos) for ti in t]


        sim_concs = [u[pore] for u in sol.u]
        println("📈 Concentrations for pore $pore: ", sim_concs)
        println("   Mean concentration: ", mean(sim_concs))
        println("   Max concentration: ", maximum(sim_concs))


        println("Number of timepoints saved: ", length(sol.u))
        println("Time range: ", sol.t[1], " → ", sol.t[end])

        println("📈 Concentrations for pore $pore: ", sim_concs)
        println("   Mean concentration: ", mean(sim_concs))
        println("   Max concentration: ", maximum(sim_concs))
        println("   Index maps to (i, j) = ($(i), $((pore - 1) ÷ pore_num + 1))")


        fit = curve_fit(model, sim_times[idx_start:idx_stop], sim_concs[idx_start:idx_stop], p0)
        println("D_eff for pore $pore : ", fit.param[1])
        println("dx = ", dx)
        println("spacing = ", spacing)
        println("g = ", g)

        plot(sim_times, sim_concs, label="Pore $pore", xlabel="Time", ylabel="Concentration", title="Concentration vs Time")
        plot!(sim_times, model(sim_times, fit.param), label="Fit", linestyle=:dash)
        display(current())

    end
end

function plot_final_concentration(sol, pore_num)
    final_C = reshape(sol[end], pore_num, pore_num)
    heatmap(final_C', title="Concentration Map (at t = end)",
        xlabel="X", ylabel="Y", c=reverse(ColorSchemes.rainbow.colors),
        yflip=true, colorbar=true)
end


function simulate_pore_network(pore_num, L, D)
    spacing = L / pore_num

    # Geometry setup
    pore_diam = spacing * 0.75
    throat_diam = pore_diam * 0.5
    throat_len = spacing - 0.5 * pore_diam - 0.5 * pore_diam
    A_throat = π * (throat_diam / 2)^2
    g = D * A_throat / throat_len
    println("✅ g computed = ", g)


    A, u0 = build_pore_network_matrix(pore_num, g)

    function f!(du, u, p, t)
        for j in 1:pore_num
            left_idx = (j - 1) * pore_num + 1     # i = 1
            right_idx = (j - 1) * pore_num + pore_num  # i = N
            u[left_idx] = C_left
            u[right_idx] = C_right
            du[left_idx] = 0.0
            du[right_idx] = 0.0
        end
        mul!(du, A, u)
    end

    prob = ODEProblem(f!, u0, tspan)
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=save_times)

    println("✅ Simulation complete.")
    println("   Number of timepoints: ", length(sol.u))
    println("   Time range: ", sol.t[1], " to ", sol.t[end])
    println("   Grid size = ", size(sol[end]))

    println("DEBUG: g = ", g)
    println("DEBUG: A[1000,1000] = ", A[1000, 1000])
    return sol, spacing
end

sol2, spacing = simulate_pore_network(pore_num, L, D)
plot_final_concentration(sol2, pore_num)

sim_times = sol2.t


center_row = 20
x_indices = [10, 20, 30]
sample_pores = [(center_row - 1) * 40 + i for i in x_indices]
fit_selected_pores(sol2, [770, 780, 790], sim_times, spacing, L, pore_num)

reversed_rainbow = reverse(ColorSchemes.rainbow.colors)

heatmap(reshape(sol[end], N, N)', title="End Concentration", c=reversed_rainbow, yflip=true)


