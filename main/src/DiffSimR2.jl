# Tastefully written by Harry Kim
# Date: June 3rd, 2025
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
using SparseArrays
using Images
using FileIO

img = load("2bURA/main/data/porous_slice.png")
if ndims(img) == 3  # color
    img = channelview(img)[1, :, :]
end
mask = Float64.(img .> 0.5)  # 1.0 = pore, 0.0 = solid

Nx, Ny = size(mask)
L = 0.01
D = 2.09488e-5

dx = L / Nx

function analytical_concentration(t, D_eff, x; C_L=1.0, L=0.01, terms=100)
    sum = 0.0
    for n in 1:terms
        sum += (1 / n) * sin(n * π * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2)
    end
    return C_L * (1 - x / L) - (2 * C_L / π) * sum
end

function fit_selected_virtual_pores(sol, mask, dx, L)
    Nx, Ny = size(mask)
    sample_pores = [(20, 10), (20, 20), (20, 30)]  # y, x (row, col) in image indexing
    colors = [:cyan, :green, :blue]
    markers = [:star5, :utriangle, :cross]

    sim_times = sol.t
    p = plot(title="Fitted D_eff at Selected Pores", xlabel="Time", ylabel="Concentration", legend=:outertopright)

    for (idx, (j, i)) in enumerate(sample_pores)
        flat_idx = (j - 1) * Nx + i
        sim_concs = [u[flat_idx] for u in sol.u]
        maxC = maximum(sim_concs)
        idx_stop = findfirst(x -> x > 0.9 * maxC, sim_concs)
        idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop

        p0 = [1e-5]
        model(t, p) = [analytical_concentration(ti, p[1], i * dx) for ti in t]
        fit = curve_fit(model, sim_times[1:idx_stop], sim_concs[1:idx_stop], p0)
        D_eff = fit.param[1]

        println("D_eff for pore ($(j), $(i)) ≈ ", D_eff)

        plot!(p, sim_times, sim_concs, label="Pore $(j),$(i) concentration", lw=2, marker=markers[idx], color=colors[idx])
        plot!(p, sim_times, model(sim_times, fit.param), label="Pore $(j),$(i) fitted", lw=2, linestyle=:dash, color=colors[idx])
    end

    display(p)
end

function build_diffusion_matrix(mask::Array{Float64,2}, dx, D)
    Nx, Ny = size(mask)
    N2 = Nx * Ny
    A = spzeros(Float64, N2, N2)
    u0 = zeros(N2)

    for i in 1:Nx, j in 1:Ny
        idx = (j - 1) * Nx + i

        if mask[i, j] == 0.0
            A[idx, idx] = 1.0
            continue
        end

        if i > 1 && mask[i-1, j] == 1.0
            A[idx, idx-1] = 1.0
        end
        if i < Nx && mask[i+1, j] == 1.0
            A[idx, idx+1] = 1.0
        end
        if j > 1 && mask[i, j-1] == 1.0
            A[idx, idx-Nx] = 1.0
        end
        if j < Ny && mask[i, j+1] == 1.0
            A[idx, idx+Nx] = 1.0
        end

        A[idx, idx] = -sum([
            i > 1 && mask[i-1, j] == 1.0,
            i < Nx && mask[i+1, j] == 1.0,
            j > 1 && mask[i, j-1] == 1.0,
            j < Ny && mask[i, j+1] == 1.0
        ])
    end

    A .*= (D / dx^2)

    for i in 1:Nx, j in 1:Ny
        idx = (j - 1) * Nx + i
        if mask[i, j] == 1.0
            if i == 1
                u0[idx] = 1.0
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
            elseif i == Nx
                u0[idx] = 0.0
                A[idx, :] .= 0.0
                A[idx, idx] = 1.0
            end
        end
    end

    return A, u0
end

function simulate_diffusion_from_image(mask::Array{Float64,2}, dx, D; tspan=(0.0, 5.0))
    A, u0 = build_diffusion_matrix(mask, dx, D)

    function f!(du, u, p, t)
        mul!(du, A, u)
    end

    prob = ODEProblem(f!, u0, tspan)
    sol = solve(prob, TRBDF2(linsolve=KrylovJL_GMRES()); saveat=0.05)

    final_C = reshape(sol[end], size(mask)...)
    heatmap(final_C', title="Final Concentration (Image-Based)", yflip=true, c=:viridis)

    return sol
end

sol = simulate_diffusion_from_image(mask, dx, D)
fit_selected_virtual_pores(sol, mask, dx, L)
