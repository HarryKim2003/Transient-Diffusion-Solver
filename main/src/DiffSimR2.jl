# Clean Julia version of Part 1 from OpenPNM Project 2
# Author: Harry Kim | June 2025 | UW 2B URA Project

using DifferentialEquations
using SparseArrays
using LinearAlgebra
using Plots
using Statistics
using LsqFit
using KrylovKit


# -------------------- Parameters --------------------
C_left = 1.0
C_right = 0.0
L = 0.01                       # Domain length (m)
N = 40                         # Grid resolution (NxN)
dx = L / N                     # Grid spacing (m)
D = 2.09488e-5                 # Bulk diffusivity of O2 in air (m^2/s)
tspan = (0.0, 5.0)             # Time span (s)
saveat = range(tspan[1], tspan[2], length=100)

# -------------------- Index Helpers --------------------
function ij_to_idx(i, j, N)
    return (j - 1) * N + i
end

# -------------------- Build Laplacian --------------------
function build_laplacian(N, dx, D)
    size = N * N
    A = spzeros(size, size)
    for j in 1:N
        for i in 1:N
            idx = ij_to_idx(i, j, N)

            if i == 1
                A[idx, idx] = 1.0  # Left Dirichlet
            elseif i == N
                A[idx, idx] = 1.0  # Right Dirichlet
            else
                A[idx, idx] = -4.0
                A[idx, ij_to_idx(i + 1, j, N)] = 1.0
                A[idx, ij_to_idx(i - 1, j, N)] = 1.0
                if j > 1
                    A[idx, ij_to_idx(i, j - 1, N)] = 1.0
                end
                if j < N
                    A[idx, ij_to_idx(i, j + 1, N)] = 1.0
                end
            end
        end
    end
    return D / dx^2 * A
end

# -------------------- ODE Function --------------------
function diffusion_ode!(du, u, p, t)
    du .= p * u
    # Dirichlet BC enforcement
    for j in 1:N
        du[ij_to_idx(1, j, N)] = 0.0  # Constant
        du[ij_to_idx(N, j, N)] = 0.0
    end
end

# -------------------- Initial & Boundary --------------------
u0 = zeros(N * N)
for j in 1:N
    u0[ij_to_idx(1, j, N)] = C_left
    u0[ij_to_idx(N, j, N)] = C_right
end

A = build_laplacian(N, dx, D)
prob = ODEProblem(diffusion_ode!, u0, tspan, A)
sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=saveat)
# -------------------- Postprocess --------------------
function reshape_solution(solvec, N)
    return reshape(solvec, N, N)'
end

final = reshape_solution(sol[end], N)
heatmap(final, title="Concentration Map (t = $(tspan[2])s)", color=:viridis, aspect_ratio=1)

# -------------------- Fit Analytical --------------------
function analytical_func(t, p, x; C_l=C_left, L=L, N_terms=100)
    D_eff = p[1]
    t = isa(t, Number) ? [t] : t
    sum_val = zeros(length(t))
    for n in 1:N_terms
        term = (1 / n) * sin(n * π * x / L) .* exp.(-n^2 * π^2 * D_eff .* t ./ L^2)
        sum_val .+= term
    end
    return C_l .* (1 .- x / L .- (2 / π) .* sum_val)
end

# Pick centerline pores: 0.25L, 0.5L, 0.75L along x, at center y
x_positions = [0.25L, 0.5L, 0.75L]
i_indices = round.(Int, x_positions ./ dx)
j_center = div(N, 2)
pore_indices = ij_to_idx.(i_indices, j_center, N)

for (i, idx) in enumerate(pore_indices)
    data_t = saveat
    data_C = sol[idx, :]
    max_C = maximum(data_C)
    start_idx = findfirst(>(0.01 * max_C), data_C)
    stop_idx = findfirst(>(0.9 * max_C), data_C)
    t_fit = data_t[start_idx:stop_idx]
    C_fit = data_C[start_idx:stop_idx]
    fit = curve_fit((t, p) -> analytical_func(t, p, x_positions[i]), t_fit, C_fit, [D])
    D_eff_est = fit.param[1]
    println("x = $(x_positions[i]), Recovered D_eff ≈ ", D_eff_est)

    plot(data_t, data_C, label="Numerical C(x,t)")
    plot!(data_t, analytical_func.(data_t, Ref(fit.param), x_positions[i]), label="Fit")
    title!("Pore @ x=$(x_positions[i])")
    xlabel!("Time (s)")
    ylabel!("Concentration")
    display(current())
end

