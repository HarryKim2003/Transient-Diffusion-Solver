# Tastefully written by Harry Kim
# Date: June 2nd, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 
# Round 2, 1st was a disaster.



# Constants and Domain
C_l = 1.0
L = 0.01           # meters
pore_num = 40
spacing = L / pore_num
D_true = 2.1e-5          # bulk diffusivity

# Time settings
start_time = 0.0
end_time = 5.0
times_to_save = range(start_time, end_time, length=100)


using SparseArrays, DifferentialEquations, LinearAlgebra

N = pore_num
dx = spacing
n_nodes = N * N

# Construct Laplacian A
function build_laplacian(N)
    A = spzeros(n_nodes, n_nodes)
    for j in 1:N, i in 1:N
        idx = (j-1)*N + i
        if i > 1
            A[idx, idx-1] = 1
        end
        if i < N
            A[idx, idx+1] = 1
        end
        if j > 1
            A[idx, idx-N] = 1
        end
        if j < N
            A[idx, idx+N] = 1
        end
        A[idx, idx] = -sum(A[idx, :] .!= 0)
    end
    return A / dx^2
end

A = build_laplacian(N)


u0 = zeros(n_nodes)
left_indices = vec([i for i in 1:N])            # left boundary
right_indices = vec([i for i in 1:N] .+ N*(N-1)) # right boundary

function bc_wrapper(u, t)
    u[left_indices] .= C_l
    u[right_indices] .= 0.0
    return u
end

function dudt!(du, u, p, t)
    du .= D_true * (A * u)
    du = bc_wrapper(du, t)
end

prob = ODEProblem(dudt!, u0, (start_time, end_time))
sol = solve(prob, saveat=times_to_save)

using LsqFit

sum_num = 100
x_pore = [0.25, 0.5, 0.75] .* L  # e.g. for 3 x-locations

function analytical_func(t, D, x)
    sum_term = zeros(length(t))
    for n in 1:sum_num
        coeff = (C_l / n) * sin(n * π * x / L)
        decay = exp.(-D[1] * n^2 * π^2 * t / L^2)
        sum_term .+= coeff .* decay
    end
    return C_l - C_l * x / L .- (2 / π) .* sum_term
end

function xp_to_index(x::Float64; spacing::Float64=L/pore_num, N::Int=pore_num)
    i = round(Int, x / spacing) + 1   # +1 because Julia arrays are 1-based
    j = div(N, 2) + 1                 # take middle row in y-direction
    return (j - 1) * N + i            # column-major flattening
end

# Fit for each x-location
for xp in x_pore
    idx = xp_to_index(xp)
    data = sol[idx, :]  # extract from solution
    fit_idx = findall(t -> t < end_time * 0.9, times_to_save)

    fit_model = (p, t) -> analytical_func(t, p, xp)  # proper signature: (params, xdata)
    p0 = [1e-5]  # initial guess for D_eff


    fit = curve_fit(fit_model, collect(times_to_save[fit_idx])[:], data[fit_idx][:], p0)
    println("Recovered D_eff at x=$(xp): ", fit.param[1])
end


