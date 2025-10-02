using DifferentialEquations
using Plots
using LinearAlgebra
using SpecialFunctions
using NaNMath
using LaTeXStrings
using Measures
using ProgressLogging
using JLD2
using PyPlot
using StaticArrays
using Dates
using Sundials    
using Statistics
using HDF5

# Physical constant
v_c = 299_792_458

function EOM_real!(dt, y, p, t)
    (t_0, u, Sigma, N, Gamma, Delta, eta, Amp, dz) = p

    rho_gg     = @view y[1:N]
    rho_ee     = @view y[N+1:2N]
    rho_eg_re  = @view y[2N+1:3N]
    rho_eg_im  = @view y[3N+1:4N]
    Omega_re   = @view y[4N+1:5N]
    Omega_im   = @view y[5N+1:6N]

    Omega = Omega_re .+ 1im .* Omega_im
    rho_eg =  rho_eg_re .+ 1im .* rho_eg_im

    dt_rho_gg     = @view dt[1:N]
    dt_rho_ee     = @view dt[N+1:2N]
    dt_rho_eg_re  = @view dt[2N+1:3N]
    dt_rho_eg_im  = @view dt[3N+1:4N]
    dt_Omega_re   = @view dt[4N+1:5N]
    dt_Omega_im   = @view dt[5N+1:6N]

    vierte_1   = @SVector[-1/4, -5/6, 3/2, -1/2, 1/12]
    vierte_i   = @SVector[1/12, -2/3, 0, 2/3, -1/12]
    vierte_N_1 = @SVector[-1/12, 1/2, -3/2, 5/6, 1/4]
    vierte_N   = @SVector[1/4, -4/3, 3, -4, 25/12]

    @inbounds for i in 1:N
        imag_part = Omega_im[i] * rho_eg_re[i] - Omega_re[i] * rho_eg_im[i]

        dt_rho_gg[i] = Gamma * rho_ee[i] + imag_part
        dt_rho_ee[i] = -Gamma * rho_ee[i] - imag_part

        tmp = (-Gamma / 2 + 1im * Delta) * rho_eg[i] + (1im / 2) * Omega[i] * (rho_gg[i] - rho_ee[i])
        dt_rho_eg_re[i] = real(tmp)
        dt_rho_eg_im[i] = imag(tmp)
        
    end

    if abs(t - t_0) <= 5 * Sigma
        gauss = -2 * Amp / Sigma^2 * (t - t_0) * exp(-((t - t_0)^2) / Sigma^2)
        dt_Omega_re[1] = gauss
        dt_Omega_im[1] = 0.0
    else
        dt_Omega_re[1] = 0.0
        dt_Omega_im[1] = 0.0
    end

    #------------------Omegas--------------------------
    @inbounds for i in 2:N
        if i==2
            dt_Omega_re[i] = real((-1/(u * dz)) * (vierte_1[1] * Omega[i-1] + vierte_1[2] * Omega[i]+ vierte_1[3] * Omega[i+1]+ vierte_1[4] * Omega[i+2]+ vierte_1[5] * 
            Omega[i+3]) + (1im * eta) / u * rho_eg[i] )
            dt_Omega_im[i] = imag((-1/(u * dz)) * (vierte_1[1] * Omega[i-1] + vierte_1[2] * Omega[i]+ vierte_1[3] * Omega[i+1]+ vierte_1[4] * Omega[i+2]+ vierte_1[5] * 
            Omega[i+3]) + (1im * eta) / u * rho_eg[i] )
        elseif 3 <= i < N-1
            dt_Omega_re[i] = real((-1/(u * dz)) * (vierte_i[1] * Omega[i-2] + vierte_i[2] * Omega[i-1]+ vierte_i[3] * Omega[i]+ vierte_i[4] * Omega[i+1]+ vierte_i[5] * 
            Omega[i+2]) + (1im * eta) / u * rho_eg[i] )
            dt_Omega_im[i] = imag((-1/(u * dz)) * (vierte_i[1] * Omega[i-2] + vierte_i[2] * Omega[i-1]+ vierte_i[3] * Omega[i]+ vierte_i[4] * Omega[i+1]+ vierte_i[5] * 
            Omega[i+2]) + (1im * eta) / u * rho_eg[i] )
        elseif i == N-1
            dt_Omega_re[i] = real((-1/(u * dz)) * (vierte_N_1[1] * Omega[i-3] + vierte_N_1[2] * Omega[i-2]+ vierte_N_1[3] * Omega[i-1]+ vierte_N_1[4] * Omega[i]+ vierte_N_1[5] * 
            Omega[i+1]) + (1im * eta) / u * rho_eg[i] )
            dt_Omega_im[i] = imag((-1/(u * dz)) * (vierte_N_1[1] * Omega[i-3] + vierte_N_1[2] * Omega[i-2]+ vierte_N_1[3] * Omega[i-1]+ vierte_N_1[4] * Omega[i]+ vierte_N_1[5] * 
            Omega[i+1]) + (1im * eta) / u * rho_eg[i] )
        elseif i == N
            dt_Omega_re[i] = real((-1/(u * dz)) * (vierte_N[1] * Omega[i-4] + vierte_N[2] * Omega[i-3]+ vierte_N[3] * Omega[i-2]+ vierte_N[4] * Omega[i-1]+ vierte_N[5] * 
            Omega[i]) + (1im * eta) / u * rho_eg[i] )
            dt_Omega_im[i] = imag((-1/(u * dz)) * (vierte_N[1] * Omega[i-4] + vierte_N[2] * Omega[i-3]+ vierte_N[3] * Omega[i-2]+ vierte_N[4] * Omega[i-1]+ vierte_N[5] * 
            Omega[i]) + (1im * eta) / u * rho_eg[i] )
        end
    end


end

function disect_sol_real(sol, N)
    time = sol.t
    Omega_re = sol[4N+1:5N, :]
    Omega_im = sol[5N+1:6N, :]
    rho_eg_re = sol[2N+1:3N, :]
    rho_eg_im = sol[3N+1:4N, :]
    rho_ee = sol[N+1:2N, :]
    rho_gg = sol[1:N, :]
    return (time, Omega_re .+ 1im .* Omega_im, rho_eg_re .+ 1im .* rho_eg_im, rho_ee, rho_gg)
end

function EOM_double_solver(params)
    # Extract parameters
    t_end = params[:t_end]
    t_0 = params[:t_0]
    dt = params[:dt]
    N = params[:N]
    c = params[:c]
    Sigma = params[:Sigma]
    Gamma = params[:Gamma]
    Delta = params[:Delta]
    d = params[:d]
    exc = params[:exc]
    tjump_AB = params[:tjump_AB]
    var_b = params[:var_b]

    # Calculate constants
    b = calc_b(params)
    params[:b] = b

    dz = d / N
    u = 1 / (c * v_c)
    Amp = (exc * sqrt(pi)) / Sigma

    eta_A = (2 * b) / d
    eta_B = (2 * b * var_b) / d

    # Common function to solve a block
    function solve_block(tspan, dt, p, dt_0)
        prob = ODEProblem(EOM_real!, dt_0, tspan, p)
        sol = solve(prob, Rodas5(), saveat=1e-3, dt=dt)
        return sol
    end

    runtime_s = @elapsed begin
        #______________________|BLOCK|A|______________________
        dt_0A = zeros(6 * N)
        dt_0A[1:N] .= 1.0  # rho_gg init
        pA = (t_0, u, Sigma, N, Gamma, Delta, eta_A, Amp, dz)
        tspanA = (0.0, tjump_AB)

        solA = solve_block(tspanA, dt, pA, dt_0A)

        #______________________|BLOCK|B|______________________
        dt_0B = [solA[i, end] for i in 1:(6 * N)]
        pB = (t_0, u, Sigma, N, Gamma, Delta, eta_B, Amp, dz)
        tspanB = (tjump_AB, t_end)

        solB = solve_block(tspanB, dt, pB, dt_0B)
    end

    # Combine results
    function combine_results(solA, solB, N)
        time = vcat(solA.t, solB.t)
        Omega_re = hcat(solA[4N+1:5N, :], solB[4N+1:5N, :])
        Omega_im = hcat(solA[5N+1:6N, :], solB[5N+1:6N, :])
        rho_eg_re = hcat(solA[2N+1:3N, :], solB[2N+1:3N, :])
        rho_eg_im = hcat(solA[3N+1:4N, :], solB[3N+1:4N, :])
        rho_ee = hcat(solA[N+1:2N, :], solB[N+1:2N, :])
        rho_gg = hcat(solA[1:N, :], solB[1:N, :])

        Omega = Omega_re .+ 1im .* Omega_im
        rho_eg = rho_eg_re .+ 1im .* rho_eg_im

        return (time, Omega, rho_eg, rho_ee, rho_gg)
    end

    time_ges, Omega_ges, rhoeg_ges, rhoee_ges, rhogg_ges = combine_results(solA, solB, N)

    # Save into params
    params[:time] = time_ges
    params[:runtime] = runtime_s

    return Omega_ges[:,1:2100], rhoeg_ges[:,1:2100], rhoee_ges[:,1:2100], rhogg_ges[:,1:2100]
end







function theo_resp(params, time, plotslice)
    Gamma = params[:Gamma]
    Delta = params[:Delta]
    d = params[:d]
    N = params[:N]
    b = params[:b]

    plot_zmax = (plotslice /N) * d

    y = zeros(size(time)) .+ 0im
    
    for (j, t) in enumerate(time)
        if t == 0
            y[j]=0
        else
            y[j] = @. (- sqrt((b / t)) * besselj1(2 * sqrt(b * t)) * exp(- Gamma * t / 2) * exp(1im * Delta * t))
        end
    end
    return y
end

function idx_finder(arr::AbstractArray{T}, value) where T
    idx = argmin(abs.(arr .- value))
    #print(abs.(arr .- value))
    return idx
end

function calc_b(params)
    material_properties = Dict(
        "alpha_Fe" => Dict(
            :rho => 83.13e-27,
            :k_0 => 73.039e-18,
            :f_LM => 0.8,
            :alpha => 8.56,
            :I_e => 3/2,
            :I_g => 1/2,
        )
    )

    Gamma = params[:Gamma]
    d = params[:d]
    type = params[:type]

    mat = material_properties[type]
    rho = mat[:rho]
    k_0 = mat[:k_0]
    f_LM = mat[:f_LM]
    alpha = mat[:alpha]
    I_e = mat[:I_e]
    I_g = mat[:I_g]

    # Calculate b
    b_num = pi * rho * f_LM * (2 * I_e + 1) * Gamma * d 
    b_den = 2 * k_0^2 * (1 + alpha) * (2 * I_g + 1)
    b = b_num / b_den
    return b
end



#__________________________________________________________________________________________________________________



function multiplesolverandsaver_hdf5()
    params = Dict(
        :t_end => 2.1,
        :t_0 => 0.1,
        :dt => 0.0001,
        :c => 1,
        :N => 200,
        :Gamma => 1,
        :Delta => 0,       
        :Sigma => 0.02,   
        :type => "alpha_Fe",
        :d => 5e-6,
        :var_b => 0.4, 
        :tjump_AB =>0.2,
        :exc => 0.01,
        :time => zeros(100),
        :runtime => 0.1,
        :slices => [5, 100, 200],
        :lb_array => zeros(20)
    )

    #________________select slices to save________________
    params[:slices] = [5,100,200]

    # Define sweep array
    par_start, par_end, par_step = 0.1, 2.1, 0.01
    lb_array = collect(par_start:par_step:par_end)
    num_sweeps = length(lb_array)

    # Run one sample solution to determine actual time steps
    sample_sol = EOM_double_solver(params)
    num_time_steps = size(sample_sol[1], 2)  # Extract the correct number of time steps

    # Open HDF5 file
    file_name = "tjumpab_c1.h5"
    h5open(file_name, "w") do file
        # Save the full sweep array
        file["lb_array"] = lb_array

        # Dataset to store runtimes
        file["runtimes"] = zeros(num_sweeps)

        # Create a dedicated group for parameters
        params_group = create_group(file, "params")

        # Save all parameters as datasets (instead of attributes)
        for (key, value) in params
            params_group[string(key)] = value  # Store everything as datasets
        end

        # Create hierarchical structure
        for group_name in ["Omega", "rho_eg", "rho_ee", "rho_gg"]
            g = create_group(file, group_name)
            for slice_index in params[:slices]
                create_dataset(g, string(slice_index), ComplexF64, num_sweeps, num_time_steps)
            end
        end

        for (j, lb_fact) in enumerate(lb_array)
            println("Solving for exc factor: $lb_fact ($j / $num_sweeps)")

            params[:tjump_AB] = lb_fact

            sol = EOM_double_solver(params)

            print("TIME IS: ", size(params[:time]))

            for slice_index in params[:slices]
                file["Omega/$(slice_index)"][j, :] = sol[1][slice_index, :]
                file["rho_eg/$(slice_index)"][j, :] = sol[2][slice_index, :]
                file["rho_ee/$(slice_index)"][j, :] = sol[3][slice_index, :]
                file["rho_gg/$(slice_index)"][j, :] = sol[4][slice_index, :]
                # Save runtime at index j
                file["runtimes"][j] = params[:runtime]
                flush(file)  # Ensure data is committed
            end

        end
    end

    println("All solutions saved to $file_name")
end



function main()
    multiplesolverandsaver_hdf5()
end

main()