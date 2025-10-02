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
using BenchmarkTools

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
        sol = solve(prob, Rodas4(), saveat=1e-4, dt=dt)
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

    return Omega_ges, rhoeg_ges, rhoee_ges, rhogg_ges
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


function double_solver()
    params = Dict(
        :t_end => 2.1,
        :t_0 => 0.1,
        :dt => 0.0001,
        :c => 1,
        :N => 400,
        :Gamma => 1,
        :Delta => 0,       
        :Sigma => 0.01,   
        :type => "alpha_Fe",
        :d => 5e-6,
        :var_b => 1, 
        :tjump_AB =>1.99,
        :exc => 0.8,
        :time => zeros(100),
        :runtime => 0.1
    )
    @time sol = EOM_double_solver(params)


    return sol, params
end # solution


function single_plotter(sol, params ; savefig = nothing)
    Omega, rho_eg, rho_ee, rho_gg = sol

    c = params[:c]
    d = params[:d]
    b = round(params[:b], digits= 2)
    t_0 = params[:t_0]
    N = params[:N]
    Sigma = params[:Sigma]
    type = params[:type]
    time = params[:time]
    runtime = round(params[:runtime], digits=2)
    exc = params[:exc]

    #geplottete Slice
    plot_slice = params[:N]
    plot_zmax = (plot_slice /N) * d

    #Timeshifts, für theo zmax*u shift, für numerik t_0
    timeshift = plot_zmax / (c * v_c)   
    time_num = time .- t_0
    time_theo = time .+ timeshift
    #time_theo = 0:0.01:4

    #Normierung, bei dem Wert norm_x
    norm_x = timeshift + 0.1
    norm_idx = idx_finder(time_num, norm_x )
    norm_idx_theo = idx_finder(time_theo, norm_x )


    exc_round = round(exc, digits=2)




    fig = figure(figsize=(8,6))
    om_ax = fig.add_axes([0.1, 0.25, 0.85, 0.7])
    
    om_ax.plot(time_num, (abs2.(Omega[plot_slice, :])) ./ abs2.(Omega[plot_slice, norm_idx]),
                label=latexstring("| \\Omega |^2 \\; \\text{num}"), color="darkblue")
    
    om_ax.plot(time, (abs2.(theo_resp(params, time, plot_slice))./(abs2.(theo_resp(params, time, plot_slice)[norm_idx_theo]))),
                label=latexstring("| \\Omega |^2 \\; \\text{theo}"), ls="--")
    
    om_ax.set_ylabel("Intensity [a.u.]")
    om_ax.set_xlabel(latexstring("\\mathrm{Time} \\;t \\; \\; [\\gamma^{-1}]"))
    om_ax.set_yscale("log")
    om_ax.set_xlim(0, 2)
    om_ax.set_ylim(1e-12, 100)
    om_ax.legend(loc="upper right")
    om_ax.set_title("Numerical vs Theoretical Omega")
    
    fig.patch.set_facecolor("whitesmoke")
    text_ax = fig.add_axes([0.1, 0.05, 0.85, 0.12])
    
    # Handle d nicely for plot
    if abs(d) < 1e-3  # micrometer range
        d_in_um = d * 1e6
        d_string = string(round(d_in_um; digits=2)) * "\\,\\mu m"
    else
        d_string = string(round(d; sigdigits=3)) * "\\,m"
    end

    # Escape type nicely
    type_latex = "\\mathrm{" * replace(type, "_" => "\\_") * "}"

    # Build full param_text
    param_text = latexstring("\\mathbf{Parameters:}\\quad \\mathbf{u} = ", string(c), "\\,c\\quad|\\quad",
                            "\\mathbf{d} = ", d_string, "\\quad|\\quad",
                            "\\mathbf{b} = ", string(b), "\\quad|\\quad",
                            "\\mathbf{t_{\\mathrm{run}}} = ", string(runtime), "\\,s\\quad|\\quad",
                            "\\mathbf{A} = ", string(exc_round), "\\,\\pi\\quad|\\quad",
                            "\\text{type} = ", type_latex)

    text_ax.text(0.5, 0.5, param_text,
    ha="center", va="center", fontsize=9,
    bbox=Dict("boxstyle"=>"round,pad=0.6", "edgecolor"=>"gray", "facecolor"=>"lavender"))

    text_ax.axis("off")
    

    print("|plot finished|")

    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # Get current time
        filename = string(savefig, "_", timestamp, ".png")  # Append timestamp
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end
    display(fig)
    close(fig) 

end

function main()

    filename = "/Users/denizadiguzel/0.maxplanck/1.anomaly(c)/optimized_EOM/singlecalc"
    sol, params = double_solver()
    single_plotter(sol, params, savefig = nothing)

    matplotlib.pyplot.close()
end

main()
