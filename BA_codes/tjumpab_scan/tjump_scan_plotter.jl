using PyPlot
using LinearAlgebra
using SpecialFunctions
using NaNMath
using LaTeXStrings
using Measures
using ProgressLogging
using HDF5
using Dates

#naturkonstante
v_c = 299792458

struct HDF5Data
    params::Dict{Symbol, Any}  
    lb_array::Vector{Float64}  
    runtimes::Vector{Float64} 
    Omega::Dict{Int, Matrix{ComplexF64}}  
    rho_eg::Dict{Int, Matrix{ComplexF64}}
    rho_ee::Dict{Int, Matrix{ComplexF64}}
    rho_gg::Dict{Int, Matrix{ComplexF64}}
end

function load_hdf5_data(filename)
    h5open(filename, "r") do file
        # Load parameters as a Dict with Symbol keys
        params = Dict(Symbol(key) => read(file["params"][key]) for key in keys(file["params"]))
        
        lb_array = read(file["lb_array"])
        runtimes = read(file["runtimes"])

        # Load solution data by slice
        Omega = Dict(parse(Int, slice) => read(file["Omega/$slice"]) for slice in keys(file["Omega"]))
        rho_eg = Dict(parse(Int, slice) => read(file["rho_eg/$slice"]) for slice in keys(file["rho_eg"]))
        rho_ee = Dict(parse(Int, slice) => read(file["rho_ee/$slice"]) for slice in keys(file["rho_ee"]))
        rho_gg = Dict(parse(Int, slice) => read(file["rho_gg/$slice"]) for slice in keys(file["rho_gg"]))

        return HDF5Data(params, lb_array,runtimes, Omega, rho_eg, rho_ee, rho_gg)
    end
end

#___________________________________________USEFUL_FUNCTIONS______________________________________________________-

function idx_finder(arr::AbstractArray{T}, value) where T
    idx = argmin(abs.(arr .- value))
    #print(abs.(arr .- value))
    return idx
end
function unwrap!(x, period = 2π)
    y = convert(eltype(x), period)
    v = first(x)
    @inbounds for k = eachindex(x)
        x[k] = v = v + rem(x[k] - v,  y, RoundNearest)
    end
    return x
end



#_______________________________________________PLOTTERS__________________________________________________________-


function heatmap_plotter(data::HDF5Data, slice_index ; savefig = nothing)
    println("\n [plotter:: single heatmap]")

    c = data.params[:c]
    b = data.params[:b]
    d = data.params[:d]
    t_0 = data.params[:t_0]
    N = data.params[:N]
    time = data.params[:time]
    exc = data.params[:exc]
    Gamma = data.params[:Gamma]
    lb_array = data.lb_array
    runtimes = data.runtimes


    #geplottete Slice
    plot_slice = data.params[:N]
    plot_zmax = (plot_slice /N) * d


    #Timeshifts, für theo zmax*u shift, für numerik t_0
    timeshift = plot_zmax / (c * v_c)   
    time_num = time .- t_0
    time_theo = time .+ timeshift

    #Normierung, bei dem Wert norm_x
    norm_x = timeshift + 0.1
    norm_idx = idx_finder(time_num, norm_x )
    norm_idx_theo = idx_finder(time_theo, norm_x )

    #calculate excitation
    exc_round = round(exc, digits=2)

    #___________ PLOTTING THE HEATMAP___________

    # Extract `Omega` data for the selected slice
    omega_slice = data.Omega[slice_index]  # Shape: (num_sweeps, num_time_steps)

    # Convert to intensity values (|Omega|^2)
    intensity_data = abs2.(omega_slice)


    heatmap_fig = figure(figsize=(8, 15))
    heatmap_ax = subplot(111)

    # Create log-scaled heatmap
    pos = heatmap_ax.pcolor(time_num .- timeshift , lb_array, intensity_data, 
                            norm=matplotlib.colors.LogNorm(1e-10, 1), 
                            cmap="gist_heat")

    heatmap_fig.colorbar(pos, ax=heatmap_ax, aspect=50)


    # Labels and limits
    #heatmap_ax.set_ylabel(latexstring("\\mathrm{ tjumpAB \\;factor}"))
    heatmap_ax.set_title("u = $c c, d = $d μm, b= $b, γ = $Gamma, Data: Omega")
    heatmap_ax.set_ylabel(latexstring("\\mathrm{ Excitation \\; [\\pi] }"))
    heatmap_ax.set_xlabel(latexstring("\\mathrm{Time} \\;t \\; \\; [\\gamma^{-1}]"))
    heatmap_ax.set_xlim(0, 7)
    heatmap_ax.set_ylim(minimum(lb_array), maximum(lb_array))
    #heatmap_ax.set_ylim(0.9, 1.20)

    print("|plot finished|")

    # Save if filename is provided
    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # Get current time
        filename = string(savefig, "_", timestamp, ".png")  # Append timestamp
        heatmap_fig.savefig(filename, dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end

    # Show and close figure
    display(heatmap_fig)
    close(heatmap_fig)

end

function runtimes_plotter(data::HDF5Data, slice_index ; savefig = nothing)
    println("\n [plotter:: runtimes]")

    c = data.params[:c]
    b = data.params[:b]
    d = data.params[:d]
    t_0 = data.params[:t_0]
    N = data.params[:N]
    time = data.params[:time]
    exc = data.params[:exc]
    lb_array = data.lb_array
    runtimes = data.runtimes


    #geplottete Slice
    plot_slice = data.params[:N]
    plot_zmax = (plot_slice /N) * d


    #Timeshifts, für theo zmax*u shift, für numerik t_0
    timeshift = plot_zmax / (c * v_c)   
    time_num = time .- t_0
    time_theo = time .+ timeshift

    #Normierung, bei dem Wert norm_x
    norm_x = timeshift + 0.1
    norm_idx = idx_finder(time_num, norm_x )
    norm_idx_theo = idx_finder(time_theo, norm_x )

    #calculate excitation
    exc_round = round(exc, digits=2)

    #___________ PLOTTING THE HEATMAP___________

    # Extract `Omega` data for the selected slice
    omega_slice = data.Omega[slice_index]  # Shape: (num_sweeps, num_time_steps)

    # Convert to intensity values (|Omega|^2)
    intensity_data = abs2.(omega_slice)


    total_runtime = sum(runtimes)

    minutes = floor(Int, total_runtime ÷ 60)
    seconds = round(total_runtime % 60; digits=2)

    println("Total runtime: $minutes min $seconds sec")

    runtimes_fig = figure(figsize=(7, 5))
    runtimes_ax = subplot(111)

    runtimes_ax.plot(lb_array, runtimes)
    #heatmap_ax.set_ylabel(latexstring("\\mathrm{ tjumpAB \\;factor}"))
    runtimes_ax.set_title("Total runtime: $minutes min $seconds sec")
    runtimes_ax.set_xlabel(latexstring("\\mathrm{ Excitation \\; [\\pi] }"))
    runtimes_ax.set_ylabel(latexstring("\\mathrm{Runtime} \\;[s] "))
    runtimes_ax.set_xlim(minimum(lb_array), maximum(lb_array))
    runtimes_ax.set_ylim(0, maximum(runtimes))

    # Labels and limits
    # #heatmap_ax.set_ylabel(latexstring("\\mathrm{ tjumpAB \\;factor}"))
    # heatmap_ax.set_title("u = $c c, d = $d μm")
    # heatmap_ax.set_ylabel(latexstring("\\mathrm{ Excitation \\; [\\pi] }"))
    # heatmap_ax.set_xlabel(latexstring("\\mathrm{Time} \\;t \\; \\; [\\gamma^{-1}]"))
    # heatmap_ax.set_xlim(0, 2)
    # heatmap_ax.set_ylim(0, maximum(exc_array))

    print("|plot finished|")

    # Save if filename is provided
    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # Get current time
        filename = string(savefig, "_", timestamp, ".png")  # Append timestamp
        runtimes_fig.savefig(filename, dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end

    # Show and close figure
    display(runtimes_fig)
    close(runtimes_fig)

end

function heatmap_plotter_small(data::HDF5Data, slice_index ; savefig = nothing)
    println("\n [plotter:: single heatmap]")

    c = data.params[:c]
    b = round(data.params[:b], digits=2)
    d = data.params[:d]
    t_0 = data.params[:t_0]
    N = data.params[:N]
    time_calc = data.params[:time]
    tjump_AB = data.params[:tjump_AB]
    exc = data.params[:exc]
    Gamma = data.params[:Gamma]
    lb_array = data.lb_array
    runtimes = data.runtimes
    runtime = data.params[:runtime]


    #geplottete Slice
    plot_slice = data.params[:N]
    plot_zmax = (plot_slice /N) * d

    time = time_calc .- t_0
    exc_round = round(exc, digits=2)

    #___________ PLOTTING THE HEATMAP___________

    omega_slice = data.Omega[slice_index]  # Shape: (num_sweeps, num_time_steps)
    dataa = abs2.(omega_slice)
    intensity_data = dataa[:,1:2000] 

    heatmap_fig = figure(figsize=(7, 5))
    heatmap_ax = subplot(111)

    # Create log-scaled heatmap
    pos = heatmap_ax.pcolor(time[1:2000] , lb_array, intensity_data, 
                            norm=matplotlib.colors.LogNorm(1e-12, 1e2), 
                            cmap="gist_heat")

    heatmap_fig.colorbar(pos, ax=heatmap_ax, aspect=50)


    # Labels and limits
    #heatmap_ax.set_ylabel(latexstring("\\mathrm{ tjumpAB \\;factor}"))
    heatmap_ax.set_title("d = $d, b= $b,  γ = $Gamma, Data: Omega")
    heatmap_ax.set_ylabel(latexstring("\\mathrm{ tjump_{AB}} \\; \\; [\\gamma^{-1}]"))
    heatmap_ax.set_xlabel(latexstring("\\mathrm{Time} \\;t \\; \\; [\\gamma^{-1}]"))
    heatmap_ax.set_xlim(0, 1.85)
    heatmap_ax.set_ylim(minimum(lb_array), maximum(lb_array))
    #heatmap_ax.set_ylim(0.9, 1.2)
    #heatmap_ax.set_ylim(1.05, 1.1)


    print("|plot finished|")

    # Save if filename is provided
    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # Get current time
        filename = string(savefig, "_", timestamp, ".png")  # Append timestamp
        heatmap_fig.savefig(filename, dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end

    # Show and close figure
    display(heatmap_fig)
    close(heatmap_fig)

end




function main()

    filename = "/home/adiguezel/master_arbeit/BA_codes/tjumpab_scan/tjumpab_c1.h5"
    #runtime_name = "/Users/denizadiguzel/0.maxplanck/1.anomaly(c)/transitions/runtimes1c"
    data = load_hdf5_data(filename)

    #runtimes_plotter(data, 200, savefig = nothing)
    #heatmap_plotter(data, 200, savefig = filename)
    heatmap_plotter_small(data, 200, savefig = filename)

end

main()