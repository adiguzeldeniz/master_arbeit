using PyPlot
using LinearAlgebra
using SpecialFunctions
using NaNMath
using LaTeXStrings
using Measures
using ProgressLogging
using HDF5
using Dates

thisdir = @__DIR__

#naturkonstante
v_c = 299792458

struct HDF5Data
    params::Dict{Symbol, Any}  
    exc_array::Vector{Float64}  
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
        
        exc_array = read(file["exc_array"])
        runtimes = read(file["runtimes"])

        # Load solution data by slice
        Omega = Dict(parse(Int, slice) => read(file["Omega/$slice"]) for slice in keys(file["Omega"]))
        rho_eg = Dict(parse(Int, slice) => read(file["rho_eg/$slice"]) for slice in keys(file["rho_eg"]))
        rho_ee = Dict(parse(Int, slice) => read(file["rho_ee/$slice"]) for slice in keys(file["rho_ee"]))
        rho_gg = Dict(parse(Int, slice) => read(file["rho_gg/$slice"]) for slice in keys(file["rho_gg"]))

        return HDF5Data(params, exc_array,runtimes, Omega, rho_eg, rho_ee, rho_gg)
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



#_______________________________________________PLOTTERS___________________________________________________

function get_slice_index(thickness::Real)::Int
    allowed_thicknesses = [0.1e-6, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
    allowed_indices = [4, 20, 40, 80, 120, 160, 200].*2
    thickness_to_index = Dict(allowed_thicknesses .=> allowed_indices)

    if thickness ∉ keys(thickness_to_index)
        error("Invalid thickness = $thickness. Allowed thicknesses are: $(allowed_thicknesses).")
    end

    return thickness_to_index[thickness]
end

function heatmap_plotter(
    data::HDF5Data,
    thickness::Real;
    quantity::String = "Omega",
    savefig::Union{String, Nothing} = nothing,
    clim::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,  # <-- allow clim=nothing
    cmap::String = "gist_heat",
    time_xlim::Union{Tuple{<:Real, <:Real}, Nothing} = (0, 2),
    exc_ylim::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
    show_colorbar::Bool = true,
    show_annotations::Bool = true,
    return_fig_ax::Bool = false)

    println("\n [plotter:: heatmap plot]")


    # Get correct slice index
    slice_index = get_slice_index(thickness)

    # Extract parameters
    params = data.params
    c = params[:c]
    b = round(params[:b], digits=2)
    t_0 = params[:t_0]
    N = params[:N]
    time_calc = params[:time]
    exc = params[:exc]
    Gamma = params[:Gamma]
    runtime = params[:runtime]

    exc_array = data.exc_array
    time = time_calc .- t_0
    exc_round = round(exc, digits=2)

    # Select quantity
    quantity_data = nothing
    quantity_title = ""

    if quantity == "Omega"
        quantity_data = data.Omega
        quantity_title = "|Omega|^2"
    elseif quantity == "rho_eg"
        quantity_data = data.rho_eg
        quantity_title = "|rho_eg|^2"
    elseif quantity == "rho_ee"
        quantity_data = data.rho_ee
        quantity_title = "rho_ee"
    elseif quantity == "rho_gg"
        quantity_data = data.rho_gg
        quantity_title = "rho_gg"
    else
        error("Unknown quantity: $quantity. Must be one of: \"Omega\", \"rho_eg\", \"rho_ee\", \"rho_gg\".")
    end

    # Get the slice and process depending on quantity
    slice = quantity_data[slice_index]
    intensity_data = (quantity in ["Omega", "rho_eg"]) ? abs2.(slice) : real.(slice)


    # Create figure and axis
    heatmap_fig = figure(figsize=(7, 5))
    heatmap_ax = subplot(111)

    # Choose clim
    if clim === nothing
        if quantity in ["Omega", "rho_eg"]
            clim_used = (1e-12, 1e3)
        elseif quantity in ["rho_ee", "rho_gg"]
            clim_used = (1e-8, 1)
        else
            error("Unknown quantity: $quantity")
        end
    else
        clim_used = clim
    end

    # Always use log scale
    norm = matplotlib.colors.LogNorm(clim_used...)

    # Plot
    pos = heatmap_ax.pcolor(
        time, exc_array, intensity_data;
        norm=norm,
        cmap=cmap
    )



    if show_colorbar
        heatmap_fig.colorbar(pos, ax=heatmap_ax, aspect=30)
    end

    # Labels and limits
    heatmap_ax.set_title(quantity_title)
    heatmap_ax.set_ylabel(latexstring("\\mathrm{Excitation\\;[\\pi]}"))
    heatmap_ax.set_xlabel(latexstring("\\mathrm{Time}\\;t\\;[\\gamma^{-1}]"))

    if time_xlim !== nothing
        heatmap_ax.set_xlim(time_xlim...)
    end
    if exc_ylim !== nothing
        heatmap_ax.set_ylim(exc_ylim...)
    else
        heatmap_ax.set_ylim(minimum(exc_array), maximum(exc_array))
    end

    # Add parameter annotation (thickness updated!)
    if show_annotations
        thickness_um = thickness * 1e6  # meters -> microns
        param_text = "u = $c c\n d = $(thickness_um) μm\n b = $b"
        heatmap_ax.text(0.98, 0.02, param_text;
            transform=heatmap_ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.8))
    end

    println("|plot finished|")

    # Save figure
    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = endswith(savefig, ".png") ? savefig : string(savefig, "_", timestamp, ".png")
        heatmap_fig.savefig(filename; dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end

    # Show and close figure
    display(heatmap_fig)

    if return_fig_ax
        return heatmap_fig, heatmap_ax
    else
        close(heatmap_fig)
    end

    return nothing
end

function runtime_plotter(
    data::HDF5Data;
    savefig::Union{String, Nothing} = nothing,
    show_annotations::Bool = true,
    return_fig_ax::Bool = false)

    println("\n [plotter:: runtime plot]")

    # Helper function to format time nicely
    function format_runtime(seconds::Real)::String
        if seconds < 60
            return "$(round(seconds, digits=2)) s"
        elseif seconds < 3600
            minutes = Int(floor(seconds / 60))
            secs = round(seconds % 60, digits=2)
            return "$(minutes) min $(secs) s"
        else
            hours = Int(floor(seconds / 3600))
            minutes = Int(floor((seconds % 3600) / 60))
            secs = round(seconds % 60, digits=2)
            return "$(hours) h $(minutes) min $(secs) s"
        end
    end

    # Extract data
    exc_array = data.exc_array
    runtimes = data.runtimes  # Already in seconds
    total_runtime = sum(runtimes)

    # Create figure and axis
    fig = figure(figsize=(6, 4))
    ax = subplot(111)

    # Simple line plot
    ax.plot(exc_array, runtimes, label="Computation time")

    # Labels
    ax.set_xlabel(latexstring("\\mathrm{Excitation\\; [\\pi]}"))
    ax.set_ylabel(latexstring("\\mathrm{Runtime\\; [s]}"))

    ax.grid(true)
    ax.set_xlim([minimum(exc_array), maximum(exc_array)])
    ax.legend()

    if show_annotations
        ax.set_title("Runtime vs Excitation")

        # Add nicely formatted total runtime
        runtime_text = "Total: $(format_runtime(total_runtime))"
        ax.text(0.98, 0.02, runtime_text;
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.8))
    end

    # Save figure
    if savefig !== nothing
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = endswith(savefig, ".png") ? savefig : string(savefig, "_", timestamp, ".png")
        fig.savefig(filename; dpi=300, bbox_inches="tight")
        println("|Plot saved as $filename|")
    end

    # Show and close or return
    display(fig)

    if return_fig_ax
        return fig, ax
    else
        close(fig)
    end

    return nothing
end






function main()

    filename = "/Users/denizadiguzel/0.maxplanck/1.anomaly(c)/transitions/transitions_paper_rodas4.h5"

    data = load_hdf5_data(filename)

    runtime_plotter(data)

    heatmap_plotter(
        data,                          
        5e-6;                          # [0.1e-6, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        quantity = "Omega",            # ["Omega", "rho_eg", "rho_ee", "rho_gg"]
        #savefig=joinpath(thisdir, "transitions_paper"),       
        cmap = "gist_heat",  
        clim=(1e-9, 1e3),          
        time_xlim = (0, 2),             
        exc_ylim = (0.98,1.08),             
        show_colorbar = true,           
        show_annotations = true,       
        return_fig_ax = false           
    )

    
    


end

main()