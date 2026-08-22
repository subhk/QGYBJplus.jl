#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example
================================================================================

MPI-parallel barotropic dipole simulation based on:

    Asselin, O., L. N. Thomas, W. R. Young, and L. Rainville (2020)
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439–3454

Run the published-resolution defaults with:

    mpiexec -n 4 julia --project=. examples/asselin_jpo2020.jl

For a quick check, override the size and number of steps through the
environment:

    QGYBJ_ASSELIN_NX=32 QGYBJ_ASSELIN_NY=32 QGYBJ_ASSELIN_NZ=16 \
    QGYBJ_ASSELIN_STEPS=2 julia --project=. examples/asselin_jpo2020.jl

The model uses the composition-first API and ETD-RK2 exclusively.
================================================================================
=#

using QGYBJplus
using Printf

_environment_int(name, fallback) = parse(Int, get(ENV, name, string(fallback)))
_environment_float(name, fallback) = parse(Float64, get(ENV, name, string(fallback)))
function _environment_optional_int(name)
    value = get(ENV, name, "")
    return isempty(value) ? nothing : parse(Int, value)
end

"""
    asselin_dipole_streamfunction(X, Y, z, amplitude, wavenumber)

Dimensional streamfunction from Asselin et al. (2020), equation (2). The model
grid uses cardinal coordinates `(X, Y)` while the paper's dipole coordinates
are rotated by 45 degrees.
"""
function asselin_dipole_streamfunction(X, Y, z, amplitude, wavenumber)
    x = (X - Y) / sqrt(2)
    y = (X + Y) / sqrt(2)
    return amplitude * sin(wavenumber * x) * cos(wavenumber * y)
end

"""
    run_asselin_example(; kwargs...) -> Simulation

Build, initialize, run, and finalize the Asselin et al. dipole example. The
returned simulation is finalized and can be inspected safely. Production
defaults reproduce the 256×256×128, 15-inertial-period setup; `size`,
`stop_iteration`, and output schedules can be reduced for tests or tutorials.
"""
function run_asselin_example(;
    size=(
        _environment_int("QGYBJ_ASSELIN_NX", 256),
        _environment_int("QGYBJ_ASSELIN_NY", 256),
        _environment_int("QGYBJ_ASSELIN_NZ", 128),
    ),
    extent=(70.0e3, 70.0e3, 3.0e3),
    coriolis_frequency=1.24e-4,
    buoyancy_frequency_squared=1.0e-5,
    inertial_periods=_environment_float("QGYBJ_ASSELIN_INERTIAL_PERIODS", 15.0),
    Δt=_environment_float("QGYBJ_ASSELIN_DT", 2.0),
    stop_iteration=_environment_optional_int("QGYBJ_ASSELIN_STEPS"),
    wave_velocity=0.10,
    surface_layer_depth=30.0,
    flow_velocity=0.335,
    wave_hyperdiffusivity=1.0e5,
    output_dir=get(ENV, "QGYBJ_ASSELIN_OUTPUT", "output_asselin"),
    output_schedule=nothing,
    diagnostics=nothing,
    verbose::Bool=true,
)
    nx, ny, nz = size
    Lx, Ly, Lz = extent
    inertial_period = 2π / coriolis_frequency

    # Paper coordinates are rotated relative to the cardinal model grid.
    dipole_wavenumber = sqrt(2) * π / Lx
    streamfunction_amplitude = flow_velocity / dipole_wavenumber
    vorticity_gradient = 2 * dipole_wavenumber^2 * flow_velocity
    rossby_rms = dipole_wavenumber * flow_velocity / coriolis_frequency
    streamfunction = (X, Y, z) -> asselin_dipole_streamfunction(
        X, Y, z, streamfunction_amplitude, dipole_wavenumber)

    grid = RectilinearGrid(
        size=size,
        x=(-Lx / 2, Lx / 2),
        y=(-Ly / 2, Ly / 2),
        z=(-Lz, 0),
    )
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=coriolis_frequency),
        stratification=ConstantStratification(
            N²=buoyancy_frequency_squared),
        closure=HorizontalHyperdiffusivity(
            flow=0,
            flow2=0,
            waves=wave_hyperdiffusivity,
            waves2=0,
            wave_laplacian_order=2,
        ),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=YBJPlus(),
        parallel_io=false,
        verbose=verbose,
    )

    simulation = nothing
    try
        set!(
            model;
            ψ=streamfunction,
            pv_method=:barotropic,
            waves=SurfaceWave(
                amplitude=wave_velocity,
                scale=surface_layer_depth,
                profile=:gaussian,
            ),
            verbose=verbose,
        )

        resolved_output_schedule = output_schedule === nothing ?
            TimeInterval(5.0 * inertial_period) : output_schedule
        resolved_diagnostics = diagnostics === nothing ?
            IterationInterval(max(1, round(Int,
                0.5 * inertial_period / Δt))) : diagnostics
        stop = stop_iteration === nothing ?
            (; stop_time=inertial_periods * inertial_period) :
            (; stop_iteration=Int(stop_iteration))

        simulation = Simulation(
            model;
            Δt=Δt,
            stop...,
            output=NetCDFOutput(
                path=output_dir,
                schedule=resolved_output_schedule,
                fields=(:ψ, :waves),
            ),
            diagnostics=resolved_diagnostics,
            verbose=verbose,
        )

        if is_root(simulation) && verbose
            println("="^70)
            println("Asselin et al. (2020) Dipole")
            println("="^70)
            @printf("Resolution: %d × %d × %d\n", nx, ny, nz)
            if stop_iteration === nothing
                @printf("Duration: %.1f inertial periods\n", inertial_periods)
            else
                @printf("Duration: %d ETD-RK2 steps\n", stop_iteration)
            end
            @printf("Domain: %.1f km × %.1f km × %.1f km\n",
                    Lx / 1e3, Ly / 1e3, Lz / 1e3)
            @printf("Timestepper: ETD-RK2, Δt = %.1f s\n", Δt)
            @printf("Dipole checks: γ = %.3e m⁻¹ s⁻¹, κU/f = %.3f\n",
                    vorticity_gradient, rossby_rms)
            println("Output directory: $output_dir")
        end

        run!(simulation)
    finally
        simulation === nothing ? finalize_model!(model) :
                                 finalize_simulation!(simulation)
    end
    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_asselin_example()
end
