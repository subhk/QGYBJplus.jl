#=
Architecture tests. The GPU code path is exercised without a GPU by running on
JLArrays, the reference GPU array implementation that executes
KernelAbstractions kernels on the host and rejects scalar indexing. A model on
`GPU(JLBackend())` therefore takes the device code path — every loop a
kernel, every host transfer explicit — and must reproduce the CPU model.
JLArrays kernels would also accept host arrays, which a real GPU rejects, so
every launch below is checked for host-array arguments. Device FFT libraries
(cuFFT) are not exercised.

JLArrays has no FFT library of its own, so the tests supply horizontal FFT
plans that round-trip through FFTW; that is the documented extension point
for backends without AbstractFFTs plans.
=#

using Test
using QGYBJplus
using JLArrays
using FFTW
using NCDatasets

JLArrays.GPUArrays.allowscalar(false)

struct HostRoundTripPlan{P, B}
    plan::P
    buffer::B
end

function Base.:*(plan::HostRoundTripPlan, x::JLArray)
    copyto!(plan.buffer, x)
    plan.plan * plan.buffer
    copyto!(x, plan.buffer)
    return x
end

function QGYBJplus.Transforms.horizontal_fft_plans(buffer::JLArray{<:Complex, 3})
    host_buffer = Array(buffer)
    return (HostRoundTripPlan(FFTW.plan_fft!(host_buffer, (2, 3)), host_buffer),
            HostRoundTripPlan(FFTW.plan_ifft!(host_buffer, (2, 3)), host_buffer))
end

const EMULATED_GPU = GPU(JLBackend())

# JLArrays kernels can read host memory but a GPU cannot, so record every
# emulated launch that is handed a host array, directly or inside a struct or
# closure.
const EMULATED_LAUNCHES = Ref(0)
const HOST_ARRAY_KERNELS = Set{Symbol}()

holds_host_array(x, depth=0) = _holds_host_array(x, depth)
_holds_host_array(::Union{Array, BitArray}, depth) = true
_holds_host_array(x::AbstractArray, depth) =
    (p = parent(x); p !== x && holds_host_array(p, depth + 1))
_holds_host_array(::Union{Type, Module, Symbol, String, Number, Nothing}, depth) = false
function _holds_host_array(x, depth)
    depth > 8 && return false
    return any(name -> holds_host_array(getfield(x, name), depth + 1),
               fieldnames(typeof(x)))
end

function QGYBJplus.launch!(kernel!::F, backend::JLBackend, ndrange::Tuple,
                           args::Vararg{Any, N}) where {F, N}
    EMULATED_LAUNCHES[] += 1
    any(holds_host_array, args) && push!(HOST_ARRAY_KERNELS, nameof(kernel!))
    any(iszero, ndrange) && return nothing
    kernel!(backend)(args...; ndrange=ndrange)
    return nothing
end

function architecture_test_model(architecture; formulation=YBJPlus(),
                                 feedback=WaveMeanFeedback(), flow=EvolvingFlow(),
                                 size=(16, 12, 6), precision=Float64)
    grid = RectilinearGrid(architecture, precision; size, extent=(2π, 2π, 1.0))
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.3),
        stratification=AnalyticalProfile(z -> 1.0 + 0.5cos(3z); returns=:N²),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity((1e-3, 1e-7), (4, 8)),
            wave=WaveHyperdiffusivity((2e-3, 1e-7), (4, 8))),
        vertical_diffusion=VerticalDiffusivity(1e-3),
        flow=flow, feedback=feedback, formulation=formulation,
        verbose=false,
    )
end

architecture_test_psi(x, y, z) =
    0.8sin(x) * (1 + 0.4z) + 0.35sin(x + y) * (1 + 0.3z^2) + 0.2cos(2x - y)

function architecture_test_wave(grid)
    return ComplexF64[
        (0.10 + 0.02k) * (sin(grid.x[i]) + 0.7im * cos(grid.y[j]) +
                          0.3sin(grid.x[i] + 2grid.y[j]))
        for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]]
end

field_names = (:q, :B, :psi, :A, :C, :u, :v, :w)

function fields_match(device_model, host_model; rtol=1e-13)
    return all(field_names) do name
        device = Array(getfield(device_model.fields, name))
        host = getfield(host_model.fields, name)
        isapprox(device, host; rtol, atol=rtol * max(maximum(abs, host), eps()))
    end
end

@testset "Architectures" begin
    @test CPU() isa AbstractArchitecture
    @test EMULATED_GPU isa GPU
    # Without a GPU package loaded, GPU() explains what to do.
    if !applicable(QGYBJplus.default_gpu_backend)
        @test_throws ArgumentError GPU()
    end

    host_vector = collect(1.0:4.0)
    device_vector = on_architecture(EMULATED_GPU, host_vector)
    @test device_vector isa JLArray
    @test on_architecture(CPU(), device_vector) == host_vector
    @test on_architecture(CPU(), host_vector) === host_vector
    @test architecture(device_vector) isa GPU
    @test architecture(host_vector) == CPU()

    grid = RectilinearGrid(EMULATED_GPU; size=(8, 8, 4), extent=(2π, 2π, 1.0))
    @test architecture(grid) === EMULATED_GPU
    # Grid coordinates stay on the host for setup and output.
    @test grid.kx isa Vector{Float64}
    fields = allocate_fields(grid)
    @test fields.q isa JLArray{ComplexF64, 3}
    @test fields.u isa JLArray{Float64, 3}
    @test architecture(fields) == architecture(grid)

    # FFTW accepts any dense array but cannot read device memory, so a device
    # array without its own FFT library gets an explanation, not a crash.
    @test_throws ArgumentError QGYBJplus.Transforms._check_fft_support(
        JLArray(zeros(ComplexF64, 2, 4, 4)))
    @test QGYBJplus.Transforms._check_fft_support(zeros(ComplexF64, 2, 4, 4)) === nothing
end

@testset "Emulated GPU model reproduces the CPU model" begin
    for (formulation, feedback, flow) in (
        (YBJPlus(), WaveMeanFeedback(), EvolvingFlow()),
        (YBJ(), WaveMeanFeedback(), EvolvingFlow()),
        (PassiveWave(), NoFeedback(), FixedFlow()),
    )
        host = architecture_test_model(CPU(); formulation, feedback, flow)
        device = architecture_test_model(EMULATED_GPU; formulation, feedback, flow)
        try
            @test architecture(device) === EMULATED_GPU
            @test device.fields.q isa JLArray
            @test device.runtime.dealias_mask isa JLArray{Bool, 2}
            @test device.runtime.device_coefficients.a_ell isa JLArray
            @test device.runtime.coefficients.a_ell == host.runtime.coefficients.a_ell

            wave = architecture_test_wave(host.grid)
            for model in (host, device)
                set!(model; ψ=architecture_test_psi, B=FieldArray(wave), verbose=false)
            end
            @test fields_match(device, host)

            host_stepper = ExponentialRungeKutta2(Δt=0.02)
            device_stepper = ExponentialRungeKutta2(Δt=0.02)
            for _ in 1:4
                step!(host, host_stepper)
                step!(device, device_stepper)
            end
            @test device_stepper.workspace.etd.Eq isa JLArray
            @test fields_match(device, host)

            compute_total_velocities!(host; t=0.4, compute_w=true)
            compute_total_velocities!(device; t=0.4, compute_w=true)
            @test fields_match(device, host)
            compute_ybj_vertical_velocity!(host; t=0.4)
            compute_ybj_vertical_velocity!(device; t=0.4)
            @test Array(device.fields.w) ≈ host.fields.w rtol=1e-13

            @test collect(QGYBJplus._energy_components(device, device.fields)) ≈
                  collect(QGYBJplus._energy_components(host, host.fields)) rtol=1e-12
            @test flow_kinetic_energy(device) ≈ flow_kinetic_energy(host) rtol=1e-12
            @test collect(wave_energy(device)) ≈ collect(wave_energy(host)) rtol=1e-12

            # Any smooth field with zero horizontal mean serves as a vorticity.
            for model in (host, device)
                set!(model; ζ=architecture_test_psi, verbose=false)
            end
            @test fields_match(device, host)
        finally
            finalize_model!(host)
            finalize_model!(device)
        end
    end
end

@testset "Single-precision models" begin
    double = architecture_test_model(CPU())
    single = architecture_test_model(CPU(); precision=Float32)
    device = architecture_test_model(EMULATED_GPU; precision=Float32)
    try
        @test eltype(single.grid) === Float32
        @test single.fields.q isa Array{ComplexF32, 3}
        @test single.fields.u isa Array{Float32, 3}
        @test eltype(single.runtime.device_coefficients.a_ell) === Float32
        @test device.fields.q isa JLArray{ComplexF32, 3}
        field_bytes(model) = sum(name -> sizeof(getfield(model.fields, name)),
                                 fieldnames(typeof(model.fields)))
        @test 2 * field_bytes(single) == field_bytes(double)

        wave = architecture_test_wave(double.grid)
        models = (double, single, device)
        for model in models
            set!(model; ψ=architecture_test_psi, B=FieldArray(wave), verbose=false)
        end
        steppers = map(_ -> ExponentialRungeKutta2(Δt=0.02), models)
        for _ in 1:4, (model, stepper) in zip(models, steppers)
            step!(model, stepper)
        end
        # Single precision tracks double precision to its round-off, and the
        # emulated GPU reproduces the single-precision CPU model.
        @test fields_match(single, double; rtol=1e-4)
        @test fields_match(device, single; rtol=1e-5)
        @test eltype(steppers[2].workspace.rhsq0) === ComplexF32
        @test all(buffer -> eltype(buffer) === ComplexF32,
                  single.runtime.workspace.pool.buffers)

        # Output and restart keep single-precision values exactly.
        mktempdir() do directory
            simulation = Simulation(single; Δt=0.02, stop_iteration=1,
                output=NetCDFOutput(path=directory, schedule=IterationInterval(1)),
                diagnostics=false, verbose=false)
            run!(simulation)
            restored = architecture_test_model(CPU(); precision=Float32)
            try
                last_file = last(sort(filter(endswith(".nc"), readdir(directory; join=true))))
                restore!(restored, last_file)
                @test restored.fields.q == single.fields.q
                @test restored.fields.B == single.fields.B
            finally
                finalize_model!(restored)
            end
        end
    finally
        foreach(finalize_model!, (double, single, device))
    end
end

@testset "Emulated GPU initial conditions" begin
    host = architecture_test_model(CPU())
    device = architecture_test_model(EMULATED_GPU)
    try
        random = RandomStreamfunction(amplitude=0.4, spectral_slope=-3, seed=42)
        for model in (host, device)
            set!(model; mean_flow=random, pv_method=:qg, verbose=false)
        end
        # JLArrays kernels use host math, so the hash-based spectrum is
        # bitwise identical (a GPU math library may round differently).
        @test Array(device.fields.psi) == host.fields.psi
        @test fields_match(device, host)

        for model in (host, device)
            set!(model; ψ=architecture_test_psi, pv_method=:barotropic,
                 waves=SurfaceWave(amplitude=0.1, scale=0.3), verbose=false)
        end
        @test fields_match(device, host)

        spectral = zeros(ComplexF64, 6, 16, 12)
        spectral[:, 2, 1] .= 1 + 0.2im
        for model in (host, device)
            set!(model; B=FieldArray(spectral; space=:spectral), verbose=false)
        end
        @test Array(device.fields.B) == host.fields.B
        @test fields_match(device, host)

        set_wave_packet!(device; amplitude=0.1, kx=1, ky=1, sigma_k=0.5)
        @test maximum(abs, device.fields.A) > 0
    finally
        finalize_model!(host)
        finalize_model!(device)
    end
end

@testset "Emulated GPU instantaneous particles reproduce the CPU" begin
    # Particles that follow the inertial oscillation apply the phase of the
    # wave amplitude at every stage, in the advection kernel on the device.
    positions = map((CPU(), EMULATED_GPU)) do architecture
        model = architecture_test_model(architecture)
        try
            set!(model; ψ=architecture_test_psi,
                 B=FieldArray(architecture_test_wave(model.grid)), verbose=false)
            initialize_particles!(model, particles_in_box(
                Float64, -0.5; x_max=2π, y_max=2π, nx=3, ny=2,
                integration_method=:rk4, wave_motion=:instantaneous))
            for n in 0:7
                advect_particles!(model, 0.05; current_time=0.05n)
            end
            particles = model.particles.particles
            # Positions and the advecting velocity stay on the architecture.
            @test QGYBJplus.architecture(particles.x) == QGYBJplus.architecture(model)
            @test QGYBJplus.architecture(model.particles.current.u) ==
                  QGYBJplus.architecture(model)
            (Array(particles.x), Array(particles.y), Array(particles.z))
        finally
            finalize_model!(model)
        end
    end
    for component in 1:3
        @test positions[2][component] ≈ positions[1][component] rtol=1e-12
    end
end

@testset "Emulated GPU simulation output, diagnostics, and particles" begin
    mktempdir() do directory
        outputs = Dict{Symbol, String}()
        models = Dict{Symbol, Any}()
        for (tag, architecture) in ((:host, CPU()), (:device, EMULATED_GPU))
            model = architecture_test_model(architecture)
            models[tag] = model
            set!(model; ψ=architecture_test_psi,
                 B=FieldArray(architecture_test_wave(model.grid)), verbose=false)
            initialize_particles!(model, particles_in_box(
                Float64, -0.5; x_max=2π, y_max=2π, nx=3, ny=2,
                integration_method=:rk2))
            path = joinpath(directory, string(tag))
            outputs[tag] = path
            simulation = Simulation(model; Δt=0.02, stop_iteration=2,
                output=NetCDFOutput(path=path, schedule=IterationInterval(1),
                                    fields=(:ψ, :waves), velocities=true),
                diagnostics=EnergyDiagnosticsOutput(
                    path=joinpath(path, "diagnostic"), schedule=IterationInterval(1)),
                particle_output=ParticleOutputManager(path;
                    save_interval_iter=1, save_interval_time=0.0),
                verbose=false)
            run!(simulation)
            @test simulation.state == Stopped
        end

        host, device = models[:host], models[:device]
        try
            @test fields_match(device, host)
            @test Array(device.particles.particles.x) ≈ host.particles.particles.x rtol=1e-12
            @test Array(device.particles.particles.y) ≈ host.particles.particles.y rtol=1e-12

            for name in ("state0001.nc", "state0003.nc")
                NCDataset(joinpath(outputs[:host], name)) do host_file
                    NCDataset(joinpath(outputs[:device], name)) do device_file
                        for variable in ("psi", "LA_real", "A_imag", "q_hat_real",
                                         "B_hat_imag", "u", "v", "w")
                            @test device_file[variable][:, :, :] ≈
                                  host_file[variable][:, :, :] rtol=1e-12
                        end
                    end
                end
            end
            NCDataset(joinpath(outputs[:host], "diagnostic", "total_energy.nc")) do host_file
                NCDataset(joinpath(outputs[:device], "diagnostic", "total_energy.nc")) do device_file
                    @test device_file["total_energy"][:] ≈
                          host_file["total_energy"][:] rtol=1e-12
                end
            end

            # Restoring a state file onto the device rebuilds the diagnostics.
            last_state = joinpath(outputs[:host], "state0003.nc")
            fill!(device.fields.q, 0)
            fill!(device.fields.B, 0)
            restore!(device, last_state)
            restore!(host, last_state)
            @test fields_match(device, host)
        finally
            finalize_model!(host)
            finalize_model!(device)
        end
    end
end

@testset "Emulated GPU kernels never receive host arrays" begin
    @test EMULATED_LAUNCHES[] > 0
    @test isempty(HOST_ARRAY_KERNELS)
end
