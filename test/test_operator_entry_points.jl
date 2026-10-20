using Test
using FFTW
using QGYBJplus

operator_test_model(; kwargs...) = QGYBJModel(;
    grid=RectilinearGrid(size=(16, 16, 6), extent=(1e5, 1e5, 1000.0)),
    coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
    verbose=false, kwargs...)

eddy(x, y, z) = 1e3 * sin(2π * x / 1e5) * cos(2π * y / 1e5) * (1 + z / 2000)
packet_waves(model) = set_wave_packet!(model; amplitude=0.1, kx=2, ky=1, sigma_k=1.0)

@testset "Transform plans follow the grid's precision" begin
    grid = RectilinearGrid(Float32; size=(16, 16, 4), extent=(2π, 2π, 1.0))
    geometry = QGYBJplus.RuntimeGeometry(grid)
    plans = QGYBJplus.plan_transforms!(geometry)
    @test plans isa QGYBJplus.Plans{<:Any, <:Any, ComplexF32}
    # A plan given another element type transformed a converted copy and
    # left the array as it was.
    mismatched = zeros(ComplexF64, 4, 16, 16)
    @test_throws ArgumentError fft_forward!(mismatched, mismatched, plans)
    @test_throws ArgumentError fft_backward!(mismatched, mismatched, plans)

    # Without plans, the low-level velocity was off by 127 on a Float32 grid.
    fields = allocate_fields(grid)
    psi = zeros(ComplexF32, 4, 16, 16)
    psi[:, 2, 1] .= psi[:, 16, 1] .= 16 * 16 / 2   # ψ = cos x, unnormalised
    copyto!(fields.psi, psi)
    compute_velocities!(fields, geometry; compute_w=false)
    @test maximum(abs, fields.v) ≈ 1 rtol=1e-5
    @test eltype(fields.v) === Float32

    @test_throws ArgumentError RectilinearGrid(Float16; size=(4, 4, 2), extent=(1.0, 1.0, 1.0))
    @test_throws ArgumentError RectilinearGrid(BigFloat; size=(4, 4, 2), extent=(1.0, 1.0, 1.0))
end

@testset "invert_B_to_A! follows the model's formulation" begin
    # It used L⁺ for YBJ() too, changing A by 2 to 67%.
    model = operator_test_model(formulation=YBJ())
    set!(model; ψ=eddy)
    packet_waves(model)
    A = copy(model.fields.A)
    compute_velocities!(model)
    w = copy(model.fields.w)
    invert_B_to_A!(model)
    @test model.fields.A ≈ A rtol=1e-10
    @test model.fields.w == w   # the flow does not change, so neither does w

    plus = operator_test_model(formulation=YBJPlus())
    packet_waves(plus)
    A = copy(plus.fields.A)
    fill!(plus.fields.A, 0)
    invert_B_to_A!(plus)
    @test plus.fields.A ≈ A
end

@testset "PassiveWave keeps A = L⁺⁻¹B after a step" begin
    # A step cleared A and C, while set!, output, and the energies use L⁺⁻¹B.
    model = operator_test_model(formulation=PassiveWave())
    set!(model; ψ=eddy)
    packet_waves(model)
    run!(Simulation(model; Δt=600.0, stop_iteration=1, output=false,
                    diagnostics=false, verbose=false))
    A = copy(model.fields.A)
    @test maximum(abs, A) > 0
    invert_B_to_A!(model)
    @test model.fields.A ≈ A
end

@testset "A YBJ fixed flow keeps the velocity of its full streamfunction" begin
    # ψ with modes beyond the two-thirds cutoff: refreshing the waves used to
    # leave the velocity of the projected ψ.
    sharp(x, y, z) = eddy(x, y, z) + 50 * sin(2π * 7 * x / 1e5)
    model = operator_test_model(formulation=YBJ(), flow=FixedFlow())
    set!(model; ψ=sharp, pv_method=:none)
    compute_velocities!(model; compute_w=false)
    u = copy(model.fields.u)
    packet_waves(model)   # refreshes the wave diagnostics
    @test model.fields.u ≈ u
end

@testset "Model energies are domain means without side effects" begin
    model = operator_test_model()
    speed = 0.1
    set!(model; ψ=(x, y, z) -> -speed * 1e5 / (2π) * sin(2π * y / 1e5))
    packet_waves(model)
    compute_velocities!(model)
    w = copy(model.fields.w)
    A = copy(model.fields.A)
    u, v = model.fields.u, model.fields.v
    energy = flow_kinetic_energy(model)
    @test energy ≈ 0.5 * sum(u .^ 2 .+ v .^ 2) / length(u) rtol=1e-10
    @test energy ≈ speed^2 / 4 rtol=1e-8
    waves = wave_energy(model)
    @test keys(waves) == (:kinetic, :potential, :correction)
    @test all(>(0), values(waves))
    # The kinetic energy is ½⟨|LA|²⟩, with LA = B + kₕ²A/4 for YBJ⁺, over
    # the modes a nonlinear model keeps.
    geometry = model.runtime.geometry
    kh² = reshape(geometry.kx .^ 2, 1, :, 1) .+ reshape(geometry.ky .^ 2, 1, 1, :)
    kept = reshape(dealias_mask(model.grid), 1, 16, 16)
    LA = ifft((model.fields.B .+ kh² .* model.fields.A ./ 4) .* kept, (2, 3))
    @test waves.kinetic ≈ 0.5 * sum(abs2, LA) / length(LA) rtol=1e-8
    # Computing them changed w (and, for YBJ, A) before.
    @test model.fields.w == w
    @test model.fields.A == A
end

@testset "Spectral flow energies are domain means" begin
    model = operator_test_model()
    set!(model; ψ=eddy)
    geometry = model.runtime.geometry
    uk = complex.(model.fields.u)
    vk = complex.(model.fields.v)
    fft_forward!(uk, uk, model.runtime.plans)
    fft_forward!(vk, vk, model.runtime.plans)
    mean_energy = 0.5 * sum(model.fields.u .^ 2 .+ model.fields.v .^ 2) / length(model.fields.u)
    # They returned 2(nx ny)² times this, less a spurious mean-mode term.
    @test QGYBJplus.flow_kinetic_energy_spectral(uk, vk, geometry) ≈ mean_energy
    uk[:, 1, 1] .+= 3 * 16 * 16   # a uniform flow of 3 adds 9/2
    @test QGYBJplus.flow_kinetic_energy_spectral(uk, vk, geometry) ≈ mean_energy + 4.5
    a = fill(2.0, 6)
    @test QGYBJplus.flow_potential_energy_spectral(vk, geometry, a) ≈
          2 * 0.5 * sum(model.fields.v .^ 2) / length(model.fields.v)
end

@testset "Stencil operators reject aliased or mis-sized arrays" begin
    model = operator_test_model()
    set!(model; ψ=eddy)
    geometry = model.runtime.geometry
    fields = model.fields
    a = model.runtime.coefficients.a_ell
    psi = copy(fields.psi)
    # In place, these read neighbours they had already overwritten.
    @test_throws ArgumentError QGYBJplus.compute_q_from_psi!(psi, psi, geometry, a, geometry.dz)
    @test_throws ArgumentError QGYBJplus.Nonlinear.dissipation_q_nv!(psi, psi, 1.0, geometry)
    @test_throws ArgumentError QGYBJplus.Diagnostics.omega_eqn_rhs!(psi, psi, geometry,
                                                                    model.runtime.plans)
    small = zeros(ComplexF64, 6, 8, 8)
    @test_throws DimensionMismatch invert_helmholtz!(psi, small, geometry; a)
    @test_throws DimensionMismatch QGYBJplus.compute_q_from_psi!(small, psi, geometry, a,
                                                                geometry.dz)
    @test_throws DimensionMismatch QGYBJplus.Diagnostics.omega_eqn_rhs!(small, psi, geometry,
                                                                        model.runtime.plans)
    sigma = zeros(ComplexF64, 16, 16)
    @test_throws DimensionMismatch QGYBJplus.YBJNormal.compute_A!(
        fields.A, fields.C, small, sigma, geometry; f=1e-4)
    @test_throws ArgumentError QGYBJplus.YBJNormal.compute_A!(
        fields.A, fields.C, fields.A, sigma, geometry; f=1e-4)
end

@testset "Float64 coefficients reach Float32 kernels in Float32" begin
    grid = RectilinearGrid(Float32; size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
                       stratification=ConstantStratification(N²=1.0), verbose=false)
    set!(model; ψ=(x, y, z) -> sin(x) * cos(y) * (1 + z))
    geometry = model.runtime.geometry
    a64 = Float64.(model.runtime.coefficients.a_ell)
    @test QGYBJplus.in_precision(a64, Float32) isa Vector{Float32}
    @test QGYBJplus.in_precision(complex.(a64), Float32) isa Vector{ComplexF32}
    q = similar(model.fields.q)
    QGYBJplus.compute_q_from_psi!(q, model.fields.psi, geometry, a64, Float64(geometry.dz))
    @test eltype(q) === ComplexF32
    @test q ≈ model.fields.q rtol=1e-5
    fields = copy_fields(model.fields)
    invert_q_to_psi!(fields, geometry; a=a64)
    @test fields.psi ≈ model.fields.psi rtol=1e-4
end
