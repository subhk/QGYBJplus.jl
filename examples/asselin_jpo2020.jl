# Asselin et al. (2020), JPO 50, 3439–3454.
# Run with: mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl

using QGYBJplus

L = 70.0e3
H = 3.0e3
f = 1.24e-4
N² = 1.0e-5
U = 0.335
inertial_period = 2π / f

κ = sqrt(2) * π / L
ψ_scale = U / κ

# dipole
ψ₀ = (X, Y, _) -> ψ_scale * sin(κ * (X - Y) / sqrt(2)) * cos(κ * (X + Y) / sqrt(2))

grid = RectilinearGrid(size=(256, 256, 128),
                       x=(-L / 2, L / 2),
                       y=(-L / 2, L / 2),
                       z=(-H, 0.0))

model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=f),
                   stratification=ConstantStratification(N²=N²),
                   closure=HorizontalHyperdiffusivity(
                       flow=FlowHyperdiffusivity(coefficient=0, order=4),
                       wave=WaveHyperdiffusivity(coefficient=1.0e5, order=4)),
                   flow=FixedFlow(),
                   feedback=NoFeedback(),
                   formulation=YBJPlus())

set!(model;
    ψ=ψ₀,
    pv_method=:barotropic,
    waves=SurfaceWave(amplitude=0.10, scale=30.0),)

simulation = Simulation(model;
                        Δt=2.0,
                        stop_time=10 * inertial_period,
                        output=NetCDFOutput(path="output_asselin",
                            schedule=TimeInterval(5 * inertial_period)))

try
    run!(simulation; progress=true, diagnostics_interval=1000)
finally
    finalize_simulation!(simulation)
end
