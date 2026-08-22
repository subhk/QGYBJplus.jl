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
ψ₀ = (X, Y, _) -> ψ_scale *
    sin(κ * (X - Y) / sqrt(2)) * cos(κ * (X + Y) / sqrt(2))

grid = RectilinearGrid(
    size=(256, 256, 128),
    extent=(L, L, H),
    centered=true,
)

model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=f),
    stratification=ConstantStratification(N²=N²),
    closure=HorizontalHyperdiffusivity(
        flow=0,
        flow2=0,
        waves=1.0e5,
        waves2=0,
    ),
    flow=FixedFlow(),
    feedback=NoFeedback(),
    formulation=YBJPlus(),
)

set!(
    model;
    ψ=ψ₀,
    pv_method=:barotropic,
    waves=SurfaceWave(amplitude=0.10, scale=30.0),
)

simulation = Simulation(
    model;
    Δt=2.0,
    stop_time=15 * inertial_period,
    output=NetCDFOutput(
        path="output_asselin",
        schedule=TimeInterval(5 * inertial_period),
    ),
)

try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
