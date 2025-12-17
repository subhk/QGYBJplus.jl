#=
================================================================================
                    pretty_printing.jl - Pretty Display Utilities
================================================================================

Custom show methods for QGYBJ types, inspired by Oceananigans.jl style.
Provides nicely formatted output with Unicode box characters.

================================================================================
=#

using Printf

# Import types from parent module
using ..QGYBJ: QGParams, Grid, State, OutputConfig, Plans, ParallelConfig,
               DomainConfig, StratificationConfig, InitialConditionConfig, ModelConfig
using ..QGYBJ.UnifiedParticleAdvection: ParticleConfig, ParticleTracker
using ..QGYBJ.UnifiedParticleAdvection.EnhancedParticleConfig: ParticleConfig3D

# ============================================================================
#                       FORMATTING UTILITIES
# ============================================================================

"""
    format_number(x) -> String

Format a number for display with appropriate precision.
"""
function format_number(x::Real)
    if x == 0
        return "0"
    elseif abs(x) >= 1e4 || abs(x) < 1e-3
        return @sprintf("%.3e", x)
    elseif x == round(x)
        return string(Int(x))
    else
        return @sprintf("%.4g", x)
    end
end

format_number(x::Int) = string(x)
format_number(x::Nothing) = "nothing"
format_number(x::Symbol) = string(x)
format_number(x::Bool) = x ? "true" : "false"
format_number(x::String) = "\"$x\""

"""
    format_tuple(t) -> String

Format a tuple for display.
"""
format_tuple(t::Tuple) = "(" * join(format_number.(t), ", ") * ")"

"""
    format_size(dims...) -> String

Format grid dimensions as "nx × ny × nz".
"""
format_size(dims...) = join(dims, " × ")

# ============================================================================
#                       BOX DRAWING CHARACTERS
# ============================================================================

const BOX_TL = "┌"  # Top-left corner
const BOX_TR = "┐"  # Top-right corner
const BOX_BL = "└"  # Bottom-left corner
const BOX_BR = "┘"  # Bottom-right corner
const BOX_H  = "─"  # Horizontal line
const BOX_V  = "│"  # Vertical line
const BOX_LT = "├"  # Left tee
const BOX_RT = "┤"  # Right tee

"""
    print_box_top(io, title, width)

Print the top border of a box with title.
"""
function print_box_top(io::IO, title::String, width::Int)
    title_str = " $title "
    title_len = length(title_str)
    left_pad = 2
    right_pad = width - left_pad - title_len - 1
    println(io, BOX_TL, repeat(BOX_H, left_pad), title_str, repeat(BOX_H, max(0, right_pad)), BOX_TR)
end

"""
    print_box_bottom(io, width)

Print the bottom border of a box.
"""
function print_box_bottom(io::IO, width::Int)
    println(io, BOX_BL, repeat(BOX_H, width), BOX_BR)
end

"""
    print_box_separator(io, width)

Print a horizontal separator inside the box.
"""
function print_box_separator(io::IO, width::Int)
    println(io, BOX_LT, repeat(BOX_H, width), BOX_RT)
end

"""
    print_box_row(io, key, value, width; key_width=20)

Print a row inside the box: │ key: value │
"""
function print_box_row(io::IO, key::String, value::String, width::Int; key_width::Int=20)
    content = rpad("$key:", key_width) * value
    padding = width - length(content) - 1
    println(io, BOX_V, " ", content, repeat(" ", max(0, padding)), BOX_V)
end

"""
    print_section_header(io, title, width)

Print a section header inside the box.
"""
function print_section_header(io::IO, title::String, width::Int)
    title_str = " $title "
    left_pad = 1
    right_pad = width - left_pad - length(title_str) - 1
    println(io, BOX_LT, repeat(BOX_H, left_pad), title_str, repeat(BOX_H, max(0, right_pad)), BOX_RT)
end

# ============================================================================
#                       QGParams PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", par::QGParams{T}) where T
    width = 60
    key_width = 22

    print_box_top(io, "QGParams{$T}", width)

    # Domain section
    print_section_header(io, "Domain", width)
    print_box_row(io, "Resolution (nx×ny×nz)", format_size(par.nx, par.ny, par.nz), width; key_width)
    print_box_row(io, "Domain size (Lx×Ly)", "$(format_number(par.Lx)) × $(format_number(par.Ly))", width; key_width)

    # Time stepping section
    print_section_header(io, "Time Stepping", width)
    print_box_row(io, "Time step (dt)", format_number(par.dt), width; key_width)
    print_box_row(io, "Number of steps (nt)", format_number(par.nt), width; key_width)
    total_time = par.dt * par.nt
    print_box_row(io, "Total time", format_number(total_time), width; key_width)
    T_inertial = 2π / par.f₀
    print_box_row(io, "Inertial periods", format_number(total_time / T_inertial), width; key_width)

    # Physical parameters section
    print_section_header(io, "Physical Parameters", width)
    print_box_row(io, "Coriolis (f₀)", format_number(par.f₀), width; key_width)
    print_box_row(io, "Stratification (N²)", format_number(par.N²), width; key_width)
    print_box_row(io, "Wave-flow ratio (W2F)", format_number(par.W2F), width; key_width)
    print_box_row(io, "Stratification type", string(par.stratification), width; key_width)

    # Derived quantities
    disp_coeff = par.N² / (2 * par.f₀)
    print_box_row(io, "Dispersion coeff", format_number(disp_coeff), width; key_width)
    print_box_row(io, "Inertial period", format_number(T_inertial), width; key_width)

    # Hyperdiffusion section
    print_section_header(io, "Hyperdiffusion (Flow)", width)
    print_box_row(io, "νₕ₁ (∇^$(2*par.ilap1))", format_number(par.νₕ₁), width; key_width)
    print_box_row(io, "νₕ₂ (∇^$(2*par.ilap2))", format_number(par.νₕ₂), width; key_width)

    print_section_header(io, "Hyperdiffusion (Waves)", width)
    print_box_row(io, "νₕ₁ʷ (∇^$(2*par.ilap1w))", format_number(par.νₕ₁ʷ), width; key_width)
    print_box_row(io, "νₕ₂ʷ (∇^$(2*par.ilap2w))", format_number(par.νₕ₂ʷ), width; key_width)
    print_box_row(io, "νz (vertical)", format_number(par.νz), width; key_width)
    print_box_row(io, "R-A filter (γ)", format_number(par.γ), width; key_width)

    # Physics switches
    print_section_header(io, "Physics Flags", width)
    print_box_row(io, "YBJ+ formulation", format_number(par.ybj_plus), width; key_width)
    print_box_row(io, "Fixed flow", format_number(par.fixed_flow), width; key_width)
    print_box_row(io, "No wave feedback", format_number(par.no_wave_feedback), width; key_width)
    print_box_row(io, "Inviscid", format_number(par.inviscid), width; key_width)
    print_box_row(io, "Linear", format_number(par.linear), width; key_width)
    print_box_row(io, "No dispersion", format_number(par.no_dispersion), width; key_width)
    print_box_row(io, "Passive scalar", format_number(par.passive_scalar), width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, par::QGParams{T}) where T
    print(io, "QGParams{$T}($(par.nx)×$(par.ny)×$(par.nz), dt=$(format_number(par.dt)), nt=$(par.nt))")
end

# ============================================================================
#                       Grid PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", G::Grid{T,AT}) where {T,AT}
    width = 55
    key_width = 20

    array_type = AT <: Array ? "CPU" : string(AT)
    print_box_top(io, "Grid{$T} ($array_type)", width)

    # Resolution
    print_section_header(io, "Resolution", width)
    print_box_row(io, "Grid points", format_size(G.nx, G.ny, G.nz), width; key_width)

    # Domain size
    print_section_header(io, "Domain", width)
    print_box_row(io, "Lx", format_number(G.Lx), width; key_width)
    print_box_row(io, "Ly", format_number(G.Ly), width; key_width)

    # Grid spacing
    print_section_header(io, "Grid Spacing", width)
    print_box_row(io, "dx", format_number(G.dx), width; key_width)
    print_box_row(io, "dy", format_number(G.dy), width; key_width)
    print_box_row(io, "dz", format_number(G.dz), width; key_width)

    # z range
    if !isempty(G.z)
        print_box_row(io, "z range", "[$(format_number(first(G.z))), $(format_number(last(G.z)))]", width; key_width)
    end

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, G::Grid{T,AT}) where {T,AT}
    print(io, "Grid{$T}($(G.nx)×$(G.ny)×$(G.nz))")
end

# ============================================================================
#                       State PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", S::State{T,RT,CT}) where {T,RT,CT}
    width = 50
    key_width = 18

    print_box_top(io, "State{$T}", width)

    # Get size from one of the arrays
    sz = size(S.psi)
    print_box_row(io, "Size", format_size(sz...), width; key_width)

    # Fields
    print_section_header(io, "Fields", width)
    print_box_row(io, "Flow", "ψ (streamfunction), q (PV)", width; key_width)
    print_box_row(io, "Waves", "B (complex envelope)", width; key_width)
    print_box_row(io, "Velocities", "u, v, w", width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, S::State{T,RT,CT}) where {T,RT,CT}
    sz = size(S.psi)
    print(io, "State{$T}($(format_size(sz...)))")
end

# ============================================================================
#                       OutputConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::OutputConfig{T}) where T
    width = 55
    key_width = 22

    print_box_top(io, "OutputConfig{$T}", width)

    print_section_header(io, "Output Directory", width)
    print_box_row(io, "Path", cfg.output_dir, width; key_width)
    print_box_row(io, "State file pattern", cfg.state_file_pattern, width; key_width)

    print_section_header(io, "Save Intervals", width)
    print_box_row(io, "ψ interval", format_number(cfg.psi_interval), width; key_width)
    print_box_row(io, "Wave interval", format_number(cfg.wave_interval), width; key_width)
    print_box_row(io, "Diag interval", format_number(cfg.diagnostics_interval), width; key_width)

    print_section_header(io, "Variables to Save", width)
    print_box_row(io, "Streamfunction (ψ)", format_number(cfg.save_psi), width; key_width)
    print_box_row(io, "Waves (LAr, LAi)", format_number(cfg.save_waves), width; key_width)
    print_box_row(io, "Velocities", format_number(cfg.save_velocities), width; key_width)
    print_box_row(io, "Vorticity", format_number(cfg.save_vorticity), width; key_width)
    print_box_row(io, "Diagnostics", format_number(cfg.save_diagnostics), width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, cfg::OutputConfig)
    print(io, "OutputConfig(\"$(cfg.output_dir)\")")
end

# ============================================================================
#                       ParticleConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::ParticleConfig{T}) where T
    width = 55
    key_width = 22

    print_box_top(io, "ParticleConfig{$T}", width)

    print_section_header(io, "Particles", width)
    n_total = cfg.nx_particles * cfg.ny_particles
    print_box_row(io, "Grid (nx × ny)", "$(cfg.nx_particles) × $(cfg.ny_particles)", width; key_width)
    print_box_row(io, "Total particles", format_number(n_total), width; key_width)
    print_box_row(io, "Depth (z-level)", format_number(cfg.z_level), width; key_width)

    print_section_header(io, "Domain", width)
    print_box_row(io, "x range", "[$(format_number(cfg.x_min)), $(format_number(cfg.x_max))]", width; key_width)
    print_box_row(io, "y range", "[$(format_number(cfg.y_min)), $(format_number(cfg.y_max))]", width; key_width)

    print_section_header(io, "Integration", width)
    print_box_row(io, "Method", string(cfg.integration_method), width; key_width)
    print_box_row(io, "Save interval", format_number(cfg.save_interval), width; key_width)
    print_box_row(io, "Max save points", format_number(cfg.max_save_points), width; key_width)

    print_section_header(io, "Physics", width)
    print_box_row(io, "3D advection", format_number(cfg.use_3d_advection), width; key_width)
    print_box_row(io, "YBJ w-velocity", format_number(cfg.use_ybj_w), width; key_width)
    print_box_row(io, "Interpolation", string(cfg.interpolation_method), width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, cfg::ParticleConfig{T}) where T
    n_total = cfg.nx_particles * cfg.ny_particles
    print(io, "ParticleConfig{$T}(n=$n_total, z=$(format_number(cfg.z_level)))")
end

# ============================================================================
#                       ParticleConfig3D PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::ParticleConfig3D{T}) where T
    width = 55
    key_width = 22

    print_box_top(io, "ParticleConfig3D{$T}", width)

    print_section_header(io, "Distribution", width)
    print_box_row(io, "Type", string(cfg.distribution_type), width; key_width)
    n_total = cfg.nx_particles * cfg.ny_particles * cfg.nz_particles
    print_box_row(io, "Grid (nx×ny×nz)", "$(cfg.nx_particles)×$(cfg.ny_particles)×$(cfg.nz_particles)", width; key_width)
    print_box_row(io, "Total particles", format_number(n_total), width; key_width)

    print_section_header(io, "Domain", width)
    print_box_row(io, "x range", "[$(format_number(cfg.x_min)), $(format_number(cfg.x_max))]", width; key_width)
    print_box_row(io, "y range", "[$(format_number(cfg.y_min)), $(format_number(cfg.y_max))]", width; key_width)
    print_box_row(io, "z range", "[$(format_number(cfg.z_min)), $(format_number(cfg.z_max))]", width; key_width)

    print_section_header(io, "Integration", width)
    print_box_row(io, "Method", string(cfg.integration_method), width; key_width)
    print_box_row(io, "Save interval", format_number(cfg.save_interval), width; key_width)
    print_box_row(io, "Interpolation", string(cfg.interpolation_method), width; key_width)

    print_section_header(io, "Physics", width)
    print_box_row(io, "3D advection", format_number(cfg.use_3d_advection), width; key_width)
    print_box_row(io, "YBJ w-velocity", format_number(cfg.use_ybj_w), width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, cfg::ParticleConfig3D{T}) where T
    n_total = cfg.nx_particles * cfg.ny_particles * cfg.nz_particles
    print(io, "ParticleConfig3D{$T}($(cfg.distribution_type), n=$n_total)")
end

# ============================================================================
#                       ParticleTracker PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", tracker::ParticleTracker{T}) where T
    width = 55
    key_width = 22

    print_box_top(io, "ParticleTracker{$T}", width)

    print_section_header(io, "Status", width)
    n_active = tracker.n_active
    n_total = length(tracker.x)
    print_box_row(io, "Active particles", format_number(n_active), width; key_width)
    print_box_row(io, "Allocated slots", format_number(n_total), width; key_width)
    print_box_row(io, "History entries", format_number(length(tracker.history_t)), width; key_width)

    print_section_header(io, "Domain", width)
    print_box_row(io, "x range", "[0, $(format_number(tracker.Lx))]", width; key_width)
    print_box_row(io, "y range", "[0, $(format_number(tracker.Ly))]", width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, tracker::ParticleTracker)
    print(io, "ParticleTracker(n_active=$(tracker.n_active))")
end

# ============================================================================
#                       ParallelConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::ParallelConfig)
    width = 50
    key_width = 18

    print_box_top(io, "ParallelConfig", width)

    print_box_row(io, "Use MPI", format_number(cfg.use_mpi), width; key_width)
    if cfg.use_mpi && cfg.comm !== nothing
        print_box_row(io, "Parallel I/O", format_number(cfg.parallel_io), width; key_width)
    end

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, cfg::ParallelConfig)
    mpi_str = cfg.use_mpi ? "MPI" : "Serial"
    print(io, "ParallelConfig($mpi_str)")
end

# ============================================================================
#                       Plans PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", plans::Plans)
    width = 50
    key_width = 18

    print_box_top(io, "FFT Plans", width)

    print_box_row(io, "Forward 2D", "FFTW plan", width; key_width)
    print_box_row(io, "Inverse 2D", "FFTW plan", width; key_width)
    print_box_row(io, "Forward 1D (z)", "FFTW plan", width; key_width)
    print_box_row(io, "Inverse 1D (z)", "FFTW plan", width; key_width)

    print_box_bottom(io, width)
end

# Compact single-line show
function Base.show(io::IO, plans::Plans)
    print(io, "Plans(FFTW)")
end

# ============================================================================
#                       DomainConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::DomainConfig{T}) where T
    width = 50
    key_width = 18

    print_box_top(io, "DomainConfig{$T}", width)

    print_section_header(io, "Grid Resolution", width)
    print_box_row(io, "Points (nx×ny×nz)", format_size(cfg.nx, cfg.ny, cfg.nz), width; key_width)

    print_section_header(io, "Domain Size", width)
    print_box_row(io, "Lx", format_number(cfg.Lx), width; key_width)
    print_box_row(io, "Ly", format_number(cfg.Ly), width; key_width)
    print_box_row(io, "Lz", format_number(cfg.Lz), width; key_width)

    if cfg.dom_x_m !== nothing
        print_section_header(io, "Physical Size (m)", width)
        print_box_row(io, "x", format_number(cfg.dom_x_m), width; key_width)
        print_box_row(io, "y", format_number(cfg.dom_y_m), width; key_width)
        print_box_row(io, "z", format_number(cfg.dom_z_m), width; key_width)
    end

    print_box_bottom(io, width)
end

function Base.show(io::IO, cfg::DomainConfig)
    print(io, "DomainConfig($(cfg.nx)×$(cfg.ny)×$(cfg.nz))")
end

# ============================================================================
#                       StratificationConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::StratificationConfig{T}) where T
    width = 50
    key_width = 18

    print_box_top(io, "StratificationConfig{$T}", width)

    print_section_header(io, "Profile Type", width)
    print_box_row(io, "Type", string(cfg.type), width; key_width)

    if cfg.type == :constant_N
        print_box_row(io, "N₀", format_number(cfg.N0), width; key_width)
    elseif cfg.type == :skewed_gaussian
        print_section_header(io, "Skewed Gaussian", width)
        print_box_row(io, "N₀²", format_number(cfg.N02_sg), width; key_width)
        print_box_row(io, "N₁²", format_number(cfg.N12_sg), width; key_width)
        print_box_row(io, "σ", format_number(cfg.sigma_sg), width; key_width)
        print_box_row(io, "z₀", format_number(cfg.z0_sg), width; key_width)
        print_box_row(io, "α", format_number(cfg.alpha_sg), width; key_width)
    elseif cfg.type == :tanh_profile
        print_section_header(io, "Tanh Profile", width)
        print_box_row(io, "N upper", format_number(cfg.N_upper), width; key_width)
        print_box_row(io, "N lower", format_number(cfg.N_lower), width; key_width)
        print_box_row(io, "z pycno", format_number(cfg.z_pycno), width; key_width)
        print_box_row(io, "Width", format_number(cfg.width), width; key_width)
    elseif cfg.type == :from_file && cfg.filename !== nothing
        print_box_row(io, "File", cfg.filename, width; key_width)
    end

    print_box_bottom(io, width)
end

function Base.show(io::IO, cfg::StratificationConfig)
    print(io, "StratificationConfig($(cfg.type))")
end

# ============================================================================
#                       InitialConditionConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::InitialConditionConfig{T}) where T
    width = 50
    key_width = 18

    print_box_top(io, "InitialConditionConfig{$T}", width)

    print_section_header(io, "Streamfunction (ψ)", width)
    print_box_row(io, "Type", string(cfg.psi_type), width; key_width)
    print_box_row(io, "Amplitude", format_number(cfg.psi_amplitude), width; key_width)
    if cfg.psi_filename !== nothing
        print_box_row(io, "File", cfg.psi_filename, width; key_width)
    end

    print_section_header(io, "Wave Field (L⁺A)", width)
    print_box_row(io, "Type", string(cfg.wave_type), width; key_width)
    print_box_row(io, "Amplitude", format_number(cfg.wave_amplitude), width; key_width)
    if cfg.wave_filename !== nothing
        print_box_row(io, "File", cfg.wave_filename, width; key_width)
    end

    print_box_row(io, "Random seed", format_number(cfg.random_seed), width; key_width)

    print_box_bottom(io, width)
end

function Base.show(io::IO, cfg::InitialConditionConfig)
    print(io, "InitialConditionConfig(ψ=$(cfg.psi_type), waves=$(cfg.wave_type))")
end

# ============================================================================
#                       ModelConfig PRETTY PRINTING
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cfg::ModelConfig{T}) where T
    width = 55
    key_width = 20

    print_box_top(io, "ModelConfig{$T}", width)

    # Domain summary
    print_section_header(io, "Domain", width)
    d = cfg.domain
    print_box_row(io, "Resolution", format_size(d.nx, d.ny, d.nz), width; key_width)
    print_box_row(io, "Size (Lx×Ly×Lz)", "$(format_number(d.Lx))×$(format_number(d.Ly))×$(format_number(d.Lz))", width; key_width)

    # Time stepping
    print_section_header(io, "Time Stepping", width)
    print_box_row(io, "dt", format_number(cfg.dt), width; key_width)
    print_box_row(io, "Total time", format_number(cfg.total_time), width; key_width)
    n_steps = round(Int, cfg.total_time / cfg.dt)
    print_box_row(io, "Steps", format_number(n_steps), width; key_width)

    # Physics
    print_section_header(io, "Physics", width)
    print_box_row(io, "f₀ (Coriolis)", format_number(cfg.f0), width; key_width)
    print_box_row(io, "Stratification", string(cfg.stratification.type), width; key_width)

    # Initial conditions
    print_section_header(io, "Initial Conditions", width)
    print_box_row(io, "ψ type", string(cfg.initial_conditions.psi_type), width; key_width)
    print_box_row(io, "Wave type", string(cfg.initial_conditions.wave_type), width; key_width)

    # Output
    print_section_header(io, "Output", width)
    print_box_row(io, "Directory", cfg.output.output_dir, width; key_width)

    print_box_bottom(io, width)
end

function Base.show(io::IO, cfg::ModelConfig)
    d = cfg.domain
    print(io, "ModelConfig($(d.nx)×$(d.ny)×$(d.nz), T=$(format_number(cfg.total_time)))")
end
