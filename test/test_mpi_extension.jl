#=
================================================================================
            MPI Extension Test Suite for QGYBJplus.jl
================================================================================

This test suite verifies that the MPI extension works correctly with
PencilArrays and PencilFFTs.

RUNNING:
--------
Serial test (no MPI):
    julia --project test/test_mpi_extension.jl --serial

Parallel test (with MPI):
    mpiexecjl -n 4 julia --project test/test_mpi_extension.jl

================================================================================
=#

using Test

# Check if we're running in serial mode
const SERIAL_MODE = "--serial" in ARGS

# Test domain size (used throughout)
const TEST_Lx = 500e3  # 500 km
const TEST_Ly = 500e3  # 500 km
const TEST_Lz = 4000.0 # 4 km

function run_serial_tests()
    # Load QGYBJplus using @eval to allow non-top-level loading
    @eval using QGYBJplus
    @eval using QGYBJplus: QGParams, Grid, State, Plans, plan_transforms!, fft_forward!, fft_backward!

    println("=" ^ 60)
    println("QGYBJplus.jl Serial Mode Test")
    println("=" ^ 60)
    println()

    @testset "Serial Mode Tests" begin
        @testset "Module Loading" begin
            @test @isdefined QGYBJplus
            println("  ✓ QGYBJplus module loaded")
        end

        @testset "Basic Types" begin
            params = QGYBJplus.default_params(nx=32, ny=32, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
            @test params.nx == 32
            @test params.ny == 32
            @test params.nz == 16
            println("  ✓ QGParams created")

            grid = QGYBJplus.init_grid(params)
            @test grid.nx == 32
            @test grid.decomp === nothing  # No MPI decomposition
            println("  ✓ Grid created (serial mode)")

            state = QGYBJplus.init_state(grid)
            @test size(state.psi) == (32, 32, 16)
            println("  ✓ State created")
        end

        @testset "FFT Transforms (Serial)" begin
            params = QGYBJplus.default_params(nx=32, ny=32, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
            grid = QGYBJplus.init_grid(params)
            plans = QGYBJplus.plan_transforms!(grid)

            @test plans.backend == :fftw
            println("  ✓ FFTW plans created")

            # Test FFT roundtrip
            src = randn(ComplexF64, 32, 32, 16)
            dst = similar(src)
            dst2 = similar(src)

            QGYBJplus.fft_forward!(dst, src, plans)
            QGYBJplus.fft_backward!(dst2, dst, plans)

            # FFTW.ifft is normalized (divides by N automatically)
            # So no manual normalization needed

            @test isapprox(src, dst2, rtol=1e-10)
            println("  ✓ FFT roundtrip successful")
        end

        @testset "MPI Stubs (Without MPI)" begin
            # These should throw informative errors
            @test_throws ErrorException QGYBJplus.setup_mpi_environment()
            @test_throws ErrorException QGYBJplus.init_mpi_grid(nothing, nothing)
            println("  ✓ MPI stubs throw appropriate errors")
        end
    end

    println()
    println("All serial tests passed!")
end

function run_mpi_tests()
    # Load MPI packages using @eval
    @eval using MPI
    @eval using PencilArrays
    @eval using PencilFFTs
    @eval using QGYBJplus

    MPI.Init()

    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)

        if rank == 0
            println("=" ^ 60)
            println("QGYBJplus.jl MPI Extension Test Suite")
            println("=" ^ 60)
            println("Processes: $nprocs")
            println()
        end

        @testset "MPI Extension Tests" begin
            @testset "MPI Environment Setup" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                @test mpi_config.nprocs == nprocs
                @test mpi_config.rank == rank
                @test mpi_config.is_root == (rank == 0)
                if rank == 0
                    println("  ✓ MPI environment initialized")
                end
            end

            @testset "Parallel Grid" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)

                grid = QGYBJplus.init_mpi_grid(params, mpi_config)
                @test grid.nx == 64
                @test grid.ny == 64
                @test grid.nz == 32
                @test grid.decomp !== nothing  # Has MPI decomposition

                if rank == 0
                    println("  ✓ Parallel grid created with decomposition")
                end
            end

            @testset "Parallel State" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
                grid = QGYBJplus.init_mpi_grid(params, mpi_config)

                state = QGYBJplus.init_mpi_state(grid, mpi_config)

                # Check that arrays are PencilArrays
                @test typeof(state.psi) <: PencilArray
                @test typeof(state.B) <: PencilArray
                @test typeof(state.u) <: PencilArray

                if rank == 0
                    println("  ✓ Parallel state created with PencilArrays")
                end
            end

            @testset "Parallel FFT Plans" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
                grid = QGYBJplus.init_mpi_grid(params, mpi_config)

                plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)

                # Note: MPIPlans uses ldiv!(dst, forward, src) for inverse FFT
                # so there's no separate 'backward' field
                @test plans.forward !== nothing
                @test plans.pencils_match  # Should be true for C2C with NoTransform on z

                if rank == 0
                    println("  ✓ PencilFFT plans created")
                end
            end

            @testset "Parallel FFT Execution" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
                grid = QGYBJplus.init_mpi_grid(params, mpi_config)
                state = QGYBJplus.init_mpi_state(grid, mpi_config)
                plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)

                # Initialize with random data
                QGYBJplus.init_mpi_random_field!(state.psi, grid, 1.0, 0)

                # Perform FFT roundtrip
                work_k = similar(state.psi)
                work = similar(state.psi)

                QGYBJplus.fft_forward!(work_k, state.psi, plans)
                QGYBJplus.fft_backward!(work, work_k, plans)

                # PencilFFTs ldiv! (used in fft_backward!) is normalized
                # So no manual normalization needed

                # Check roundtrip accuracy
                parent_psi = parent(state.psi)
                parent_work = parent(work)
                local_error = maximum(abs.(parent_psi .- parent_work))

                # Reduce max error across all ranks
                global_error = MPI.Allreduce(local_error, MPI.MAX, comm)

                @test global_error < 1e-10

                if rank == 0
                    println("  ✓ Parallel FFT roundtrip successful (error: $global_error)")
                end
            end

            @testset "MPI Communication" begin
                mpi_config = QGYBJplus.setup_mpi_environment()

                # Test barrier
                QGYBJplus.mpi_barrier(mpi_config)

                # Test reduce
                local_val = Float64(rank + 1)
                global_sum = QGYBJplus.mpi_reduce_sum(local_val, mpi_config)
                expected_sum = nprocs * (nprocs + 1) / 2

                @test global_sum ≈ expected_sum

                if rank == 0
                    println("  ✓ MPI communication (barrier, reduce) working")
                end
            end

            @testset "Transpose Operations (Two-Step)" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
                grid = QGYBJplus.init_mpi_grid(params, mpi_config)
                state = QGYBJplus.init_mpi_state(grid, mpi_config)

                # Initialize with deterministic data for roundtrip test
                QGYBJplus.init_mpi_random_field!(state.psi, grid, 1.0, 123)

                # Allocate z-pencil array
                psi_z = QGYBJplus.allocate_z_pencil(grid, ComplexF64)

                # Test transpose to z-pencil (xy→z)
                # This uses the two-step transpose: xy(2,3) → xz(1,3) → z(1,2)
                QGYBJplus.transpose_to_z_pencil!(psi_z, state.psi, grid)

                # Verify z is fully local after transpose
                psi_z_arr = parent(psi_z)
                @test size(psi_z_arr, 3) == grid.nz  # z should be fully local

                # Test roundtrip: transpose back to xy-pencil (z→xy)
                # This uses the two-step transpose: z(1,2) → xz(1,3) → xy(2,3)
                psi_roundtrip = similar(state.psi)
                QGYBJplus.transpose_to_xy_pencil!(psi_roundtrip, psi_z, grid)

                # Check roundtrip accuracy
                parent_psi = parent(state.psi)
                parent_roundtrip = parent(psi_roundtrip)
                local_error = maximum(abs.(parent_psi .- parent_roundtrip))

                # Reduce max error across all ranks
                global_error = MPI.Allreduce(local_error, MPI.MAX, comm)

                @test global_error < 1e-12

                if rank == 0
                    println("  ✓ Two-step transpose roundtrip successful (error: $global_error)")
                end

                # Test that z-pencil has correct decomposition properties
                # After transpose to z-pencil, we should be able to do vertical operations
                # because z is fully local
                nz = grid.nz
                for j in 1:size(psi_z_arr, 2), i in 1:size(psi_z_arr, 1)
                    # Verify we can access all z levels (this would fail if z wasn't local)
                    for k in 1:nz
                        _ = psi_z_arr[i, j, k]
                    end
                end

                if rank == 0
                    println("  ✓ Z-pencil has z fully local (required for vertical ops)")
                end

                QGYBJplus.mpi_barrier(mpi_config)
            end

            @testset "Gather/Scatter" begin
                mpi_config = QGYBJplus.setup_mpi_environment()
                params = QGYBJplus.default_params(nx=64, ny=64, nz=32, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
                grid = QGYBJplus.init_mpi_grid(params, mpi_config)
                state = QGYBJplus.init_mpi_state(grid, mpi_config)

                # Initialize with deterministic values
                QGYBJplus.init_mpi_random_field!(state.psi, grid, 1.0, 42)

                # Gather to root
                gathered = QGYBJplus.gather_to_root(state.psi, grid, mpi_config)

                if mpi_config.is_root
                    @test gathered !== nothing
                    @test size(gathered) == (64, 64, 32)
                    println("  ✓ Gather to root successful")
                else
                    @test gathered === nothing
                end

                QGYBJplus.mpi_barrier(mpi_config)

                # Test scatter_from_root
                # Create a global array on root with known values
                if mpi_config.is_root
                    global_arr = zeros(ComplexF64, 64, 64, 32)
                    for k in 1:32, j in 1:64, i in 1:64
                        global_arr[i, j, k] = Complex(i + j*100 + k*10000, 0.0)
                    end
                else
                    global_arr = nothing
                end

                # Scatter to all processes
                scattered = QGYBJplus.scatter_from_root(global_arr, grid, mpi_config)

                # Verify each process received correct data
                local_range = QGYBJplus.get_local_range_xy(grid)
                parent_arr = parent(scattered)

                all_correct = true
                for k_local in axes(parent_arr, 3)
                    k_global = local_range[3][k_local]
                    for j_local in axes(parent_arr, 2)
                        j_global = local_range[2][j_local]
                        for i_local in axes(parent_arr, 1)
                            i_global = local_range[1][i_local]
                            expected = Complex(i_global + j_global*100 + k_global*10000, 0.0)
                            if parent_arr[i_local, j_local, k_local] != expected
                                all_correct = false
                                break
                            end
                        end
                    end
                end

                @test all_correct

                # Verify using global reduction
                local_correct = all_correct ? 1 : 0
                global_correct = MPI.Allreduce(local_correct, MPI.MIN, comm)
                @test global_correct == 1

                if rank == 0
                    println("  ✓ Scatter from root successful")
                end

                QGYBJplus.mpi_barrier(mpi_config)
            end
        end

        QGYBJplus.mpi_barrier(QGYBJplus.setup_mpi_environment())

        if rank == 0
            println()
            println("All MPI tests passed!")
        end
    finally
        MPI.Finalize()
    end
end

# Run appropriate tests based on mode
if SERIAL_MODE
    run_serial_tests()
else
    run_mpi_tests()
end
