#=
================================================================================
            MPI Extension Test Suite for QGYBJ.jl
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

if SERIAL_MODE
    println("=" ^ 60)
    println("QGYBJ.jl Serial Mode Test")
    println("=" ^ 60)
    println()

    @testset "Serial Mode Tests" begin
        @testset "Module Loading" begin
            @test_nowarn using QGYBJ
            println("  ✓ QGYBJ module loaded")
        end

        @testset "Basic Types" begin
            using QGYBJ: QGParams, Grid, State

            params = QGYBJ.default_params(nx=32, ny=32, nz=16)
            @test params.nx == 32
            @test params.ny == 32
            @test params.nz == 16
            println("  ✓ QGParams created")

            grid = QGYBJ.init_grid(params)
            @test grid.nx == 32
            @test grid.decomp === nothing  # No MPI decomposition
            println("  ✓ Grid created (serial mode)")

            state = QGYBJ.init_state(grid)
            @test size(state.psi) == (32, 32, 16)
            println("  ✓ State created")
        end

        @testset "FFT Transforms (Serial)" begin
            using QGYBJ: Plans, plan_transforms!, fft_forward!, fft_backward!

            params = QGYBJ.default_params(nx=32, ny=32, nz=16)
            grid = QGYBJ.init_grid(params)
            plans = plan_transforms!(grid)

            @test plans.backend == :fftw
            println("  ✓ FFTW plans created")

            # Test FFT roundtrip
            src = randn(ComplexF64, 32, 32, 16)
            dst = similar(src)
            dst2 = similar(src)

            fft_forward!(dst, src, plans)
            fft_backward!(dst2, dst, plans)

            # FFTW ifft is unnormalized, so divide by n
            dst2 ./= (32 * 32)

            @test isapprox(src, dst2, rtol=1e-10)
            println("  ✓ FFT roundtrip successful")
        end

        @testset "MPI Stubs (Without MPI)" begin
            # These should throw informative errors
            @test_throws ErrorException QGYBJ.setup_mpi_environment()
            @test_throws ErrorException QGYBJ.init_mpi_grid(nothing, nothing)
            println("  ✓ MPI stubs throw appropriate errors")
        end
    end

    println()
    println("All serial tests passed!")

else
    # MPI Mode
    println("Loading MPI packages...")

    using MPI
    using PencilArrays
    using PencilFFTs
    using QGYBJ

    function main()
        MPI.Init()

        try
            run_mpi_tests()
        finally
            MPI.Finalize()
        end
    end

    function run_mpi_tests()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)

        if rank == 0
            println("=" ^ 60)
            println("QGYBJ.jl MPI Extension Test Suite")
            println("=" ^ 60)
            println("Processes: $nprocs")
            println()
        end

        @testset "MPI Extension Tests" begin
            @testset "MPI Environment Setup" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                @test mpi_config.nprocs == nprocs
                @test mpi_config.rank == rank
                @test mpi_config.is_root == (rank == 0)
                if rank == 0
                    println("  ✓ MPI environment initialized")
                end
            end

            @testset "Parallel Grid" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                params = QGYBJ.default_params(nx=64, ny=64, nz=32)

                grid = QGYBJ.init_mpi_grid(params, mpi_config)
                @test grid.nx == 64
                @test grid.ny == 64
                @test grid.nz == 32
                @test grid.decomp !== nothing  # Has MPI decomposition

                if rank == 0
                    println("  ✓ Parallel grid created with decomposition")
                end
            end

            @testset "Parallel State" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                params = QGYBJ.default_params(nx=64, ny=64, nz=32)
                grid = QGYBJ.init_mpi_grid(params, mpi_config)

                state = QGYBJ.init_mpi_state(grid, mpi_config)

                # Check that arrays are PencilArrays
                @test typeof(state.psi) <: PencilArray
                @test typeof(state.B) <: PencilArray
                @test typeof(state.u) <: PencilArray

                if rank == 0
                    println("  ✓ Parallel state created with PencilArrays")
                end
            end

            @testset "Parallel FFT Plans" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                params = QGYBJ.default_params(nx=64, ny=64, nz=32)
                grid = QGYBJ.init_mpi_grid(params, mpi_config)

                plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

                @test plans isa QGYBJ.QGYBJMPIExt.MPIPlans
                @test plans.forward !== nothing
                @test plans.backward !== nothing

                if rank == 0
                    println("  ✓ PencilFFT plans created")
                end
            end

            @testset "Parallel FFT Execution" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                params = QGYBJ.default_params(nx=64, ny=64, nz=32)
                grid = QGYBJ.init_mpi_grid(params, mpi_config)
                state = QGYBJ.init_mpi_state(grid, mpi_config)
                plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

                # Initialize with random data
                QGYBJ.init_mpi_random_field!(state.psi, grid, 1.0, 0)

                # Perform FFT roundtrip
                work_k = similar(state.psi)
                work = similar(state.psi)

                QGYBJ.fft_forward!(work_k, state.psi, plans)
                QGYBJ.fft_backward!(work, work_k, plans)

                # Normalize (PencilFFTs ifft is unnormalized)
                work ./= (grid.nx * grid.ny)

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
                mpi_config = QGYBJ.setup_mpi_environment()

                # Test barrier
                QGYBJ.mpi_barrier(mpi_config)

                # Test reduce
                local_val = Float64(rank + 1)
                global_sum = QGYBJ.mpi_reduce_sum(local_val, mpi_config)
                expected_sum = nprocs * (nprocs + 1) / 2

                @test global_sum ≈ expected_sum

                if rank == 0
                    println("  ✓ MPI communication (barrier, reduce) working")
                end
            end

            @testset "Gather/Scatter" begin
                mpi_config = QGYBJ.setup_mpi_environment()
                params = QGYBJ.default_params(nx=64, ny=64, nz=32)
                grid = QGYBJ.init_mpi_grid(params, mpi_config)
                state = QGYBJ.init_mpi_state(grid, mpi_config)

                # Initialize with deterministic values
                QGYBJ.init_mpi_random_field!(state.psi, grid, 1.0, 42)

                # Gather to root
                gathered = QGYBJ.gather_to_root(state.psi, grid, mpi_config)

                if mpi_config.is_root
                    @test gathered !== nothing
                    @test size(gathered) == (64, 64, 32)
                    println("  ✓ Gather to root successful")
                else
                    @test gathered === nothing
                end

                QGYBJ.mpi_barrier(mpi_config)
            end
        end

        QGYBJ.mpi_barrier(QGYBJ.setup_mpi_environment())

        if rank == 0
            println()
            println("All MPI tests passed!")
        end
    end

    main()
end
