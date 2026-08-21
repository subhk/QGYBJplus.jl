using Test
using MPI
using PencilArrays
using QGYBJplus

@testset "ModelRuntime distributed metadata" begin
    grid = RectilinearGrid(size = (8, 8, 4),
                           x = (-π, π),
                           y = (-π, π),
                           z = (-1.0, 0.0))
    model = QGYBJModel(
        grid = grid,
        coriolis = FPlane(f = 1),
        stratification = ConstantStratification(N² = 1),
        closure = HorizontalHyperdiffusivity(flow = 0, flow2 = 0,
                                              waves = 0, waves2 = 0),
        verbose = false,
    )

    try
        runtime = model.runtime
        @test model.grid === grid
        @test !hasproperty(grid, :decomp)
        @test runtime.decomposition.global_dims == (4, 8, 8)
        @test runtime.mpi.nprocs == MPI.Comm_size(MPI.COMM_WORLD)
        @test grid.x == collect(range(-π, step=2π / 8, length=8))
        @test grid.y == collect(range(-π, step=2π / 8, length=8))
        @test grid.kx == runtime.computational_grid.kx
        @test grid.ky == runtime.computational_grid.ky

        runtime_methods = (
            applicable(get_local_range, runtime),
            applicable(get_local_range, model),
            applicable(QGYBJplus.get_local_range_physical, runtime),
            applicable(QGYBJplus.get_local_range_spectral, runtime),
            applicable(local_to_global, 1, 1, runtime),
            applicable(get_kh2, 1, 1, 1, model.fields.q, model),
        )
        @test all(runtime_methods)

        if all(runtime_methods)
            physical_range = QGYBJplus.get_local_range_physical(runtime)
            spectral_range = QGYBJplus.get_local_range_spectral(runtime)
            @test get_local_range(runtime) == spectral_range
            @test get_local_range(model) == spectral_range
            @test physical_range == PencilArrays.range_local(runtime.plans.input_pencil)
            @test spectral_range == PencilArrays.range_local(runtime.plans.output_pencil)
            @test local_to_global(1, 1, runtime) == first(spectral_range[1])

            for k_local in axes(parent(model.fields.q), 1),
                i_local in axes(parent(model.fields.q), 2),
                j_local in axes(parent(model.fields.q), 3)

                i_global = local_to_global(i_local, 2, model.fields.q)
                j_global = local_to_global(j_local, 3, model.fields.q)
                @test get_kh2(i_local, j_local, k_local,
                              model.fields.q, model) == grid.kh2[i_global, j_global]
            end
        end

        physical = QGYBJplus.allocate_fft_backward_dst(model.fields.q,
                                                        runtime.plans)
        physical_values = parent(physical)
        physical_range = PencilArrays.range_local(runtime.plans.input_pencil)
        for k_local in axes(physical_values, 1),
            i_local in axes(physical_values, 2),
            j_local in axes(physical_values, 3)

            k_global = physical_range[1][k_local]
            i_global = physical_range[2][i_local]
            j_global = physical_range[3][j_local]
            physical_values[k_local, i_local, j_local] =
                complex(k_global + 2i_global, -j_global)
        end
        original = copy(physical_values)
        recovered = similar(physical)

        transform_methods = (
            applicable(fft_forward!, model.fields.q, physical, runtime),
            applicable(fft_backward!, recovered, model.fields.q, runtime),
        )
        @test all(transform_methods)
        if all(transform_methods)
            fft_forward!(model.fields.q, physical, runtime)
            fft_backward!(recovered, model.fields.q, runtime)
            local_error = maximum(abs, parent(recovered) .- original)
            global_error = MPI.Allreduce(local_error, MPI.MAX, runtime.mpi.comm)
            @test global_error < 1e-12
        end
    finally
        finalize_model!(model)
    end
end
