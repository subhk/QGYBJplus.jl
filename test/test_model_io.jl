using Test
using NCDatasets
using QGYBJplus

@testset "Simulation-owned NetCDF output and restart" begin
    mktempdir() do output_dir
        grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
        N²_function = z -> 4.0 + 0.2z
        stratification = AnalyticalProfile{Float64, typeof(N²_function)}(
            N²_function, true)
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=2.0),
            stratification=stratification,
            closure=HorizontalHyperdiffusivity(
                flow=0, flow2=0, waves=0, waves2=0),
            flow=FixedFlow(),
            formulation=PassiveWave(),
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
        model.fields.q[2, 2, 1] = 0.25 - 0.1im
        model.fields.B[2, 2, 1] = 0.5 + 0.2im

        simulation = Simulation(
            model;
            Δt=0.1,
            stop_iteration=2,
            output=NetCDFOutput(
                path=output_dir,
                schedule=IterationInterval(1),
                fields=(:ψ, :waves),
                velocities=true,
            ),
            verbose=false,
        )

        try
            run!(simulation)
            files = sort(filter(name -> endswith(name, ".nc"), readdir(output_dir)))
            @test files == ["state0001.nc", "state0002.nc", "state0003.nc"]

            last_file = joinpath(output_dir, last(files))
            NCDataset(last_file, "r") do ds
                @test ds.attrib["iteration"] == 2
                @test ds["time"][1] ≈ 0.2
                @test ds["N2"][:] ≈ model.runtime.coefficients.N²
                @test !all(==(first(ds["N2"][:])), ds["N2"][:])
                @test ds["a_ell"][:] ≈
                      model.physics.coriolis.f^2 ./ ds["N2"][:]
                @test all(name -> haskey(ds, name),
                    ("psi", "LAr", "LAi", "q_real", "q_imag",
                     "B_real", "B_imag", "u", "v", "w"))
            end

            q_saved = copy(parent(model.fields.q))
            B_saved = copy(parent(model.fields.B))
            fill!(parent(model.fields.q), 0)
            fill!(parent(model.fields.B), 0)
            restore!(model, last_file)
            @test parent(model.fields.q) ≈ q_saved
            @test parent(model.fields.B) ≈ B_saved
            @test simulation.output_manager.closed

            expected_flow_energy = flow_kinetic_energy_global(
                model.fields.u, model.fields.v, model.runtime.mpi)
            expected_wave_energy = wave_energy_global(
                model.fields.B, model.fields.A, model.runtime.mpi)
            @test flow_kinetic_energy(model) ≈ expected_flow_energy
            model_wave_energy = wave_energy(model)
            @test first(model_wave_energy) ≈ first(expected_wave_energy)
            @test last(model_wave_energy) ≈ last(expected_wave_energy)

            selected_path = joinpath(output_dir, "selected")
            selected_simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=3,
                output=NetCDFOutput(
                    path=selected_path,
                    schedule=IterationInterval(2),
                    fields=(:ψ,),
                ),
                verbose=false,
            )
            run!(selected_simulation)
            @test sort(readdir(selected_path)) ==
                  ["state0001.nc", "state0002.nc", "state0003.nc"]
            NCDataset(joinpath(selected_path, "state0003.nc"), "r") do ds
                @test ds.attrib["iteration"] == 3
                @test haskey(ds, "psi")
                @test haskey(ds, "B_real")
                @test !haskey(ds, "LAr")
                @test !haskey(ds, "u")
            end

            failure_path = joinpath(output_dir, "forced_failure")
            mkpath(joinpath(failure_path, "state0001.nc"))
            failing_simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=1,
                output=NetCDFOutput(path=failure_path),
                verbose=false,
            )
            @test_throws ErrorException run!(failing_simulation)
            @test failing_simulation.state == Failed
            @test failing_simulation.output_manager.closed
        finally
            finalize_simulation!(simulation)
        end
    end
end
