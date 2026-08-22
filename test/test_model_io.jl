using Test
using NCDatasets
using QGYBJplus
using Random
using FFTW

module EnergyExample
include(joinpath(@__DIR__, "..", "examples", "compute_energy.jl"))
end

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
                @test collect((ds.attrib["Lx"], ds.attrib["Ly"],
                               ds.attrib["Lz"])) ≈ collect(grid.extent)
                @test ds.attrib["wave_formulation"] == "PassiveWave"
                @test ds["N2"][:] ≈ model.runtime.coefficients.N²
                @test !all(==(first(ds["N2"][:])), ds["N2"][:])
                @test ds["a_ell"][:] ≈
                      model.physics.coriolis.f^2 ./ ds["N2"][:]
                @test all(name -> haskey(ds, name),
                    ("psi", "A_real", "A_imag", "LA_real", "LA_imag",
                     "q_hat_real", "q_hat_imag", "B_hat_real",
                     "B_hat_imag", "u", "v", "w"))

                A = ds["A_real"][:, :, :] .+
                    im .* ds["A_imag"][:, :, :]
                LA = ds["LA_real"][:, :, :] .+
                     im .* ds["LA_imag"][:, :, :]
                B_hat = ds["B_hat_real"][:, :, :] .+
                        im .* ds["B_hat_imag"][:, :, :]
                kh² = reshape(grid.kx .^ 2, :, 1, 1) .+
                     reshape(grid.ky .^ 2, 1, :, 1)
                forward = FFTW.plan_fft(A, (1, 2))
                @test forward * LA ≈
                      B_hat .+ 0.25 .* kh² .* (forward * A)

                for old_name in ("LAr", "LAi", "Ar", "Ai",
                                 "q_real", "q_imag", "B_real", "B_imag")
                    @test !haskey(ds, old_name)
                end
            end

            analysis_grid = EnergyExample.grid_from_snapshot(last_file)
            @test analysis_grid.size == grid.size
            @test collect(analysis_grid.extent) ≈ collect(grid.extent)
            @test analysis_grid.kx ≈ grid.kx
            @test analysis_grid.ky ≈ grid.ky

            energy_file = joinpath(output_dir, "energy0003.nc")
            EnergyExample.write_energy_snapshot(
                last_file, energy_file, analysis_grid)
            NCDataset(energy_file, "r") do ds
                @test size(ds["flow_KE"]) == grid.size
                @test size(ds["wave_KE"]) == grid.size
                @test ds["total_KE"][:, :, :] ≈
                      ds["flow_KE"][:, :, :] .+ ds["wave_KE"][:, :, :]
                @test all(isfinite, ds["total_KE"][:, :, :])
            end

            q_saved = copy(parent(model.fields.q))
            B_saved = copy(parent(model.fields.B))
            fill!(parent(model.fields.q), 0)
            fill!(parent(model.fields.B), 0)
            restore!(model, last_file)
            @test parent(model.fields.q) ≈ q_saved
            @test parent(model.fields.B) ≈ B_saved
            @test simulation.output_manager.closed

            expected_flow_energy = 0.5 * sum(
                abs2.(parent(model.fields.u)) .+
                abs2.(parent(model.fields.v)))
            expected_wave_energy = (
                sum(abs2, parent(model.fields.B)),
                sum(abs2, parent(model.fields.A)),
            )
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
                @test haskey(ds, "B_hat_real")
                @test !haskey(ds, "LA_real")
                @test !haskey(ds, "u")
            end

            timed_path = joinpath(output_dir, "timed")
            timed_simulation = Simulation(
                model;
                Δt=0.6,
                stop_time=3.0,
                output=NetCDFOutput(
                    path=timed_path,
                    schedule=TimeInterval(1.0),
                    fields=(:ψ,),
                ),
                verbose=false,
            )
            run!(timed_simulation)
            timed_files = sort(filter(
                name -> endswith(name, ".nc"), readdir(timed_path)))
            @test timed_files == [
                "state0001.nc", "state0002.nc", "state0003.nc", "state0004.nc"]
            timed_values = map(timed_files) do name
                NCDataset(joinpath(timed_path, name), "r") do dataset
                    dataset["time"][1]
                end
            end
            @test timed_values ≈ [0.0, 1.2, 2.4, 3.0]

            original_override_path = joinpath(output_dir, "override_original")
            effective_override_path = joinpath(output_dir, "override_effective")
            override_simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=2,
                output=NetCDFOutput(
                    path=original_override_path,
                    schedule=IterationInterval(2),
                    fields=(:ψ,),
                ),
                verbose=false,
            )
            run!(override_simulation;
                output_dir=effective_override_path,
                save_interval=0.15,
                save_psi=false,
                save_waves=true,
                save_velocities=true)
            @test !isdir(original_override_path)
            @test sort(readdir(effective_override_path)) ==
                  ["state0001.nc", "state0002.nc"]
            NCDataset(joinpath(effective_override_path, "state0002.nc"), "r") do ds
                @test !haskey(ds, "psi")
                @test haskey(ds, "LA_real")
                @test haskey(ds, "u")
            end

            diagnostic_path = joinpath(output_dir, "diagnostic_output")
            diagnostic_simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=2,
                output=false,
                diagnostics=EnergyDiagnosticsOutput(
                    path=diagnostic_path,
                    schedule=IterationInterval(1),
                ),
                verbose=false,
            )
            run!(diagnostic_simulation)
            @test diagnostic_simulation.diagnostics_manager.closed
            @test sort(readdir(diagnostic_path)) == [
                "mean_flow_KE.nc",
                "mean_flow_PE.nc",
                "total_energy.nc",
                "wave_CE.nc",
                "wave_KE.nc",
                "wave_PE.nc",
            ]
            NCDataset(joinpath(diagnostic_path, "total_energy.nc"), "r") do ds
                @test ds["time"][:] ≈ [0.0, 0.1, 0.2]
                @test all(name -> haskey(ds, name), (
                    "wave_KE", "wave_PE", "wave_CE",
                    "mean_flow_KE", "mean_flow_PE",
                    "total_wave_energy", "total_flow_energy",
                    "total_energy"))
                @test ds["total_wave_energy"][:] ≈
                      ds["wave_KE"][:] .+ ds["wave_PE"][:] .+
                      ds["wave_CE"][:]
                @test ds["total_flow_energy"][:] ≈
                      ds["mean_flow_KE"][:] .+ ds["mean_flow_PE"][:]
                @test all(isfinite, ds["total_energy"][:])
                @test first(ds["wave_KE"][:]) > 0
                @test ds["wave_KE"][:] ≈ fill(first(ds["wave_KE"][:]), 3)
            end

            for dispersion in (Dispersive(), NoDispersion())
                tag = string(nameof(typeof(dispersion)))
                ybj_model = QGYBJModel(
                    grid=grid,
                    coriolis=FPlane(f=2.0),
                    stratification=stratification,
                    closure=HorizontalHyperdiffusivity(
                        flow=0, flow2=0, waves=0, waves2=0),
                    flow=FixedFlow(),
                    feedback=NoFeedback(),
                    formulation=YBJ(),
                    linear=LinearDynamics(),
                    no_dispersion=dispersion,
                    topology=(1, 1),
                    verbose=false,
                )
                try
                    B = zeros(ComplexF64, 4, 8, 8)
                    B[:, 2, 1] .= (1, -1, 1, -1)
                    set!(ybj_model;
                        B=FieldArray(B; space=:spectral), verbose=false)
                    @test sum(abs2, parent(ybj_model.fields.A)) > 0
                    @test sum(abs2, parent(ybj_model.fields.C)) > 0

                    fill!(parent(ybj_model.fields.A), 0)
                    fill!(parent(ybj_model.fields.C), 0)
                    envelope_energy, amplitude_energy = wave_energy(ybj_model)
                    @test envelope_energy > 0
                    @test amplitude_energy > 0

                    state_path = joinpath(output_dir, "normal_ybj_state_$tag")
                    state_simulation = Simulation(
                        ybj_model;
                        Δt=0.1,
                        stop_iteration=1,
                        output=NetCDFOutput(
                            path=state_path,
                            schedule=IterationInterval(1),
                            fields=(:waves,),
                        ),
                        diagnostics=false,
                        verbose=false,
                    )
                    run!(state_simulation)
                    initial_state = joinpath(state_path, "state0001.nc")
                    NCDataset(initial_state, "r") do ds
                        LA = ds["LA_real"][:, :, :] .+
                             im .* ds["LA_imag"][:, :, :]
                        B_hat = ds["B_hat_real"][:, :, :] .+
                                im .* ds["B_hat_imag"][:, :, :]
                        @test sum(abs2, LA) > 0
                        @test fft(LA, (1, 2)) ≈ B_hat
                    end

                    fill!(parent(ybj_model.fields.A), 0)
                    fill!(parent(ybj_model.fields.C), 0)
                    restore!(ybj_model, initial_state)
                    @test sum(abs2, parent(ybj_model.fields.A)) > 0
                    @test sum(abs2, parent(ybj_model.fields.C)) > 0

                    fill!(parent(ybj_model.fields.A), 0)
                    fill!(parent(ybj_model.fields.C), 0)
                    ybj_diagnostic_path = joinpath(
                        output_dir, "normal_ybj_diagnostic_$tag")
                    diagnostic_simulation = Simulation(
                        ybj_model;
                        Δt=0.1,
                        stop_iteration=1,
                        output=false,
                        diagnostics=EnergyDiagnosticsOutput(
                            path=ybj_diagnostic_path,
                            schedule=IterationInterval(1),
                        ),
                        verbose=false,
                    )
                    run!(diagnostic_simulation)
                    NCDataset(joinpath(
                        ybj_diagnostic_path, "total_energy.nc"), "r") do ds
                        @test ds["time"][:] ≈ [0.0, 0.1]
                        @test all(>(0), ds["wave_KE"][:])
                        @test all(isfinite, ds["total_energy"][:])
                    end
                finally
                    finalize_model!(ybj_model)
                end
            end

            feedback_model = QGYBJModel(
                grid=grid,
                coriolis=FPlane(f=2.0),
                stratification=stratification,
                closure=HorizontalHyperdiffusivity(
                    flow=0, flow2=0, waves=0, waves2=0),
                flow=EvolvingFlow(),
                feedback=WaveMeanFeedback(),
                formulation=YBJ(),
                linear=NonlinearDynamics(),
                no_dispersion=Dispersive(),
                topology=(1, 1),
                verbose=false,
            )
            try
                rng = MersenneTwister(41)
                parent(feedback_model.fields.q) .=
                    1e-2 .* randn(rng, ComplexF64,
                                   size(parent(feedback_model.fields.q)))
                parent(feedback_model.fields.q)[:, 1, 1] .= 0
                parent(feedback_model.fields.B) .=
                    1e-2 .* randn(rng, ComplexF64,
                                   size(parent(feedback_model.fields.B)))
                invert_q_to_psi!(feedback_model)
                baseline_psi = copy(parent(feedback_model.fields.psi))
                prognostic_q = copy(parent(feedback_model.fields.q))

                expected = QGYBJplus.copy_fields(feedback_model.fields)
                context = QGYBJplus._operator_context(feedback_model)
                options = QGYBJplus.ETDModelOptions(
                    feedback_model.physics, feedback_model.numerics)
                QGYBJplus._finalize_etdrk2_state!(
                    expected, context.grid, options, context.plans,
                    context.a, context.mask;
                    workspace=context.workspace,
                    N2_profile=context.N2,
                )
                @test maximum(abs,
                    parent(expected.psi) .- baseline_psi) > 1e-10

                QGYBJplus._refresh_wave_diagnostics!(feedback_model)
                @test parent(feedback_model.fields.psi) ≈ parent(expected.psi)
                @test parent(feedback_model.fields.A) ≈ parent(expected.A)
                @test parent(feedback_model.fields.C) ≈ parent(expected.C)
                @test parent(feedback_model.fields.q) == prognostic_q
            finally
                finalize_model!(feedback_model)
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
