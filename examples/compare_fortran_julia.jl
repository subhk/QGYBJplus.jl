"""
compare_fortran_julia.jl

Minimal harness to compare Julia QGYBJ operators with a Fortran output file.

Usage (example):
  julia --project=. examples/compare_fortran_julia.jl \
      --nc fortran_output.nc \
      --psi_var psi \
      --B_real_var BR \
      --B_imag_var BI \
      [--A_real_var AR --A_imag_var AI]

Notes:
- Requires NCDatasets.jl in the environment.
- Expects complex spectral fields for ψ and B split into real/imag parts
  or a single complex-like representation depending on your file.
"""

using QGYBJ
using Printf

function main()
    # Parse simple CLI args
    ncfile = nothing
    psi_var = "psi"
    B_real_var = "BR"
    B_imag_var = "BI"
    A_real_var = nothing
    A_imag_var = nothing
    for (i,arg) in enumerate(ARGS)
        if arg == "--nc"; ncfile = ARGS[i+1]; end
        if arg == "--psi_var"; psi_var = ARGS[i+1]; end
        if arg == "--B_real_var"; B_real_var = ARGS[i+1]; end
        if arg == "--B_imag_var"; B_imag_var = ARGS[i+1]; end
        if arg == "--A_real_var"; A_real_var = ARGS[i+1]; end
        if arg == "--A_imag_var"; A_imag_var = ARGS[i+1]; end
    end
    if ncfile === nothing
        println("Provide --nc <path> to a Fortran NetCDF output file.")
        return
    end

    try
        import NCDatasets
    catch
        println("NCDatasets.jl not available. Add it to the environment to run this harness.")
        return
    end

    println("Opening NetCDF: $ncfile")
    NCDatasets.Dataset(ncfile, "r") do ds
        # Try to infer grid sizes from variables
        # Expect dimensions like (nx, ny, nz)
        function read3(varname)
            haskey(ds, varname) || error("Variable $(varname) not found in $ncfile")
            Array(ds[varname][:,:,:])
        end

        # Read fields (assumed spectral)
        psiR = read3(psi_var*"_real") if haskey(ds, psi_var*"_real") else nothing
        psiI = read3(psi_var*"_imag") if haskey(ds, psi_var*"_imag") else nothing
        if psiR === nothing || psiI === nothing
            # try direct psi
            haskey(ds, psi_var) || error("Could not find $(psi_var)_real/_imag or $psi_var")
            psiR = Array(real(ds[psi_var][:,:,:]))
            psiI = Array(imag(ds[psi_var][:,:,:]))
        end

        BR = read3(B_real_var)
        BI = read3(B_imag_var)
        AR = (A_real_var === nothing || !haskey(ds, A_real_var)) ? nothing : read3(A_real_var)
        AI = (A_imag_var === nothing || !haskey(ds, A_imag_var)) ? nothing : read3(A_imag_var)

        nx, ny, nz = size(BR)
        @printf("Grid sizes: nx=%d ny=%d nz=%d\n", nx, ny, nz)

        # Set up Julia model
        par = default_params(nx=nx, ny=ny, nz=nz, stratification=:constant_N)
        G, S, plans, a = setup_model(; par)

        # Load Fortran spectral fields into Julia state
        @assert size(S.psi) == (nx,ny,nz)
        @assert size(S.B) == (nx,ny,nz)
        S.psi .= complex.(psiR, psiI)
        S.B   .= complex.(BR, BI)

        # Recover A via YBJ+ inversion and recompute ψ from q=∇²ψ-ψzz (not available here)
        invert_B_to_A!(S, G, par, a)

        # Basic diagnostic comparisons: check self-consistency of inversion
        # (No Fortran A provided -> we report norms for sanity)
        EB, EA = wave_energy(S.B, S.A)
        @printf("Energy-like norms: EB=%.6e EA=%.6e\n", EB, EA)

        # If Fortran A provided, compute error norms
        if AR !== nothing && AI !== nothing
            A_fortran = complex.(AR, AI)
            @assert size(A_fortran) == size(S.A)
            diff = S.A .- A_fortran
            l2 = sqrt(sum(abs2, diff))
            linf = maximum(abs, diff)
            @printf("A error norms: L2=%.6e Linf=%.6e\n", l2, linf)
        end

        # Optional: compute q^w for a consistency check
        L = dealias_mask(G)
        BRk = Complex.(real.(S.B))
        BIk = Complex.(imag.(S.B))
        qwk = similar(S.q)
        compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)
        @printf("qw spectral L2=%.6e\n", sqrt(sum(abs2, qwk)))

        println("Comparison harness completed. For more detailed parity checks,\n" *
                "export Fortran A/psi/q fields and extend this script to compare \n" *
                "field-wise L2/L∞ norms after synchronized steps.")
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
