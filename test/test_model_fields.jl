using Test
using QGYBJplus

@testset "ModelFields allocation and copying" begin
    required_names = (:ModelFields, :allocate_fields, :copy_fields)
    for name in required_names
        @test isdefined(QGYBJplus, name)
    end

    @test !isdefined(QGYBJplus, :State)
    @test !isdefined(QGYBJplus, :init_state)
    @test !isdefined(QGYBJplus, :copy_state)

    if all(name -> isdefined(QGYBJplus, name), required_names)
        fields = ModelFields(Float64, (4, 8, 8))

        @test size(fields.q) == (4, 8, 8)
        @test size(fields.B) == (4, 8, 8)
        @test eltype(fields.q) == ComplexF64
        @test eltype(fields.u) == Float64
        @test all(iszero, fields.q)
        @test all(iszero, fields.u)

        copied = copy_fields(fields)
        copied.q[1] = 1 + 2im
        copied.u[1] = 3
        @test fields.q[1] == 0
        @test fields.u[1] == 0

        grid = RectilinearGrid(size = (8, 8, 4), extent = (2π, 2π, 1.0))
        grid_fields = allocate_fields(grid)
        @test grid_fields isa ModelFields
        @test size(grid_fields.q) == (4, 8, 8)
    end
end
