using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function sdp_chordal_data(Type::Type{T}) where {T <: AbstractFloat}


    P = spzeros(T,8,8)
    c = T[-1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]

    m = 28
    n = 8
    colptr = 1 .+ [0, 1, 4, 5, 8, 9, 10, 13, 16]
    rowval = 1 .+ [24, 7, 10, 22, 8, 12, 15, 25, 9, 13, 18, 21, 26, 0, 23, 27]
    nzval  = T[
        -1.0,
        -sqrt(2.),
        -1.0,
        -1.0,
        -sqrt(2.),
        -sqrt(2.),
        -1.0,
        -1.0,
        -sqrt(2.),
        -sqrt(2.),
        -sqrt(2.),
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ] 
    A = SparseMatrixCSC(m,n,colptr,rowval,nzval);

    b= T[
        0.0,
        3.0,
        2. * sqrt(2.),
        2.0,
        sqrt(2.),
        sqrt(2.),
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    
    cones = Clarabel.SupportedCone[
        Clarabel.NonnegativeConeT(1),
        Clarabel.PSDTriangleConeT(6),
        Clarabel.PowerConeT(0.3333333333333333),
        Clarabel.PowerConeT(0.5),
        ]

    return (P,c,A,b,cones)

end

@testset "Chordal SDP Tests" begin

    for FloatT in UnitTestFloats

        @testset "Chordal SDP Tests (T = $(FloatT))" begin


            P,c,A,b,cones = sdp_chordal_data(FloatT)
            settings = Clarabel.Settings{FloatT}()
            settings.chordal_decomposition_enable = true

            for compact in [false,true]
                for complete_dual in [false,true]
                    for merge_method in [:clique_graph,:parent_child,:none]

                        settings.chordal_decomposition_compact = compact
                        settings.chordal_decomposition_complete_dual = complete_dual
                        settings.chordal_decomposition_merge_method = merge_method


                        solver   = Clarabel.Solver(P,c,A,b,cones,settings)
                        Clarabel.solve!(solver)

                        @test solver.solution.status == Clarabel.SOLVED

                    end
                end
            end
        end

    end # UnitTestFloats

end #"Basic SDP Tests"

nothing




