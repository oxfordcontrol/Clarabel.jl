using Test, Clarabel


@testset "Settings and Solver constructors" begin

    for FloatT in [Float32,Float64,BigFloat]

        @testset "Settings and Solver (T = $(FloatT))" begin

            @test begin 
                #no settings 
                Clarabel.Solver{FloatT}()

                #settings via struct 
                Clarabel.Solver{FloatT}(Clarabel.Settings{FloatT}())

                #settings via dict 
                Clarabel.Solver{FloatT}(Dict("verbose" => true))

                #settings via parameter list  
                Clarabel.Settings{FloatT}(verbose = true)

                #settings via dict 
                Clarabel.Settings{FloatT}(Dict("verbose" => true))

                true
            end

        end
    end

    @testset "Settings and Solver constructors (default types)" begin

        @test begin
            #no settings 
            Clarabel.Solver()

            #settings via struct 
            Clarabel.Solver(Clarabel.Settings())

            #settings via dict 
            Clarabel.Solver(Dict("verbose" => true))

            #settings via parameter list  
            Clarabel.Settings(verbose = true)

            #settings via dict 
            Clarabel.Settings(Dict("verbose" => true))

            #fail on mixed types (assumes 64 bit default)
            @test_throws MethodError Clarabel.Solver{Float32}(Clarabel.Settings{Float64}())
            @test_throws MethodError Clarabel.Solver(Clarabel.Settings{Float32}())  
            @test_throws MethodError Clarabel.Solver{Float32}(Clarabel.Settings())  

            true
        end
    end 

end
nothing
