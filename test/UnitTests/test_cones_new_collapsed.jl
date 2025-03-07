using Test, Clarabel

@testset "cones_new_collapsed" begin


    @testset "test_cones_new_collapsed_no_changes" begin

        cones = [
            Clarabel.NonnegativeConeT(3),
            Clarabel.SecondOrderConeT(4),
            Clarabel.ExponentialConeT(),
        ];

        expected = deepcopy(cones)
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

    @testset "test_cones_new_collapsed_consolidate_nonnegative" begin

        cones = [
            Clarabel.NonnegativeConeT(3),
            Clarabel.NonnegativeConeT(2),
            Clarabel.SecondOrderConeT(4),
        ];

        expected = [
            Clarabel.NonnegativeConeT(5),
            Clarabel.SecondOrderConeT(4),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)

    end

    @testset "test_cones_new_collapsed_remove_empty" begin 

        cones = [
            Clarabel.NonnegativeConeT(3),
            Clarabel.ZeroConeT(0),
            Clarabel.SecondOrderConeT(4),
            Clarabel.NonnegativeConeT(0),
        ];

        expected = [
            Clarabel.NonnegativeConeT(3),
            Clarabel.SecondOrderConeT(4),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

    @testset "test_cones_new_collapsed_soc_to_nonnegative" begin

        cones = Clarabel.SupportedCone[
            Clarabel.SecondOrderConeT(1),
            Clarabel.SecondOrderConeT(4),
        ];

        expected = [
            Clarabel.NonnegativeConeT(1),
            Clarabel.SecondOrderConeT(4),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

    @testset "test_cones_new_collapsed_psd_to_nonnegative" begin
        cones = [
            Clarabel.PSDTriangleConeT(1),
            Clarabel.SecondOrderConeT(4),
        ];

        expected = [
            Clarabel.NonnegativeConeT(1),
            Clarabel.SecondOrderConeT(4),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

    @testset "test_cones_new_collapsed_mixed" begin
        cones = [
            Clarabel.SecondOrderConeT(1),
            Clarabel.NonnegativeConeT(3),
            Clarabel.NonnegativeConeT(2),
            Clarabel.ExponentialConeT(),
            Clarabel.NonnegativeConeT(0),
            Clarabel.SecondOrderConeT(1),
        ];

        expected = [
            Clarabel.NonnegativeConeT(6),
            Clarabel.ExponentialConeT(),
            Clarabel.NonnegativeConeT(1),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

     @testset "test_cones_new_collapsed_mixed_sdp" begin
        cones = [
            Clarabel.NonnegativeConeT(3),
            Clarabel.NonnegativeConeT(2),
            Clarabel.ZeroConeT(0),
            Clarabel.SecondOrderConeT(1),
            Clarabel.PSDTriangleConeT(1),
            Clarabel.SecondOrderConeT(4),
            Clarabel.NonnegativeConeT(0),
        ];

        expected = [
            Clarabel.NonnegativeConeT(7),
            Clarabel.SecondOrderConeT(4),
        ];
        result = Clarabel.cones_new_collapsed(cones)

        @test isequal(result, expected)
    end

end 