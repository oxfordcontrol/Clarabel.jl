using MathOptInterface

function __precompile_printfcns()

    # Verbose printing of SnoopPrecompile examples is disabled,  
    # so force precompile of solver print statements here by 
    # providing necessary signatures.   
    
    # NB: It is not sufficient to wrap the SnoopPrecompile 
    # block inside a redirect_stdout(..), since then all of 
    # the println and @printf calls get compiled with signatures 
    # like println(io::Base.DevNull,...)

    stdoutT = typeof(stdout)

    precompile(Clarabel.print_banner, (Bool,))
    precompile(Clarabel.print_settings, 
        (stdoutT, Clarabel.Settings{Float64},)
    )

    precompile(Clarabel.info_print_configuration,
        (
            stdoutT, 
            Clarabel.DefaultInfo{Float64},
            Clarabel.Settings{Float64},
            Clarabel.DefaultProblemData{Float64},
            Clarabel.CompositeCone{Float64},
         )
    )

    for fcn in (Clarabel.info_print_status_header, 
                Clarabel.info_print_status, 
                Clarabel.info_print_footer)
        precompile(fcn,
            (
                stdoutT,
                Clarabel.DefaultInfo{Float64},
                Clarabel.Settings{Float64},
            )
        )   
    end

end 

function __precompile_native()
    cones = [
        Clarabel.NonnegativeConeT(1),
        Clarabel.ZeroConeT(1),
        Clarabel.SecondOrderConeT(2),
        Clarabel.ExponentialConeT(),
        Clarabel.PowerConeT(0.5),
        Clarabel.GenPowerConeT([0.5;0.5],1)
        ];
    nvars = sum(Clarabel.nvars.(cones))
    P = A = sparse(I(nvars)*1.)
    b = c = ones(nvars)
    settings = Clarabel.Settings(max_iter = 1)
    solver   = Clarabel.Solver(P,c,A,b,cones,settings)
    Clarabel.solve!(solver);

    #separate initialization one for PSD cones as we don't support PSD and the nonsymmetric cones simultaneously at present
    cones = [
        Clarabel.PSDTriangleConeT(1)  
        ];
    nvars = sum(Clarabel.nvars.(cones))
    P = A = sparse(I(nvars)*1.)
    b = c = ones(nvars)
    settings = Clarabel.Settings(max_iter = 1)
    solver   = Clarabel.Solver(P,c,A,b,cones,settings)
    Clarabel.solve!(solver);
end

function __precompile_moi()

    MOI = MathOptInterface

    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(Clarabel.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), false) 
    MOI.set(model,MOI.RawOptimizerAttribute("max_iter"),1)

    # variables 
    x = MOI.add_variables(model, 3)
    y, _ = MOI.add_constrained_variables(model, MOI.Nonnegatives(2))
    z = MOI.add_variable(model)
    MOI.supports(model, MOI.VariableName(), typeof(x[1]))
    MOI.set(model, MOI.VariableName(), x[1], "x1")
    f = 1.0*x[1] + x[2] + x[3]

    # constraints 
    MOI.add_constraint(model, x, MOI.Nonnegatives(3))
    MOI.add_constraint(model, x, MOI.Zeros(3))
    MOI.add_constraint(model, x, MOI.SecondOrderCone(3))
    MOI.add_constraint(model, x, MOI.ExponentialCone())
    MOI.add_constraint(model, x, MOI.PowerCone(0.5))
    MOI.add_constraint(model, x, Clarabel.MOI.GenPowerCone([0.5;0.5],1))     #Support for GeneralizedPowerCone under MOI

    for (i,C) in enumerate((MOI.GreaterThan,MOI.LessThan,MOI.EqualTo))
        for F in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64})
            MOI.supports_constraint(model, F, C)
        end
        MOI.add_constraint(model, x[i], C(0.0))
        MOI.add_constraint(model, f, C(0.0))
    end

    c = MOI.add_constraint(model, f, MOI.GreaterThan(0.0))
    MOI.supports(model, MOI.ConstraintName(), typeof(c))
    MOI.set(model, MOI.ConstraintName(), c, "c")
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
    end

    # objectives 
    f = f + 1.0*x[1]*x[1] + x[2];
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.supports(model, MOI.ObjectiveFunction{typeof(f)}())
    MOI.set(model, MOI.ObjectiveFunction{typeof(x[1])}(), x[1])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    
    # solve and output
    MOI.optimize!(model)
    MOI.get(model, MOI.TerminationStatus())
    MOI.get(model, MOI.PrimalStatus())
    MOI.get(model, MOI.DualStatus())
    MOI.get(model, MOI.VariablePrimal(), x);

    #separate initialization one for PSD cones as we don't support PSD and the nonsymmetric cones simultaneously at present
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(Clarabel.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), false) 
    MOI.set(model,MOI.RawOptimizerAttribute("max_iter"),1)

    # variables 
    x = MOI.add_variables(model, 3)
    y, _ = MOI.add_constrained_variables(model, MOI.Nonnegatives(2))
    z = MOI.add_variable(model)
    MOI.supports(model, MOI.VariableName(), typeof(x[1]))
    MOI.set(model, MOI.VariableName(), x[1], "x1")
    f = 1.0*x[1] + x[2] + x[3]

    # constraints 
    MOI.add_constraint(model, x, MOI.PositiveSemidefiniteConeTriangle(2))

    for (i,C) in enumerate((MOI.GreaterThan,MOI.LessThan,MOI.EqualTo))
        for F in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64})
            MOI.supports_constraint(model, F, C)
        end
        MOI.add_constraint(model, x[i], C(0.0))
        MOI.add_constraint(model, f, C(0.0))
    end

    c = MOI.add_constraint(model, f, MOI.GreaterThan(0.0))
    MOI.supports(model, MOI.ConstraintName(), typeof(c))
    MOI.set(model, MOI.ConstraintName(), c, "c")
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
    end

    # objectives 
    f = f + 1.0*x[1]*x[1] + x[2];
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.supports(model, MOI.ObjectiveFunction{typeof(f)}())
    MOI.set(model, MOI.ObjectiveFunction{typeof(x[1])}(), x[1])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    
    # solve and output
    MOI.optimize!(model)
    MOI.get(model, MOI.TerminationStatus())
    MOI.get(model, MOI.PrimalStatus())
    MOI.get(model, MOI.DualStatus())
    MOI.get(model, MOI.VariablePrimal(), x);
end

