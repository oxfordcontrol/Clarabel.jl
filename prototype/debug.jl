

function debug_cap_centering_param(iter,σ,μ)
    #don't allow solutions that are *too* optimal
    floor = max(1e-24,30^(-iter/3.))
    if(iter > 4 && μ*σ < floor)
        σ = floor/μ
    end

    return σ
end

function debug_print(s)

print("VARIABLES: ")
dump(s.variables,maxdepth=1)

print("RESIDUALS: ")
dump(s.residuals,maxdepth=1)

end

function debug_check_cones(solver)

cone_types = solver.data.cone_info.types

for i = 1:length(cone_types)
    type = cone_types[i]
    s = solver.variables.s.views[i]
    z = solver.variables.z.views[i]

    if(type == NonnegativeConeT)
        if(any(s .< 0))
            error("Negative s element in NNC")
        end
        if(any(z .< 0))
            error("Negative s element in NNC")
        end

    elseif(type == SecondOrderConeT)
        if(s[1] < 0)
            error("Negative s norm in SOC")
        end
        if(z[1] < 0)
            error("Negative z norm in SOC")
        end
    end
end

end

function IRsolve(M,b)

    s = sign.(diag(M))
    s = Int.(Array(s))
    s[s .== 0] .= 1

    #static regularization
    M = M + Diagonal(s).*1e-7

    F = qdldl(M; Dsigns = s)

    x = F\b

    for i = 1:1
        e = b - M*x
        dx = F\e
        x .+= dx
    end

    return x

end

function debug_print_variables(v)

print("VARIABLES: \n")
println("x=",v.x)
println("s=",v.s.vec)
println("z=",v.z.vec)
println("τ=",v.τ)
println("κ=",v.κ)
print("\n\n")

end

function debug_rescale(variables)

vars = variables
τ     = vars.τ
κ     = vars.κ
scale = max(τ,κ)

vars.x ./= scale
vars.z.vec ./= scale
vars.s.vec ./= scale
vars.τ /= scale
vars.κ /= scale

end

function debug_soc_step_length(αz,αs,z,s,dz,ds)

    return

    println("\n SOC DEBUG step length:")
    println("αz = ", αz, " | αs = ", αs)

    #find the prospective full step for s and z
    sfull = s + ds.*αs
    zfull = z + dz.*αz

    norms = norm(sfull[2:end])
    normz = norm(zfull[2:end])

    gaps = sfull[1] - norms
    gapz = zfull[1] - normz

    @printf("S gap = %f, norm sfull = %f\n", gaps, norms)
    @printf("Z gap = %f, norm zfull = %f\n\n", gapz, normz)
end


function debug_manual_kkt_solve45(variables,data,scalings,res,lhs_step)
    #make a manual check of the KKT affine solve (version CVXOPT eqn 45)

    #print all of the manual steps
    println("\n--- MANUAL STEP CHECK (1 step condensed) --- ")

    Z = zeros
    m = data.m
    n = data.n
    P = data.P
    A = data.A
    b = data.b
    c = data.c
    z = variables.z
    s = variables.s
    τ = variables.τ
    κ = variables.κ
    λ = scalings.λ
    l2 = deepcopy(λ)
    cones_circle_op!(scalings.cones, l2, λ, λ)

    L = debug_cones_matrixCircleOp(scalings.cones,λ)
    W = debug_cones_make_W(scalings.cones,s,z)
    WtW = debug_cones_make_WtW(scalings.cones,s,z)
    Wi = inv(Matrix(W))
    LW = L*W
    LWit = L*Wi'

    #form big linear system as in (36)
    M = [
     P         A'     c;
    -A       WtW      b;
    -c'       -b'     κ/τ;
    ]

    #println("M (1 step)")
    #display(Matrix(M))

    #for the right hand side
    dx  = res.rx
    dz  = res.rz
    dτ  = res.rτ
    ds  = l2.vec
    dκ  = κ*τ

    rhs = [dx;
           dz-s.vec;
           dτ - dκ/τ]

    println("COMPARE RHS      term = ",norm( (dz-s.vec) - (dz - W'*(L\ds)) ) )

    #solve as a big linear system
    lhs = M\rhs
    println("Full rhs")
    display(rhs)
    println("Full lhs")
    display(lhs)

    #peel the pieces of from the front
    Δx  = splice!(lhs,1:n)
    Δz  = splice!(lhs,1:m)
    Δτ  = splice!(lhs,1:1)[1]

    #back solve for \Delta s and \Delta kappa
    Δs1  = -W'*(L\ds + W*Δz)
    Δs2  = -s.vec - WtW*Δz   #better?
    Δs   = Δs2
    Δκ  = -(dκ + κ*Δτ)/τ
    println("COMPARE RECOVERY term = ",norm( Δs1 - Δs2) )


    #print all of the manual steps

    println("\n")
    println("    Δx = ", Δx)
    println("    Δz = ", Δz)
    println("    Δs = ", Δs)
    println("    Δτ = ", Δτ)
    println("    Δκ = ", Δκ)
    println("\n")


    #work out the result of a full steps
    xnew = variables.x     + Δx
    znew = variables.z.vec + Δz
    snew = variables.s.vec + Δs
    τnew = variables.τ     + Δτ
    κnew = variables.κ     + Δκ

    # #compute the residuals
    # println("    solve acc. row1 = ", (-data.A*Δx  - Δs + Δτ*data.b - res.rz))
    # println("    solve acc. row2 = ", (data.P*Δx + data.A'*Δz + Δτ*data.c - res.rx))
    # println("    solve acc. row3 = ", -Δκ - c'*Δx - b'*Δz - res.rτ)
    # println()
    #
    # #compute the residuals
    # println("    affine step acc. row1 = ", (-data.A*xnew  - snew + τnew*data.b))
    # println("    affine step acc. row2 = ", (data.P*xnew + A'*znew + τnew*data.c))
    # println("    affine step acc. row3 = ", -κnew - c'*xnew - b'*znew)
    # println("    affine step acc. row4 = ", -κnew - c'*xnew - b'*znew)
    # println()
    #
    #
    # #How does this compare to the compute step?
    # println("    step deviation x = ", Δx - lhs_step.x)
    # println("    step deviation z = ", Δz - lhs_step.z.vec)
    # println("    step deviation s = ", Δs - lhs_step.s.vec)
    # println("    step deviation τ = ", Δτ - lhs_step.τ)
    # println("    step deviation κ = ", Δκ - lhs_step.κ)
    # println("\n")

    # # #override the computed step
    # lhs_step.x .= Δx
    # lhs_step.z.vec .= Δz
    # lhs_step.s.vec .= Δs
    # lhs_step.τ = Δτ
    # lhs_step.κ = Δκ

end

function debug_manual_kkt_solve_2step(variables,data,scalings,res,lhs_step,kktsolver)
    #make a manual check of the KKT affine solve (version CVXOPT eqn 46)

    #print all of the manual steps
    println("\n--- MANUAL STEP CHECK (2 step) --- \n")

    Z = zeros
    m = data.m
    n = data.n
    p = 2*data.cone_info.type_counts[SecondOrderConeT]
    P = data.P
    A = data.A
    b = data.b
    c = data.c
    z = variables.z
    s = variables.s
    τ = variables.τ
    κ = variables.κ
    λ = scalings.λ
    l2 = deepcopy(λ)
    cones_circle_op!(scalings.cones, l2, λ, λ)

    L = debug_cones_matrixCircleOp(scalings.cones,λ)
    W = debug_cones_make_W(scalings.cones,s,z)
    WtW = debug_cones_make_WtW(scalings.cones,s,z)
    WtWsp = debug_cones_make_WtWsparse(scalings.cones,s,z)
    Wi = inv(Matrix(W))
    LW = L*W
    LWit = L*Wi'

    #form big linear system as in (46)
    M = [
     P         A';
      A       -WtW
    ]

    rhs = [-data.c;
           data.b]

    #solve as a big linear system
    lhs = IRsolve(M,rhs)

    #and now the sparse way
    Msp = [
     P         A'     Z(n,p);
      A       Z(m,m)  Z(m,p);
      Z(p,n)  Z(p,m)  Z(p,p)
    ]
    Msp[n+1:end,n+1:end] = -WtWsp
    #println("2 step M")
    #display(Matrix(M))

    rhssp = [-data.c;
              data.b;zeros(p)]

    #solve as a big linear system
    lhssp = IRsolve(Msp,rhssp)
    println("    RHS1",rhssp)

    #peel the pieces of from the front
    x1dense  = splice!(lhs,1:n)
    z1dense  = splice!(lhs,1:m)

    #peel the pieces of from the front
    x1  = splice!(lhssp,1:n)
    z1  = splice!(lhssp,1:m)

    #now the variable part

    #for the right hand side
    dx  = res.rx
    dz  = res.rz
    dτ  = res.rτ
    ds  = l2.vec
    dκ  = κ*τ

    rhs = [dx;
           -(dz - W'*(L\ds));zeros(p)]
    #solve as a big linear system
    lhs = IRsolve(Msp,rhs)
    println("    RHS2",rhs)

    #peel the pieces of from the front
    x2  = splice!(lhs,1:n)
    z2  = splice!(lhs,1:m)

    println()

    Wz1 = W*z1
    num = ( (dτ - dκ*τ) + c'*x2 + b'*z2  )

    den1 = κ/τ - (c'*x1 + b'*z1)
    den2 = (κ/τ + dot(Wz1,Wz1))
    den = den1

    println("    Δτ pieces: \n     ----------")
    println("    dτ - dκ/τ = ", (dτ - dκ*τ))
    println("    + c'Δx2   = ", c'*x2)
    println("    + b'Δz2   = ", b'*z2 )
    println("      tau_num = ", num)
    println()
    println("    τ/κ       = ", κ/τ)
    println("    - c'Δx1   = ", -c'*x1)
    println("    - b'Δz1   = ", -b'*z1)
    println("    norm(Wz1) = ",dot(Wz1,Wz1))
    println("    tau_den   = ", den1)
    println("    W_den     = ", den2)
    println()

    Δτ =  num/den

    Δx = x2 + Δτ.*x1
    Δz = z2 + Δτ.*z1

    #back solve for \Delta s and \Delta kappa
    Δs1  = -W'*(L\ds + W*Δz)
    Δs2  = -s.vec - WtW*Δz   #better?
    Δs   = Δs2
    Δκ  = -(dκ + κ*Δτ)/τ

    #print all of the manual steps

    println("\n")
    println("    x1 = ", x1)
    println("    z1 = ", z1)
    println("    x2 = ", x2)
    println("    z2 = ", z2)
    println()
    println("    Δx = ", Δx)
    println("    Δz = ", Δz)
    println("    Δs = ", Δs)
    print("    Δτ = ", Δτ)
    print("   (num = ",num,"   den = ",den,")\n")
    println("    Δκ = ", Δκ)
    println("\n")


    #work out the result of a full steps
    xnew = variables.x     + Δx
    znew = variables.z.vec + Δz
    snew = variables.s.vec + Δs
    τnew = variables.τ     + Δτ
    κnew = variables.κ     + Δκ

    # #compute the residuals
    # println("    solve acc. row1 = ", (-data.A*Δx  - Δs + Δτ*data.b - res.rz))
    # println("    solve acc. row2 = ", (data.P*Δx + data.A'*Δz + Δτ*data.c - res.rx))
    # println("    solve acc. row3 = ", -Δκ - c'*Δx - b'*Δz - res.rτ)
    # println()
    #
    # #compute the residuals
    # println("    affine step acc. row1 = ", (-data.A*xnew  - snew + τnew*data.b))
    # println("    affine step acc. row2 = ", (data.P*xnew + A'*znew + τnew*data.c))
    # println("    affine step acc. row3 = ", -κnew - c'*xnew - b'*znew)
    # println("    affine step acc. row4 = ", -κnew - c'*xnew - b'*znew)
    # println()
    #
    #
    # #How does this compare to the compute step?
    println("    step deviation x = ", Δx - lhs_step.x)
    println("    step deviation z = ", Δz - lhs_step.z.vec)
    println("    step deviation s = ", Δs - lhs_step.s.vec)
    println("    step deviation τ = ", Δτ - lhs_step.τ)
    println("    step deviation κ = ", Δκ - lhs_step.κ)
    println("\n")

    # # #override the computed step
    # lhs_step.x .= Δx
    # lhs_step.z.vec .= Δz
    # lhs_step.s.vec .= Δs
    # lhs_step.τ = Δτ
    # lhs_step.κ = Δκ


end


function debug_manual_kkt_solve(variables,data,scalings,res,lhs_step)
    #make a manual check of the KKT affine solve

    #print all of the manual steps
    println("\n--- MANUAL STEP CHECK (full) --- ")

    Z = zeros
    m = data.m
    n = data.n
    P = data.P
    A = data.A
    b = data.b
    c = data.c
    z = variables.z
    s = variables.s
    τ = variables.τ
    κ = variables.κ
    λ = scalings.λ
    l2 = deepcopy(λ)
    cones_circle_op!(scalings.cones, l2, λ, λ)

    L = debug_cones_matrixCircleOp(scalings.cones,λ)
    W = debug_cones_make_W(scalings.cones,s,z)
    Wi = inv(Matrix(W))
    LW = L*W
    LWit = L*Wi'

    #form big linear system as in (36)
    M = [
    Z(n,m)    Z(n,1)    -P        -A'    -c;
     I(m)     Z(m,1)     A     Z(m,m)    -b;
    Z(1,m)      1        c'        b'     0;
    LWit      Z(m,1)  Z(m,n)      LW    Z(m,1);
    Z(1,m)      τ     Z(1,n)   Z(1,m)     κ
    ]

    #for the right hand side
    dx  = res.rx
    dz  = res.rz
    dτ  = res.rτ
    ds  = l2.vec
    dκ  = κ*τ
    rhs = -[dx;dz;dτ;ds;dκ]

    #solve as a big linear system
    lhs = M\rhs

    #peel the pieces of from the front
    Δs  = splice!(lhs,1:m)
    Δκ  = splice!(lhs,1:1)[1]
    Δx  = splice!(lhs,1:n)
    Δz  = splice!(lhs,1:m)
    Δτ  = splice!(lhs,1:1)[1]

    #print all of the manual steps

    println("\n")
    println("    Δx = ", Δx)
    println("    Δz = ", Δz)
    println("    Δs = ", Δs)
    println("    Δτ = ", Δτ)
    println("    Δκ = ", Δκ)
    println("\n")


    #work out the result of a full steps
    xnew = variables.x     + Δx
    znew = variables.z.vec + Δz
    snew = variables.s.vec + Δs
    τnew = variables.τ     + Δτ
    κnew = variables.κ     + Δκ

    # #compute the residuals
    # println("    solve acc. row1 = ", (-data.A*Δx  - Δs + Δτ*data.b - res.rz))
    # println("    solve acc. row2 = ", (data.P*Δx + data.A'*Δz + Δτ*data.c - res.rx))
    # println("    solve acc. row3 = ", -Δκ - c'*Δx - b'*Δz - res.rτ)
    # println()
    #
    # #compute the residuals
    # println("    affine step acc. row1 = ", (-data.A*xnew  - snew + τnew*data.b))
    # println("    affine step acc. row2 = ", (data.P*xnew + A'*znew + τnew*data.c))
    # println("    affine step acc. row3 = ", -κnew - c'*xnew - b'*znew)
    # println("    affine step acc. row4 = ", -κnew - c'*xnew - b'*znew)
    # println()


    #How does this compare to the compute step?
    println("    step deviation x = ", Δx - lhs_step.x)
    println("    step deviation z = ", Δz - lhs_step.z.vec)
    println("    step deviation s = ", Δs - lhs_step.s.vec)
    println("    step deviation τ = ", Δτ - lhs_step.τ)
    println("    step deviation κ = ", Δκ - lhs_step.κ)
    println("\n")

    #override the computed step
    # lhs_step.x .= Δx
    # lhs_step.z.vec .= Δz
    # lhs_step.s.vec .= Δs
    # lhs_step.τ = Δτ
    # lhs_step.κ = Δκ


end




function debug_check_full_step_residuals(variables,lhs,res,data,scalings,rhs)

    #work out the result of a full steps
    xnew = variables.x     + lhs.x
    znew = variables.z.vec + lhs.z.vec
    snew = variables.s.vec + lhs.s.vec
    τnew = variables.τ     + lhs.τ
    κnew = variables.κ     + lhs.κ

    #compute the residuals
    println("Primal feasibility res = ", norm(data.A*xnew  + snew - data.b))
    println("Dual   feasibility res = ", norm(data.P*xnew + data.A'*znew + data.c))
    println()

    #check to see if we have solved the step equation correctly
    #we are checking here whether we are solving CVXOPT (44a/b) correctly
    r = res
    d = lhs

    row1 = -data.P*d.x - data.A'*d.z.vec - data.c*d.τ + r.rx

    row2 = +data.A*d.x + d.s.vec - data.b*d.τ + r.rz

    row3 = d.κ + data.c'*d.x + data.b'*d.z.vec + r.rτ

    #row 4-----------
    λ = scalings.λ
    tmp1 = deepcopy(λ)
    tmp2 = deepcopy(λ)
    tmp3 = deepcopy(λ)

    #compute WΔz
    cones_gemv_W!(scalings.cones,false,d.z,tmp1,1.0,0.0)
    #compute W^{-T}Δs
    cones_gemv_Winv!(scalings.cones,true,d.s,tmp2,1.0,0.0)
    #put the sum in tmp2
    tmp2.vec .+= tmp1.vec
    #λ∘sum
    cones_circle_op!(scalings.cones,tmp3,λ,tmp2)
    cones_circle_op!(scalings.cones,tmp1,λ,λ)

    row4 = tmp3.vec .+ tmp1.vec
    #-----------------

    row5 = variables.κ*d.τ + variables.τ*d.κ + variables.τ*variables.κ



    println("actual row 1 residual = ",norm(row1))
    println("actual row 2 residual = ",norm(row2))
    println("actual row 3 residual = ",norm(row3))
    println("actual row 4 residual = ",norm(row4))
    println("actual row 5 residual = ",norm(row5))

    println("\n")

end
