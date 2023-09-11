# ---------------
# KKT System
# ---------------

mutable struct DefaultKKTSystem{T} <: AbstractKKTSystem{T}

    #the KKT system solver
    kktsolver::AbstractKKTSolver{T}

    #solution vector for constant part of KKT solves
    x1::Vector{T}
    z1::Vector{T}

    #solution vector for general KKT solves
    x2::Vector{T}
    z2::Vector{T}

    #work vectors for assembling/disassembling vectors
    workx::Vector{T}
    workz::ConicVector{T}
    work_conic::ConicVector{T}

    #workspace for outer iterative refinement 
    vars_dx::DefaultVariables{T}
    vars_e::DefaultVariables{T}

        function DefaultKKTSystem{T}(
            data::DefaultProblemData{T},
            cones::CompositeCone{T},
            settings::Settings{T}
        ) where {T}

        #basic problem dimensions
        (m, n) = (data.m, data.n)

        #create the linear solver.  Always LDL for now
        kktsolver = DirectLDLKKTSolver{T}(data.P,data.A,cones,m,n,settings)

        #the LHS constant part of the reduced solve
        x1   = Vector{T}(undef,n)
        z1   = Vector{T}(undef,m)

        #the LHS for other solves
        x2   = Vector{T}(undef,n)
        z2   = Vector{T}(undef,m)

        #workspace compatible with (x,z)
        workx   = Vector{T}(undef,n)
        workz   = ConicVector{T}(cones)

        #workspace for outer iterative refinement
        vars_dx = DefaultVariables{T}(n,cones)
        vars_e  = DefaultVariables{T}(n,cones)

        #additional conic workspace vector compatible with s and z
        work_conic = ConicVector{T}(cones)

        return new(kktsolver,x1,z1,x2,z2,workx,workz,work_conic,vars_dx,vars_e)

    end

end

DefaultKKTSystem(args...) = DefaultKKTSystem{DefaultFloat}(args...)

function kkt_update!(
    kktsystem::DefaultKKTSystem{T},
    data::DefaultProblemData{T},
    cones::CompositeCone{T}
) where {T}

    #update the linear solver with new cones
    is_success  = kktsolver_update!(kktsystem.kktsolver,cones)

    #bail if the factorization has failed 
    is_success || return is_success

    #calculate KKT solution for constant terms
    is_success = _kkt_solve_constant_rhs!(kktsystem,data)

    return is_success
end

function _kkt_solve_constant_rhs!(
    kktsystem::DefaultKKTSystem{T},
    data::DefaultProblemData{T}
) where {T}

    @. kktsystem.workx = -data.q;

    kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, data.b)
    is_success = kktsolver_solve!(kktsystem.kktsolver, kktsystem.x2, kktsystem.z2)

    return is_success

end


function kkt_solve_initial_point!(
    kktsystem::DefaultKKTSystem{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    if iszero(nnz(data.P))
        # LP initialization
        # solve with [0;b] as a RHS to get (x,-s) initializers
        # zero out any sparse cone variables at end
        kktsystem.workx .= zero(T)
        kktsystem.workz .= data.b
        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, variables.x, variables.s)
        variables.s .= -variables.s

        if !is_success return is_success end

        # solve with [-q;0] as a RHS to get z initializer
        # zero out any sparse cone variables at end
        @. kktsystem.workx = -data.q
        kktsystem.workz .=  zero(T)

        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, nothing, variables.z)
    else
        # QP initialization
        @. kktsystem.workx = -data.q
        @. kktsystem.workz = data.b
        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, variables.x, variables.z)
        @. variables.s = -variables.z
    end

    return is_success

end

function _debug_check_residuals(    
    kktsystem::DefaultKKTSystem{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::CompositeCone{T},) where{T}


    dz = deepcopy(rhs.z)
    dτ = deepcopy(rhs.τ)
    ds = deepcopy(rhs.s)
    dx = deepcopy(rhs.x)
    dκ = deepcopy(rhs.κ)
    ξ = variables.x/variables.τ

    P = Symmetric(data.P)

    # manually check (44a)
    row1 = dx - P*lhs.x - data.A'*lhs.z - data.q*lhs.τ 
    row2 = dz.vec + data.A*lhs.x + lhs.s - data.b*lhs.τ 


    row3 = dτ + lhs.κ + 2*quad_form(ξ,P,lhs.x) +  dot(data.q,lhs.x) + dot(data.b,lhs.z) - ξ'*P*ξ*lhs.τ
    
    row4 = deepcopy(rhs.s); row4.vec .= 0.

    #dz was used.  Recycle as workspace 
    refine_ds!(cones,row4,lhs.z,lhs.s, dz)
    row4 = ds + row4
    

    row5 = dκ + (variables.κ*lhs.τ + variables.τ*lhs.κ)



    println("(44a) errors: row 1 = ", norm(row1,Inf))
    println("(44a) errors: row 2 = ", norm(row2,Inf))
    println("(44a) errors: row 3 = ", norm(row3,Inf))
    println("(44a) errors: row 4 = ", norm(row4,Inf))
    println("(44a) errors: row 5 = ", norm(row5,Inf))


    # ξ = variables.x / variables.τ
    # Δκ = - dτ - (data.q + 2*data.P*ξ)'*lhs.x - dot(data.b,lhs.z) + ξ'*data.P*ξ*lhs.τ

end 


function kkt_solve!(
    kktsystem::DefaultKKTSystem{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::CompositeCone{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    dx = kktsystem.vars_dx
    e  = kktsystem.vars_e

    # println("rhs.x  = ", norm(rhs.x,Inf))
    # println("rhs.z  = ", norm(rhs.z,Inf))
    # println("rhs.τ  = ", abs(rhs.τ))
    # println("rhs.s  = ", norm(rhs.s,Inf))
    # println("rhs.κ  = ", abs(rhs.κ))

    is_success = kkt_solve_inner!(kktsystem,lhs,rhs,data,variables,cones,steptype)

    # _debug_check_residuals(    
    #         kktsystem,lhs,rhs,data,variables,cones) 

    rhs_norminf = variables_norminf(rhs)

    #compute the error residual 

    for i in 1:3

        variables_refine_step_rhs!(e,lhs,variables,data,cones)

        #e = b - Ax  :: NB b = -rhs ??? 
        #PJG: some sign weirdness here.   Fixing 
        #it would maybe make the correction term 
        #below come out with a more attractive "+" sign
        #variables_axpby!(e,rhs,-one(T),-one(T))
        e.x .+=  rhs.x
        e.z .+=  rhs.z 
        e.s .+=  rhs.s
        e.τ  +=  rhs.τ
        e.κ  +=  rhs.κ 

        e_norminf = variables_norminf(e)

        # println("$i: ex  = ", norm(e.x,Inf))
        # println("$i: ez  = ", norm(e.z,Inf))
        # println("$i: eτ  = ", abs(e.τ))
        # println("$i: es  = ", norm(e.s,Inf))
        # println("$i: eκ  = ", abs(e.κ))


        if e_norminf <= 1e-10*(1 + rhs_norminf)
           break 
        end
        #println("Refining: norms = ($e_norminf, $rhs_norminf)")

        is_success = kkt_solve_inner!(kktsystem,dx,e,data,variables,cones,:refine)

        lhs.x .+=  dx.x
        lhs.z .+=  dx.z 
        lhs.s .+=  dx.s
        lhs.τ +=  dx.τ
        lhs.κ +=  dx.κ 

        # println("$i: ex  = ", norm(e.x,Inf))
        # println("$i: ez  = ", norm(e.z,Inf))
        # println("$i: eτ  = ", abs(e.τ))
        # println("$i: es  = ", norm(e.s,Inf))
        # println("$i: eκ  = ", abs(e.κ))

    #    _debug_check_residuals(    
    #        kktsystem,lhs,rhs,data,variables,cones) 

    end

    #println("\n----IR done -----\n")


    # we don't check the validity of anything
    # after the KKT solve, so just return is_success
    # without further validation
    return is_success

end


function kkt_solve_inner!(
    kktsystem::DefaultKKTSystem{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::CompositeCone{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    (x1,z1) = (kktsystem.x1, kktsystem.z1)
    (x2,z2) = (kktsystem.x2, kktsystem.z2)
    (workx,workz) = (kktsystem.workx, kktsystem.workz)

    #solve for (x1,z1)
    #-----------
    @. workx = rhs.x

    # compute the vector c in the step equation HₛΔz + Δs = -c,  
    # with shortcut in affine case
    Δs_const_term = kktsystem.work_conic

    if steptype == :affine
        @. Δs_const_term = variables.s

    elseif steptype == :refine 
        @. Δs_const_term .= zero(T)

    else  #:combined expected, but any general RHS should do this
        #we can use the overall LHS output as additional workspace for the moment
        Δs_from_Δz_offset!(cones,Δs_const_term,rhs.s,lhs.z,variables.z)
    end

    @. workz = Δs_const_term - rhs.z


    #---------------------------------------------------
    #this solves the variable part of reduced KKT system
    kktsolver_setrhs!(kktsystem.kktsolver, workx, workz)
    is_success = kktsolver_solve!(kktsystem.kktsolver,x1,z1)

    if !is_success return false end

    #solve for Δτ.
    #-----------
    # Numerator first
    ξ   = workx
    @. ξ = variables.x / variables.τ

    P   = Symmetric(data.P)

    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,x1) + dot(data.b,z1) + 2*quad_form(ξ,P,x1)

    #offset ξ for the quadratic form in the denominator
    ξ_minus_x2    = ξ   #alias to ξ, same as workx
    @. ξ_minus_x2  -= x2

    tau_den  = variables.κ/variables.τ - dot(data.q,x2) - dot(data.b,z2)
    tau_den += quad_form(ξ_minus_x2,P,ξ_minus_x2) - quad_form(x2,P,x2)

    #solve for (Δx,Δz)
    #-----------
    lhs.τ  = tau_num/tau_den
    @. lhs.x = x1 + lhs.τ * x2
    @. lhs.z = z1 + lhs.τ * z2


    #solve for Δs
    #-------------
    # compute the linear term HₛΔz, where Hs = WᵀW for symmetric
    # cones and Hs = μH(z) for asymmetric cones
    mul_Hs!(cones,lhs.s,lhs.z,workz)
    @. lhs.s = -(lhs.s + Δs_const_term)

    #ALTERNATIVE: just take directly from our paper (14a)
    #lhs.s .= -(data.A*lhs.x - data.b*lhs.τ + rhs.z)
    #println("s recovery error = ", norm(lhs.s.vec - s2,Inf))

    #solve for Δκ
    #--------------
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ


    # we don't check the validity of anything
    # after the KKT solve, so just return is_success
    # without further validation
    return is_success

end
