# ---------------
# KKT Solver
# ---------------

mutable struct DefaultKKTSolver{T} <: AbstractKKTSolver{T}

    #the linear solver engine
    linsys::AbstractLinearSolver{T}

    #solution vector for constant part of KKT solves
    x1::Vector{T}
    z1::Vector{T}

    #solution vector for general KKT solves
    x2::Vector{T}
    z2::Vector{T}

    #work vectors for assembling/dissambling vectors
    workx::Vector{T}
    workz::Vector{T}
    work_conic::ConicVector{T}

        function DefaultKKTSolver{T}(
            data::DefaultProblemData{T},
            cones::ConeSet{T},
            settings::Settings{T}
        ) where {T}

        #basic problem dimensions
        (m, n) = (data.m, data.n)

        #create the linear solver
        solverengine = settings.direct_solve_method
        if solverengine == :qdldl
            linsys = QDLDLLinearSolver{T}(data.P,data.A,cones,m,n,settings)
        elseif solverengine == :mkl
            linsys = MKLPardisoLinearSolver{T}(data.P,data.A,cones,m,n,settings)
        else
            error("Unknown solver engine type: ", solverengine)
        end

        #the LHS constant part of the reduced solve
        x1   = Vector{T}(undef,n)
        z1   = Vector{T}(undef,m)

        #the LHS for other solves
        x2   = Vector{T}(undef,n)
        z2   = Vector{T}(undef,m)

        #workspace compatible with (x,z)
        workx   = Vector{T}(undef,n)
        workz   = Vector{T}(undef,m)

        #a conic workspace vector compatible with s and z
        work_conic = ConicVector{T}(cones)

        return new(linsys,x1,z1,x2,z2,workx,workz,work_conic)

    end

end

DefaultKKTSolver(args...) = DefaultKKTSolver{DefaultFloat}(args...)


function kkt_update!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T},
    cones::ConeSet{T}
) where {T}

    #update the linear solver with new scalings
    linsys_update!(kktsolver.linsys,cones)

    #calculate KKT solution for constant terms
    _kkt_solve_constant_rhs!(kktsolver,data)

    return nothing
end


function _kkt_solve_constant_rhs!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T}
) where {T}

    linsys_setrhs!(kktsolver.linsys, -data.q, data.b)
    linsys_solve!(kktsolver.linsys,kktsolver.x2, kktsolver.z2)

    return nothing
end


function kkt_solve_initial_point!(
    kktsolver::DefaultKKTSolver{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    # solve with [0;b] as a RHS to get (x,s) initializers
    # zero out any sparse cone variables at end
    kktsolver.workx .= zero(T)
    kktsolver.workz .= data.b
    linsys_setrhs!(kktsolver.linsys, kktsolver.workx, kktsolver.workz)
    linsys_solve!(kktsolver.linsys, variables.x, variables.s)

    # solve with [-c;0] as a RHS to get z initializer
    # zero out any sparse cone variables at end
    kktsolver.workx .= -data.q
    kktsolver.workz .=  zero(T)

    linsys_setrhs!(kktsolver.linsys, kktsolver.workx, kktsolver.workz)
    linsys_solve!(kktsolver.linsys, nothing, variables.z)

    return nothing
end


function kkt_solve!(
    kktsolver::DefaultKKTSolver{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::ConeSet{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    (x1,z1) = (kktsolver.x1, kktsolver.z1)
    (x2,z2) = (kktsolver.x2, kktsolver.z2)
    (workx,workz) = (kktsolver.workx, kktsolver.workz)

    #solve for (x1,z1)
    #-----------
    workx .= rhs.x

    if steptype == :affine
        #use -rz + s here as a shortcut in the affine step
        @. workz = -rhs.z + variables.s

    else  #:combined expected, but any general RHS should do this

        #we can use the overall LHS output as
        #additional workspace for the moment
        tmp1 = lhs.s; tmp2 = lhs.z
        @. tmp1 = rhs.z  #Don't want to modify our RHS
        cones_λ_inv_circ_op!(cones, tmp2, rhs.s)               #tmp2 = λ \ ds
        cones_gemv_W!(cones, :T, tmp2, tmp1, one(T), -one(T))  #tmp1 = - rhs.z + Wᵀ(tmp2)
        workz .= tmp1
    end

    #this solves the variable part of reduced KKT system
    linsys_setrhs!(kktsolver.linsys, workx, workz)
    linsys_solve!(kktsolver.linsys,x1,z1)

    #solve for Δτ.
    #-----------
    # Numerator first
    ξ   = workx
    ξ  .= variables.x / variables.τ
    P   = data.Psym

    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,x1) + dot(data.b,z1) + 2*symdot(ξ,P,x1)

    #offset ξ for the quadratic form in the denominator
    ξ_minus_x2    = ξ   #alias to ξ, same as workx
    ξ_minus_x2  .-= x2

    tau_den  = variables.κ/variables.τ - dot(data.q,x2) - dot(data.b,z2)
    tau_den += symdot(ξ_minus_x2,P,ξ_minus_x2) - symdot(x2,P,x2)

    #solve for (Δx,Δz)
    #-----------
    lhs.τ  = tau_num/tau_den
    @. lhs.x = x1 + lhs.τ * x2
    @. lhs.z = z1 + lhs.τ * z2

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz) = -Wᵀ(λ \ dₛ) - WᵀWΔz
    #where the first part is already in work_conic
    #-------------
    #PJG: We are unncessarily calculating λ \ dₛ twice.   Once here, and
    #once at ~line 195.   Do I even need it at all in the :affine case?
    tmpsv = kktsolver.work_conic
    cones_λ_inv_circ_op!(cones, tmpsv, rhs.s)                  #Δs = λ \ dₛ
    cones_gemv_W!(cones, :N, lhs.z, tmpsv,  one(T), one(T))    #Δs = WΔz + Δs
    cones_gemv_W!(cones, :T, tmpsv, lhs.s, -one(T), zero(T))   #Δs = -WᵀΔs

    #solve for Δκ
    #--------------
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
