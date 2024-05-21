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
    workz::Vector{T}
    work_conic::Vector{T}

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
        x1   = zeros(T,n)
        z1   = zeros(T,m)

        #the LHS for other solves
        x2   = zeros(T,n)
        z2   = zeros(T,m)

        #workspace compatible with (x,z)
        workx   = zeros(T,n)
        workz   = zeros(T,m)

        #additional conic workspace vector compatible with s and z
        work_conic = zeros(T,m)

        return new(kktsolver,x1,z1,x2,z2,workx,workz,work_conic)

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

    if nnz(data.P) == 0
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


function kkt_solve!(
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

    #solve for Δκ
    #--------------
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    # we don't check the validity of anything
    # after the KKT solve, so just return is_success
    # without further validation
    return is_success

end

#update the KKT system with new P and A
function kkt_update_P!(
    kktsystem::DefaultKKTSystem{T},
    P::SparseMatrixCSC{T}
) where{T}
    kktsolver_update_P!(kktsystem.kktsolver,P)
    return nothing
end

function kkt_update_A!(
    kktsystem::DefaultKKTSystem{T},
    A::SparseMatrixCSC{T}
) where{T}
    kktsolver_update_A!(kktsystem.kktsolver,A)
    return nothing
end