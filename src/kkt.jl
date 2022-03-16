# ---------------
# KKT Solver
# ---------------

mutable struct DefaultKKTSolver{T} <: AbstractKKTSolver{T}

    n  #cols in A
    m  #rows in A
    p  #extra KKT columns for sparse SOCs

    #the linear solver engine
    linsys::AbstractLinearSolver{T}

    #solution vector for constant part of KKT solves
    lhs_const::Vector{T}
    lhs_const_x::VectorView{T}
    lhs_const_z::VectorView{T}
    lhs_const_p::VectorView{T}

    #solution vector for general KKT solves
    lhs::Vector{T}
    lhs_x::VectorView{T}
    lhs_z::VectorView{T}
    lhs_p::VectorView{T}

    #work vector for solves, e.g. right hand sides
    work::Vector{T}
    work_x::VectorView{T}
    work_z::VectorView{T}
    work_p::VectorView{T}

    #a work ConicVector to simplify solving for Δs
    work_sv::ConicVector{T}

        function DefaultKKTSolver{T}(
            data::DefaultProblemData{T},
            scalings::DefaultScalings{T},
            settings::Settings{T}
        ) where {T}

        #basic problem dimensions
        n = data.n
        m = data.m

        #create the linear solver
        solverengine = settings.direct_solve_method
        if solverengine == :qdldl
            linsys = QDLDLLinearSolver{T}(data.P,data.A,scalings,m,n,settings)
        elseif solverengine == :mkl
            linsys = MKLPardisoLinearSolver{T}(data.P,data.A,scalings,m,n,settings)
        else
            error("Unknown solver engine type: ", solverengine)
        end


        #does our solver use sparse SOC format?
        if linsys_is_soc_sparse_format(linsys)
            p = linsys_soc_sparse_variables(linsys)
        else
            p = 0
        end

        #the LHS constant part of the reduced solve
        lhs_const   = Vector{T}(undef,n + m + p)
        #views into the LHS for x/z partition
        lhs_const_x = view(lhs_const,1:n)
        lhs_const_z = view(lhs_const,(n+1):(n+m))
        lhs_const_p = view(lhs_const,(n+m+1):(n+m+p))

        #the LHS for other solves
        lhs      = Vector{T}(undef,n + m + p)
        #views into the LHS for x/z partition
        lhs_x    = view(lhs,1:n)
        lhs_z    = view(lhs,(n+1):(n+m))
        lhs_p    = view(lhs,(n+m+1):(n+m+p))

        #work vector for other solves
        work  = Vector{T}(undef, n + m + p)
        #views into the work vector
        work_x = view(work,1:n)
        work_z = view(work,(n+1):(n+m))
        work_p = view(work,(n+m+1):(n+m+p))

        #a split vector compatible with s and z
        work_sv = ConicVector{T}(data.cone_info)


        return new(
            n,m,p,linsys,
            lhs_const,lhs_const_x,lhs_const_z,lhs_const_p,
            lhs,lhs_x,lhs_z,lhs_p,
            work, work_x, work_z, work_p,work_sv)

    end

end

DefaultKKTSolver(args...) = DefaultKKTSolver{DefaultFloat}(args...)


function kkt_update!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T},
    scalings::DefaultScalings{T}
) where {T}

    #update the linear solver with new scalings
    linsys_update!(kktsolver.linsys,scalings)

    #calculate KKT solution for constant terms
    _kkt_solve_constant_rhs!(kktsolver,data)

    return nothing
end


function _kkt_solve_constant_rhs!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T}
) where {T}

    #make the RHS for the constant part
    #of the reduced solve
    kktsolver.work_x .= -data.q;
    kktsolver.work_z .=  data.b;
    kktsolver.work_p .=  zero(T);

    linsys_solve!(kktsolver.linsys,kktsolver.lhs_const,kktsolver.work)

    return nothing
end


function kkt_solve_initial_point!(
    kktsolver::DefaultKKTSolver{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    # solve with [0;b] as a RHS to get (x,s) initializers
    # zero out any sparse cone variables at end
    kktsolver.work_x .= zero(T)
    kktsolver.work_z .= data.b
    kktsolver.work_p .= zero(T)

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)
    variables.x .=  kktsolver.lhs_x
    variables.s .= -kktsolver.lhs_z

    # solve with [-c;0] as a RHS to get z initializer
    # zero out any sparse cone variables at end
    kktsolver.work_x .= -data.q
    kktsolver.work_z .=  zero(T)
    kktsolver.work_p .=  zero(T)

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)
    variables.z .= kktsolver.lhs_z

    return nothing
end


function kkt_solve!(
    kktsolver::DefaultKKTSolver{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
    data::DefaultProblemData{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    cones = scalings.cones
    constx = kktsolver.lhs_const_x
    constz = kktsolver.lhs_const_z

    # assemble the right hand side and solve.  We need to
    # modify terms for the z part here since this solve
    # function is based on the condensed KKT solve approach
    # of CVXOPT
    kktsolver.work_x .= rhs.x
    kktsolver.work_p .= 0

    if steptype == :affine
        #use -rz + s here as a shortcut in the affine step
        @. kktsolver.work_z = -rhs.z + variables.s

    else  #:combined expected, but any general RHS should do this

        #we can use the LHS outputs for work space
        #here since we haven't solved yet
        tmp1 = lhs.s; tmp2 = lhs.z
        @. tmp1 = rhs.z  #Don't want to modify our RHS
        cones_λ_inv_circ_op!(cones, tmp2, rhs.s)               #tmp2 = λ \ ds
        cones_gemv_W!(cones, :T, tmp2, tmp1, one(T), -one(T))  #tmp1 = - rhs.z + W(tmp2)
        kktsolver.work_z .= tmp1

    end

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x .= kktsolver.lhs_x
    lhs.z .= kktsolver.lhs_z

    #use workx as scratch space now that lhs is copied
    ξ   = kktsolver.work_x
    ξ  .= variables.x / variables.τ
    P   = data.Psym

    #solve for Δτ.
    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,lhs.x) + dot(data.b,lhs.z) + 2*symdot(ξ,P,lhs.x)

    #now offset ξ for the quadratic form in the denominator
    ξ_minus_x2    = ξ   #alias to ξ, same as work_x
    ξ_minus_x2  .-= constx

    tau_den = (variables.κ/variables.τ - dot(data.q,constx) - dot(data.b,constz) + symdot(ξ_minus_x2,P,ξ_minus_x2) - symdot(constx,P,constx))

    # Δτ = tau_num/tau_den
    lhs.τ  = tau_num/tau_den

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x .+= lhs.τ .* constx
    lhs.z .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    #PJG: We are unncessarily calculating λ \ dₛ twice.   Once here, and
    #once at ~line 195.   Do I even need it at all in the :affine case?
    tmpsv = kktsolver.work_sv
    cones_λ_inv_circ_op!(cones, tmpsv, rhs.s)                  #Δs = λ \ dₛ
    cones_gemv_W!(cones, :N, lhs.z, tmpsv,  one(T), one(T))    #Δs = WΔz + Δs
    cones_gemv_W!(cones, :T, tmpsv, lhs.s, -one(T), zero(T))   #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
