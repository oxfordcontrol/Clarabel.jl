using IterativeSolvers
using LinearMaps
import Krylov

# ---------------
# KKT Solver (indirect method)
# ---------------

mutable struct DefaultKKTSolverIndirect{T} <: AbstractKKTSolver{T}

    n  #cols in A
    m  #rows in A
    KKTmap::LinearMap{T}
    #internal memory for LinearMap multiply
    work::SplitVector{T}

    #RHS and solution vector for constant part of KKT solves
    rhs_cb::Vector{T}
    lhs_cb::Vector{T}
    #views into constant RHS and solution
    rhs_cb_x::VectorView{T}
    rhs_cb_z::VectorView{T}
    lhs_cb_x::VectorView{T}
    lhs_cb_z::VectorView{T}

    #RHS/LHS solution vector for variable/step part of KKT solves
    rhs_step::Vector{T}
    lhs_step::Vector{T}


    function DefaultKKTSolverIndirect{T}(
        data::DefaultProblemData{T},
        scalings::DefaultScalings{T}) where {T}

        n = data.n
        m = data.m

        #for storing intermediate products in KKT mapping
        work  = SplitVector{T}(data.cone_info)

        KKTmap = initialize_kkt_map(data.A,scalings,work)

        #the solution for the constant part of 3x3 KKT system
        rhs_cb = Vector{T}(undef,n + m)
        lhs_cb = Vector{T}(undef,n + m)
        #views into the solution solution for x/z partition
        rhs_cb_x = view(rhs_cb,1:n)
        rhs_cb_z = view(rhs_cb,(n+1):(n+m))
        lhs_cb_x = view(lhs_cb,1:n)
        lhs_cb_z = view(lhs_cb,(n+1):(n+m))

        #hold a copy of the RHS built from constant data terms
        rhs_cb_x .= -data.c;
        rhs_cb_z .=  data.b;

        ##solution vector for variable/step part of 3x3 KKT solves
        rhs_step = Vector{T}(undef,n + m)
        lhs_step = Vector{T}(undef,n + m)

        #zero all the LHS terms for minres! so we
        #don't accidentally warm start somewhere gnarly
        lhs_cb   .= 0.0
        lhs_step .= 0.0


        return new(n,m,KKTmap,work,
            rhs_cb,lhs_cb,
            rhs_cb_x,rhs_cb_z,lhs_cb_x,lhs_cb_z,
            rhs_step,lhs_step)

    end

end

DefaultKKTSolverIndirect(args...) = DefaultKKTSolverIndirect{DefaultFloat}(args...)

function initialize_kkt_map(
    A::AbstractMatrix{T},
    scalings::DefaultScalings{T},
    work::SplitVector{T}
) where{T}

    m = size(A,1)
    n = size(A,2)
    mymap = (y,x) -> _kkt_mul!(A,m,n,scalings,x,y,work)

    KKTmap = LinearMap(mymap, m+n; issymmetric=true,isposdef=false)

    return KKTmap
end

function _kkt_mul!(
    A::AbstractMatrix{T},
    m::DefaultInt,
    n::DefaultInt,
    scalings::DefaultScalings{T},
    x::Vector{T},
    y::Vector{T},
    work::SplitVector{T}
) where {T}

    # it doesn't seem possible to pass through views
    # created in the main solver object into this
    # function since they go via minres!.   Just make
    # new ones and hope that it is fast.
    x1 = view(x,1:n)
    x2 = view(x,(n+1):(m+n))
    y1 = view(y,1:n)
    y2 = view(y,(n+1):(m+n))

    #form the product in the first block row
    mul!(y1,A',x2)

    #form the product in the second block row
    work.vec .= x2
    cones_mul_WtW!(scalings.cones,work,work) #work = W^TW*work
    mul!(y2,A,x1)
    y2 .-= work.vec      #z <- Ax - W^TWz

    return nothing
end

function _solve_kkt!(
    kktsolver::DefaultKKTSolverIndirect{T},
    x::Vector{T},
    b::Vector{T},
) where {T}

    tmp,stats = Krylov.minres(kktsolver.KKTmap,b)
    x .= tmp

    return nothing
end


function kkt_update!(
    kktsolver::DefaultKKTSolverIndirect{T},
    scalings::DefaultScalings{T}) where {T}

    #calculate KKT solution for constant terms
    _kkt_solve_constant_rhs!(kktsolver)

    #nothing else needs to be done since we
    #only ever multiply by WtW.  There is
    #no KKT matrix update/factor to perform like
    #in a direct method
    return nothing
end


function _kkt_solve_constant_rhs!(
    kktsolver::DefaultKKTSolverIndirect{T}
) where {T}

    #rhs prebuilt in the constructor
    _solve_kkt!(kktsolver,kktsolver.lhs_cb,kktsolver.rhs_cb)

    return nothing
end

function kkt_solve_initial_point!(
    kktsolver::DefaultKKTSolverIndirect{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    # arbitrarly choose lhs_cb/rhs_cb space for solving,
    # then set it back to zero when done.

    # solve with [0;b] as a RHS to get (x,s) initializers
    kktsolver.rhs_cb_x .= 0.
    kktsolver.rhs_cb_z .= data.b
    kktsolver.lhs_cb   .= 0.

    _solve_kkt!(kktsolver,kktsolver.lhs_cb,kktsolver.rhs_cb)

    variables.x      .= kktsolver.lhs_cb_x
    variables.s.vec  .= kktsolver.rhs_cb_z

    # solve with [-c;0] as a RHS to get z initializer
    # zero LHS so I don't warm start from (x,s) solution
    kktsolver.rhs_cb_x .= -data.c
    kktsolver.rhs_cb_z .=  0.
    kktsolver.lhs_cb   .=  0.
    _solve_kkt!(kktsolver,kktsolver.lhs_cb,kktsolver.rhs_cb)
    variables.z.vec  .= kktsolver.lhs_cb_z

    #rezero the LHS one last time so that SolveKKTConstantRHS!
    #isn't warm starting from this old solution
    kktsolver.lhs_cb .= 0.

    return nothing
end

function kkt_solve!(
    kktsolver::DefaultKKTSolverIndirect{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
    data::DefaultProblemData{T}
) where{T}

    m = data.m
    n = data.n
    cones = scalings.cones

    constx = kktsolver.lhs_cb_x
    constz = kktsolver.lhs_cb_z
    lhs_step = kktsolver.lhs_step
    rhs_step = kktsolver.rhs_step

    # assemble the right hand side and solve in place
    rhs_step[1:n]     .= rhs.x
    rhs_step[n+1:n+m] .= rhs.z.vec

    # use a different lhs vector for the :affine
    # and :combined step to improve warm starting

    _solve_kkt!(kktsolver,lhs_step,rhs_step)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x     .= lhs_step[1:n]
    lhs.z.vec .= lhs_step[n+1:n+m]

    #solve for Δτ
    lhs.τ  = rhs.τ - rhs.κ/variables.τ + dot(data.c,lhs.x) + dot(data.b,lhs.z.vec)
    lhs.τ /= variables.κ/variables.τ - dot(data.c,constx) - dot(data.b,constz)

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x     .+= lhs.τ .* constx
    lhs.z.vec .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    cones_inv_circle_op!(cones, lhs.s, scalings.λ, rhs.s) #Δs = λ \ dₛ
    cones_gemv_W!(cones, false, lhs.z, lhs.s,  1., 1.)    #Δs = WΔz + Δs
    cones_gemv_W!(cones,  true, lhs.s, lhs.s, -1., 0.)    #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ      = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
