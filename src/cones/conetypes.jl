    # -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractCone{T} end

# -------------------------------------
# Zero Cone
# -------------------------------------

struct ZeroCone{T} <: AbstractCone{T}

    dim::DefaultInt

    function ZeroCone{T}(dim::Integer) where {T}
        dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
        new(dim)
    end

end

ZeroCone(args...) = ZeroCone{DefaultFloat}(args...)


# ------------------------------------
# Nonnegative Cone
# -------------------------------------

struct NonnegativeCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W and λ
    w::Vector{T}
    λ::Vector{T}

    function NonnegativeCone{T}(dim) where {T}

        dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
        w = zeros(T,dim)
        λ = zeros(T,dim)
        return new(dim,w,λ)

    end

end

NonnegativeCone(args...) = NonnegativeCone{DefaultFloat}(args...)

# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

mutable struct SecondOrderCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W and its products
    w::Vector{T}

    #scaled version of (s,z)
    λ::Vector{T}

    #vectors for rank 2 update representation of W^2
    u::Vector{T}
    v::Vector{T}

    #additional scalar terms for rank-2 rep
    d::T
    η::T

    function SecondOrderCone{T}(dim::Integer) where {T}
        dim >= 2 ? new(dim) : throw(DomainError(dim, "dimension must be >= 2"))
        w = zeros(T,dim)
        λ = zeros(T,dim)
        u = zeros(T,dim)
        v = zeros(T,dim)
        d = one(T)
        η = zero(T)
        return new(dim,w,λ,u,v,d,η)
    end

end

SecondOrderCone(args...) = SecondOrderCone{DefaultFloat}(args...)

# ------------------------------------
# Positive Semidefinite Cone (Scaled triangular form)
# ------------------------------------

mutable struct PSDConeWork{T}

    cholS::Union{Nothing,Cholesky{T,Matrix{T}}}
    cholZ::Union{Nothing,Cholesky{T,Matrix{T}}}
    SVD::Union{Nothing,SVD{T,T,Matrix{T}}}
    λ::Vector{T}
    Λisqrt::Diagonal{T,Vector{T}}
    R::Matrix{T}
    Rinv::Matrix{T}
    kronRR::Matrix{T}
    B::Matrix{T}
    WtW::Matrix{T}

    #workspace for various internal use
    workmat1::Matrix{T}
    workmat2::Matrix{T}
    workvec::Vector{T}

    function PSDConeWork{T}(n::Int) where {T}

        #there is no obvious way of pre-allocating
        #or recycling memory in these factorizations
        (cholS,cholZ,SVD) = (nothing,nothing,nothing)

        λ      = zeros(T,n)
        Λisqrt = Diagonal(zeros(T,n))
        R      = zeros(T,n,n)
        Rinv   = zeros(T,n,n)
        kronRR = zeros(T,n^2,n^2)
        B      = zeros(T,((n+1)*n)>>1,n^2)
        WtW    = zeros(T,size(B,1),size(B,1))

        workmat1 = zeros(T,n,n)
        workmat2 = zeros(T,n,n)
        workvec  = zeros(T,(n*(n+1))>>1)

        return new(cholS,cholZ,SVD,λ,Λisqrt,R,Rinv,
                   kronRR,B,WtW,workmat1,workmat2,workvec)
    end
end


struct PSDTriangleCone{T} <: AbstractCone{T}

        n::DefaultInt  #this is the matrix dimension, i.e. representing n /times n
    numel::DefaultInt  #this is the total number of elements in the matrix
     work::PSDConeWork{T}

    function PSDTriangleCone{T}(n) where {T}

        n >= 1 || throw(DomainError(dim, "dimension must be positive"))
        numel = (n*(n+1))>>1
        work = PSDConeWork{T}(n)

        return new(n,numel,work)

    end

end

PSDTriangleCone(args...) = PSDTriangleCone{DefaultFloat}(args...)


# ----------------------------------------------------
# Exponential Cone
# ----------------------------------------------------
#
# #   YC: contain both primal & dual variables at present
# mutable struct ExponentialCone{T} <: AbstractCone{T}
#
#     dim::DefaultInt
#
#     #internal working variables for W: s, δ, z, r, t
#     #In Mosek, W = [s/sqrt(<x,s>); δs/sqrt(<δx,δs>); sqrt(t)*z]⊤, W^{-1} = [x/sqrt(<x,s>); δx/sqrt(<δx,δs>); r/sqrt(t)]⊤
#
#     Hs::Matrix{T}       #Hessian workspace
#
#     st::Vector{T}
#     δs::Vector{T}
#     zt::Vector{T}
#     δz::Vector{T}
#     q::Vector{T}
#     r::Vector{T}
#     t::T
#     v::Vector{T}        #x
#     vt::Vector{T}
#
#
#     function ExponentialCone{T}() where {T}
#         dim = 3
#
#         #explicit initialzation of x,s as Mosek
#         # s = [1.290928; 0.805102; -0.827838]
#         # z = [1.290928; 0.805102; -0.827838]
#         st = zeros(T,3)
#         zt = zeros(T,3)
#         δs = zeros(T,3)
#         δz = zeros(T,3)
#         q = zeros(T,3)
#         r = zeros(T,3)
#         t = T(1)
#
#         W = zeros(T,3,3)
#         invW = zeros(T,3,3)
#         Hs = zeros(T,3,3)
#
#         v = W*z
#         vt = W*zt
#
#         return new(dim,W,invW,Hs,s,st,δs,z,zt,δz,q,r,t,v,vt)
#     end
#
# end


# ------------------------------------
# Exponential Cone
# ------------------------------------

# gradient and Hessian for the dual barrier function
mutable struct ExponentialCone{T} <: AbstractCone{T}

    dim::DefaultInt
    μH::Matrix{T}       #μ*H for the linear sysmtem
    grad::Vector{T}

    # workspace for centrality check
    μHWork::Matrix{T}
    gradWork::Vector{T}
    vecWork::Vector{T}
    FWork::Union{SuiteSparse.UMFPACK.UmfpackLU,Nothing}
    z::Vector{T}            # temporary storage for current z

    function ExponentialCone{T}() where {T}
        dim = 3
        μH = Matrix{T}(undef,3,3)
        grad = Vector{T}(undef,3)

        μHWork = Matrix{T}(undef,3,3)
        gradWork = Vector{T}(undef,3)
        vecWork = Vector{T}(undef,3)
        FWork = nothing
        z = Vector{T}(undef,3)
        return new(dim,μH,grad,μHWork,gradWork,vecWork,FWork,z)
    end
end

ExponentialCone(args...) = ExponentialCone{DefaultFloat}(args...)

# # ------------------------------------
# # Power Cone
# # ------------------------------------

# gradient and Hessian for the dual barrier function
mutable struct PowerCone{T} <: AbstractCone{T}

    dim::DefaultInt
    α::T
    μH::Matrix{T}       #μ*H for the linear sysmtem
    grad::Vector{T}

    # workspace for centrality check
    μHWork::Matrix{T}
    gradWork::Vector{T}
    vecWork::Vector{T}
    FWork::Union{SuiteSparse.UMFPACK.UmfpackLU,Nothing}
    z::Vector{T}            # temporary storage for current z

    function PowerCone{T}(α::T) where {T}
        dim = 3
        μH = Matrix{T}(undef,3,3)
        grad = Vector{T}(undef,3)

        μHWork = Matrix{T}(undef,3,3)
        gradWork = Vector{T}(undef,3)
        vecWork = Vector{T}(undef,3)
        FWork = nothing             #initialization
        z = Vector{T}(undef,3)
        return new(dim,α,μH,grad,μHWork,gradWork,vecWork,FWork,z)
    end
end

PowerCone(args...) = PowerCone{DefaultFloat}(args...)

# -------------------------------------
# Enum and dict for user interface
# -------------------------------------
"""
    SupportedCones
An Enum of supported cone type for passing to [`setup!`](@ref). The currently
supported types are:

* `ZeroConeT`       : The zero cone.  Used to define equalities.
* `NonnegativeConeT`: The nonnegative orthant.
* `SecondOrderConeT`: The second order / Lorentz / ice-cream cone.
# `PSDTriangleConeT`: The positive semidefinite cone (triangular format).
# `ExponentialConeT`: The exponetial cone.
# `PowerConeT`: The power cone (under development).
"""
@enum SupportedCones begin
    ZeroConeT
    NonnegativeConeT
    SecondOrderConeT
    PSDTriangleConeT
    ExponentialConeT
    PowerConeT
end

"""
    ConeDict
A Dict that maps the user-facing SupportedCones enum values to
the types used internally in the solver.   See [SupportedCones](@ref)
"""
const ConeDict = Dict(
           ZeroConeT => ZeroCone,
    NonnegativeConeT => NonnegativeCone,
    SecondOrderConeT => SecondOrderCone,
    PSDTriangleConeT => PSDTriangleCone,
    ExponentialConeT => ExponentialCone,
          PowerConeT => PowerCone,
)


# set of nonsymmetric cones
const NonsymmetricCones = [ExponentialConeT; PowerConeT]
