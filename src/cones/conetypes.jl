# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractCone{T} end

function Base.deepcopy(m::Type{<: AbstractCone{T}}) where {T}
    typeof(m)(deepcopy(m.dim))
end

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

#   YC: contain both primal & dual variables at present
mutable struct ExponentialCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W: s, δ, z, r, t
    #W = [s/sqrt(<x,s>); δs/sqrt(<δx,δs>); sqrt(t)*z]⊤, W^{-1} = [x/sqrt(<x,s>); δx/sqrt(<δx,δs>); r/sqrt(t)]⊤
    W::Matrix{T}        #Scaling matrix W
    invW::Matrix{T}     #inverse scaling matrix W^{-1}

    x::Vector{T}
    xt::Vector{T}
    δx::Vector{T}
    s::Vector{T}
    st::Vector{T}
    δs::Vector{T}
    z::Vector{T}
    r::Vector{T}
    t::T
    v::Vector{T}        #x
    vt::Vector{T}


    function ExponentialCone{T}() where {T}
        dim = 3

        #explicit initialzation of x,s as Mosek
        x = [1.290928; 0.805102; -0.827838]
        s = [1.290928; 0.805102; -0.827838]
        xt = 
        st =
        δx =
        δs =
        z =
        r =
        t =

        W =
        invW =

        v = W*x
        vt = W*xt

        return new(dim,W,invW,x,xt,δx,s,st,δs,z,r,t,v,vt)
    end

end

ExponentialCone(args...) = ExponentialCone{DefaultFloat}(args...)

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
# `ExponentialConeTT`: The exponetial cone (under development).
"""
@enum SupportedCones begin
    ZeroConeT
    NonnegativeConeT
    SecondOrderConeT
    PSDTriangleConeT
    ExponentialConeT
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
)
