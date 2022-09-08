using StaticArrays
# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractCone{T <: AbstractFloat} end

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
    Hs::Matrix{T}

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
        Hs    = zeros(T,size(B,1),size(B,1))

        workmat1 = zeros(T,n,n)
        workmat2 = zeros(T,n,n)
        workvec  = zeros(T,(n*(n+1))>>1)

        return new(cholS,cholZ,SVD,λ,Λisqrt,R,Rinv,
                   kronRR,B,Hs,workmat1,workmat2,workvec)
    end
end


struct PSDTriangleCone{T} <: AbstractCone{T}

        n::DefaultInt  #this is the matrix dimension, i.e. matrix is n /times n
    numel::DefaultInt  #this is the total number of elements (lower triangle of) the matrix
     work::PSDConeWork{T}

    function PSDTriangleCone{T}(n) where {T}

        n >= 1 || throw(DomainError(dim, "dimension must be positive"))
        numel = (n*(n+1))>>1
        work = PSDConeWork{T}(n)

        return new(n,numel,work)

    end

end

PSDTriangleCone(args...) = PSDTriangleCone{DefaultFloat}(args...)


# ------------------------------------
# Exponential Cone
# ------------------------------------

# Exp and power cones always use fixed 3x1 or 3x3 fields, which 
# are best handled using MArrays from StaticArrays.jl.  However, 
# that doesn't work for non isbits type (specifically BigFloat), 
# so we need to use SizedArrays in that case.   Either way we still 
# want the ExponentialCone and PowerCone structs to be concrete, 
# hence the monstrosity of a constructor below.

function _static_3d_cone_types(T)
    M3T_ISBITS{T}     = MMatrix{3,3,T,9}
    M3T_NOT_ISBITS{T} = SizedMatrix{3, 3, T, 2, Matrix{T}} 
    V3T_ISBITS{T}     = MVector{3,T} 
    V3T_NOT_ISBITS{T} = SizedVector{3,T,Vector{T}}
    if isbitstype(T)
        return (M3T_ISBITS{T}, V3T_ISBITS{T})
    else 
        return (M3T_NOT_ISBITS{T},V3T_NOT_ISBITS{T})
    end
end

mutable struct ExponentialCone{T,M3T,V3T} <: AbstractCone{T}

    H_dual::M3T      #Hessian of the dual barrier at z 
    Hs::M3T          #scaling matrix
    grad::V3T        #gradient of the dual barrier at z 
    z::V3T           #holds copy of z at scaling point

    function ExponentialCone{T}() where {T}

        (M3T,V3T) = _static_3d_cone_types(T)
        H_dual = M3T(zeros(T,3,3))
        Hs     = M3T(zeros(T,3,3))
        grad   = V3T(zeros(T,3))
        z      = V3T(zeros(T,3))

        return new{T,M3T,V3T}(H_dual,Hs,grad,z)
    end
end

ExponentialCone(args...) = ExponentialCone{DefaultFloat}(args...)

# # ------------------------------------
# # Power Cone
# # ------------------------------------

# gradient and Hessian for the dual barrier function
mutable struct PowerCone{T} <: AbstractCone{T}

    α::T
    H_dual::MMatrix{3,3,T,9}        #Hessian of the dual barrier at z 
    Hs::MMatrix{3,3,T,9}            #scaling matrix
    grad::MVector{3,T}              #gradient of the dual barrier at z 
    z::MVector{3,T}                 #holds copy of z at scaling point
    cholH::MMatrix{3,3,T,9}         # workspace for 3x3 Cholesky factorization

    function PowerCone{T}(α::T) where {T}

        H_dual = @MMatrix zeros(T,3,3)
        Hs = @MMatrix zeros(T,3,3)
        grad = @MVector zeros(T,3)
        z = @MVector zeros(T,3)
        cholH = @MMatrix zeros(T,3,3)

        return new(α,H_dual,Hs,grad,z,cholH)
    end
end

PowerCone(args...) = PowerCone{DefaultFloat}(args...)


"""
    ConeDict
A Dict that maps the user-facing SupportedCone types to
the types used internally in the solver.   See [SupportedCone](@ref)
"""
const ConeDict = Dict(
           ZeroConeT => ZeroCone,
    NonnegativeConeT => NonnegativeCone,
    SecondOrderConeT => SecondOrderCone,
    PSDTriangleConeT => PSDTriangleCone,
    ExponentialConeT => ExponentialCone,
          PowerConeT => PowerCone,
)
