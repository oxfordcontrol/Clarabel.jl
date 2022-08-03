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


#####################################
# LAPACK Implementation
#####################################
import LinearAlgebra.BLAS
import LinearAlgebra.BLAS.@blasfunc
using Base: iszero, require_one_based_indexing
using LinearAlgebra: chkstride1, checksquare
# For LU decomposition
const DGETRF_ = (BLAS.@blasfunc(dgetrf_),Base.liblapack_name)
const DGETRS_ = (BLAS.@blasfunc(dgetrs_),Base.liblapack_name)

mutable struct LuBlasWorkspace{T}
    dim::Int64
    ipiv::Vector{BLAS.BlasInt}
    info::Base.RefValue{BLAS.BlasInt}

    function LuBlasWorkspace{T}(n::Int64) where {T <: AbstractFloat}

        #workspace data for BLAS
        dim = n
        ipiv = Vector{BLAS.BlasInt}(undef,n)
        info  = Ref{BLAS.BlasInt}()

        new(dim,ipiv,info)
    end
end

(getrf, getrs, elty) = (:DGETRF_, :DGETRS_, :Float64)

@eval begin
    # SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )
    # *     .. Scalar Arguments ..
    #       INTEGER            INFO, LDA, M, N
    # *     .. Array Arguments ..
    #       INTEGER            IPIV( * )
    #       DOUBLE PRECISION   A( LDA, * )

    function getrf!(A::AbstractMatrix{$elty},ws::LuBlasWorkspace{$elty})
        require_one_based_indexing(A)
        chkstride1(A)
        n = ws.dim
        lda  = max(1,stride(A, 2))
        ipiv = ws.ipiv
        info = ws.info
        ccall($getrf, Cvoid,
                (Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ptr{$elty},
                Ref{BLAS.BlasInt}, Ptr{BLAS.BlasInt}, Ptr{BLAS.BlasInt}),
                n, n, A, lda, ipiv, info)
        LAPACK.chkargsok(info[])

    end

    ########################################
    #compute the inverse of a LU factorization
    ########################################

    #     SUBROUTINE DGETRS( TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO )
    #*     .. Scalar Arguments ..
    #      CHARACTER          TRANS
    #      INTEGER            INFO, LDA, LDB, N, NRHS
    #     .. Array Arguments ..
    #      INTEGER            IPIV( * )
    #      DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
    function getrs!(A::AbstractMatrix{$elty}, ws::LuBlasWorkspace{$elty}, B::AbstractVecOrMat{$elty})
        trans = 'N'
        ipiv = ws.ipiv
        require_one_based_indexing(A, ipiv, B)
        chkstride1(A, B, ipiv)
        n = ws.dim
        info = ws.info
        ccall($getrs, Cvoid,
                (Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ptr{$elty}, Ref{BLAS.BlasInt},
                Ptr{BLAS.BlasInt}, Ptr{$elty}, Ref{BLAS.BlasInt}, Ptr{BLAS.BlasInt}, Clong),
                trans, n, size(B,2), A, max(1,stride(A,2)), ipiv, B, max(1,stride(B,2)), info, 1)
        LAPACK.chklapackerror(info[])
    end
end

# ------------------------------------
# Exponential Cone
# ------------------------------------

# gradient and Hessian for the dual barrier function
mutable struct ExponentialCone{T} <: AbstractCone{T}

    H::Matrix{T}       #μ*H for the linear sysmtem
    Hsym::Symmetric{T, Matrix{T}}
    grad::Vector{T}

    # workspace for centrality check
    HBFGS::Matrix{T}
    HBFGSsym::Symmetric{T, Matrix{T}}
    gradWork::Vector{T}
    vecWork::Vector{T}
    z::Vector{T}            # temporary storage for current z
    ws::LuBlasWorkspace{T}

    function ExponentialCone{T}() where {T}

        # PJG: If the dim is hard coded to 3
        # then there should not be a dim field.
        # YC: need to be modified.

        H = Matrix{T}(undef,3,3)
        Hsym = Symmetric(H)
        grad = Vector{T}(undef,3)

        HBFGS = Matrix{T}(undef,3,3)
        HBFGSsym = Symmetric(HBFGS)
        gradWork = Vector{T}(undef,3)
        vecWork = Vector{T}(undef,3)
        z = Vector{T}(undef,3)
        ws = LuBlasWorkspace{T}(3)
        return new(H,Hsym,grad,HBFGS,HBFGSsym,gradWork,vecWork,z,ws)
    end
end

ExponentialCone(args...) = ExponentialCone{DefaultFloat}(args...)

# # ------------------------------------
# # Power Cone
# # ------------------------------------

# gradient and Hessian for the dual barrier function
mutable struct PowerCone{T} <: AbstractCone{T}

    α::T
    H::Matrix{T}       #μ*H for the linear sysmtem
    grad::Vector{T}

    # PJG: internal variables not following
    # variable naming conventions

    # workspace for centrality check
    HBFGS::Matrix{T}
    gradWork::Vector{T}
    vecWork::Vector{T}
    z::Vector{T}            # temporary storage for current z
    ws::LuBlasWorkspace{T}

    function PowerCone{T}(α::T) where {T}

        # PJG: If the dim is hard coded to 3 (should it be?),
        # then there should not be a dim field.

        H = Matrix{T}(undef,3,3)
        grad = Vector{T}(undef,3)



        HBFGS = Matrix{T}(undef,3,3)
        gradWork = Vector{T}(undef,3)
        vecWork = Vector{T}(undef,3)
        z = Vector{T}(undef,3)
        ws = LuBlasWorkspace{T}(3)
        return new(α,H,grad,HBFGS,gradWork,vecWork,z,ws)
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


# set of nonsymmetric cones
const NonsymmetricCones = [ExponentialConeT; PowerConeT]
