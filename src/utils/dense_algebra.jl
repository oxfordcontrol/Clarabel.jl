# Custom wrappers for BLAS functions to avoid allocations.  Julia eigvals! and svd
# blas interfaces always allocate, even the variants that operate on matrix data in 
# place 

using GenericLinearAlgebra  # extends SVD, eigs etc for BigFloats
import Base: showarg, eltype

#symmetric eigen
const DSYEVR_ = (BLAS.@blasfunc(dsyevr_),Base.liblapack_name)
const SSYEVR_ = (BLAS.@blasfunc(ssyevr_),Base.liblapack_name)

#SVD (divide and conquer method)
const DGESDD_ = (BLAS.@blasfunc(dgesdd_),Base.liblapack_name)
const SGESDD_ = (BLAS.@blasfunc(sgesdd_),Base.liblapack_name)

# -------------------------------------------
# Eigenvalue decomposition / ?syevr variants 
# -------------------------------------------

mutable struct EigEngine{T <: AbstractFloat}  

    # computed eigenvalues in ascending order 
    λ::Vector{T}

    # computed eigenvectors (optional)
    V::Option{Matrix{T}}

    # BLAS workspace (allocated vecs)
    isuppz::Vector{BLAS.BlasInt}
    work::Vector{T}
    iwork::Vector{BLAS.BlasInt}

    # BLAS writeable scalars.  Held here rather than computed on demand as  
    # in rust because the RefValues allocate 
    m::Base.RefValue{BLAS.BlasInt}
    info::Base.RefValue{BLAS.BlasInt}

    function EigEngine{T}(n::Integer) where {T}

        BlasInt = BLAS.BlasInt

        λ      = zeros(T,n)
        V      = nothing

        isuppz = zeros(BlasInt, 2*n)
        work   = ones(T, 1)
        iwork  = ones(BlasInt, 1)

        m      = Ref{BlasInt}()
        info   = Ref{BlasInt}()

        new(λ,V,isuppz,work,iwork,m,info)
    end
end

EigEngine(n) = EigEngine{Float64}(n)

for (xsyevr, elty) in
    ((DSYEVR_,:Float64),
     (SSYEVR_,:Float32))
   @eval begin
        function xsyevr!(engine::EigEngine{$elty},A::Matrix{$elty}, jobz::Char)

            LinearAlgebra.checksquare(A)

            An       = size(A,1)

            # allocate for eigenvectors on first request
            if jobz == 'V' && isnothing(engine.V)
                engine.V = zeros($elty,An,An)
            end 

            range  = 'A'  # compute all eigenvalues
            uplo   = 'U'  # always assume triu form 
            n      = An   # matrix dimension 
            a      = A    # matrix data 
            lda    = n  
            vl     = zero($elty)   # eig value lb (range = A => not used
            vu     = zero($elty)   # eig value ub (range = A => not used)
            il     = BLAS.BlasInt(0)  # eig interval lb (range = A => not used)
            iu     = BLAS.BlasInt(0)  # eig interval ub (range = A => not used)
            abstol  = $elty(-1)    # forces default tolerance
            m      = engine.m  # returns number of computed eigenvalues
            w      = engine.λ  # eigenvalues go here 
            ldz    = n   # leading dimension of eigenvector matrix 
            isuppz = engine.isuppz  
            work   = engine.work 
            lwork  = BLAS.BlasInt(-1)  # -1 => config to request required work size
            iwork  = engine.iwork 
            liwork = BLAS.BlasInt(-1) # -1 => config to request required work size
            info   = engine.info

            # target for computed eigenvectors (if any)
            # PG: assigning `work` as a placeholder target for V is perhaps iffy, 
            # but should be OK because we aren't writing to the vectors anyway.
            # Doing this instead of putting a placeholder [0.] because the 
            # placeholder vec is allocating  
            z = isnothing(engine.V) ? work : engine.V 

            for i = 0:1

                ccall($xsyevr, Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt},
                Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ref{$elty},
                Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{BLAS.BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt}, Ptr{BLAS.BlasInt},
                Ptr{$elty}, Ref{BLAS.BlasInt}, Ptr{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                Ptr{BLAS.BlasInt}),
                jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z,ldz,
                isuppz,work,lwork,iwork,liwork,info
                )

                LAPACK.chklapackerror(info[])
         
                if i == 0 
                    lwork = BLAS.BlasInt(work[1]);
                    liwork = iwork[1];
                    resize!(work, lwork);
                    resize!(iwork, liwork);
                end 
            end 
        end
    end #@eval
end #for

#BLAS types use the engine 

function eigvals!(engine::EigEngine{T},A::Matrix{T}) where{T <: BLAS.BlasReal}
    xsyevr!(engine,A,'N')
end 

function eigen!(engine::EigEngine{T},A::Matrix{T}) where{T <: BLAS.BlasReal}
    xsyevr!(engine,A,'V')
end 

# non-BLAS types ignore the engine and compute eigvals directly

function eigvals!(engine::EigEngine{T},A::Matrix{T}) where{T <: AbstractFloat}
    engine.λ = GenericLinearAlgebra.eigvals!(Hermitian(A))
end 

function eigen!(engine::EigEngine{T},A::Matrix{T}) where{T <: AbstractFloat}
    F = GenericLinearAlgebra.eigen!(Hermitian(A))
    copyto!(engine.λ, F.values)
    copyto!(engine.V, F.vectors)
end 



# -------------------------------------------
# SVD (divide and conquer method only) 
# -------------------------------------------


mutable struct SVDEngine{T <: AbstractFloat}  

    # Computed singular values
    s::Vector{T}

    # Left and right SVD matrices, each containing.
    # min(m,n) vectors.  Note right singular vectors
    # are stored in transposed form.
    U::Matrix{T}
    Vt::Matrix{T}

    # BLAS workspace (allocated vecs only)
    work::Vector{T}
    iwork::Vector{BLAS.BlasInt}

    # BLAS writeable scalars.  
    info::Base.RefValue{BLAS.BlasInt}

    function SVDEngine{T}(dim::Tuple{Ti, Ti}) where {T,Ti <: Integer}

        (m,n) = dim
        s  = zeros(T, min(m,n))
        U  = zeros(T, m, min(m,n))
        Vt = zeros(T, min(m,n), n)
        work =  [one(T)]
        iwork = ones(BLAS.BlasInt,8 * min(m, n))
        info   = Ref{BLAS.BlasInt}()

        new(s,U,Vt,work,iwork,info)
    end
end

SvdEngine(dim) = SvdEngine{Float64}(dim)

for (xgesdd, elty) in
    ((DGESDD_,:Float64),
     (SGESDD_,:Float32))
   @eval begin
        function xgesdd!(engine::SVDEngine{$elty},A::Matrix{$elty})

            (m,n) = size(A)

            size(engine.U)[1]  == m || throw(DimensionMismatch("Inconsistent internal state."))
            size(engine.Vt)[2] == n || throw(DimensionMismatch("Inconsistent internal state."))

            # standard BLAS ?gesdd and/or ?gesvd arguments for economy size SVD.

            job = 'S' # compact.
            a = A
            lda = m
            s = engine.s # singular values go here
            u = engine.U # U data goes here
            ldu = m      # leading dim of U
            vt = engine.Vt # Vt data goes here
            ldvt = min(m, n) # leading dim of Vt
            work = engine.work
            lwork = BLAS.BlasInt(-1) # -1 => config to request required work size
            iwork = engine.iwork
            info = engine.info; # output info

            for i = 0:1

                # Two calls to BLAS. First one gets size for work.

                ccall($xgesdd, Cvoid,
                (Ref{UInt8}, #job 
                Ref{BLAS.BlasInt},#m
                Ref{BLAS.BlasInt},#n
                Ptr{$elty}, #A
                Ref{BLAS.BlasInt}, #lda
                Ptr{$elty}, #s
                Ptr{$elty}, #u
                Ref{BLAS.BlasInt}, #ldu
                Ptr{$elty}, #vt
                Ref{BLAS.BlasInt}, #ldvt
                Ptr{$elty}, #work
                Ref{BLAS.BlasInt}, #lwork
                Ptr{BLAS.BlasInt}, #iwork
                Ptr{BLAS.BlasInt}  #info
                ),
                job, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info, 
                )

                LAPACK.chklapackerror(info[])
         
                if i == 0 
                    lwork = BLAS.BlasInt(work[1]);
                    resize!(work, lwork);
                end 
            end 
        end
    end #@eval
end #for

function factor!(engine::SVDEngine{T},A::Matrix{T}) where {T <: BLAS.BlasReal}
    xgesdd!(engine,A)
end 

function factor!(engine::SVDEngine{T},A::Matrix{T}) where {T <: AbstractFloat}
    F = GenericLinearAlgebra.svd!(A)
    copyto!(engine.U, F.U)
    copyto!(engine.s, F.S)
    copyto!(engine.Vt, F.Vt)
end 