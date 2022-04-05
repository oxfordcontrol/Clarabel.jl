is_symmetric(::ExponentialCone{T}) where {T} = false

function update_scaling!(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T
) where {T}

    st = K.st
    zt = K.zt
    q = K.q
    r = K.r
    Hs = K.Hs
    δs = K.δs
    δz = K.δz


    #st = -F*'(z), zt = -F'(s)
    DualGradF(st, z)
    st .*= -1
    GradF(zt, s)
    zt .*= -1

    CrossProduct!(q,s,st)   #compute q =( s ⨱ st)/|| s ⨱ st||
    normalize!(q)
    CrossProduct!(r,z,zt)   #compute r = (z ⨱ zt)/<z ⨱ zt, q>
    r ./= dot(r,q)

    #YC: temporarily, we compute \tilde{μ} for each cone separately rather than globally
    μt = dot(st,zt)
    HessianF(Hs,s)          #compute Hessian at s
    mul!(δs,Hs,st)          #δs,δz as an intermediate variable
    δz .= δs - μt*zt
    t = μ*norm(Hs - zt*zt'/numel(K) - δz*δz'/(dot(st,δs) - numel(K)*μt^2))

    # δs = s - μ*st, δz = z - μ*zt
    δs .= s - μ*st
    δz .= z - μ*zt


end


###################################
# 1st-3rd order information for exponential cones
###################################

# YC: In the future, we could possibly combine the follwing operators to avoid repeated computation

# gradient g of the log-barrier F(x) at the point x
function GradF(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];

    div = x2/x1
    logdiv = log(div)
    ψ = -x2*logdiv - x3

    #compute gradient -ψ'(x)/ψ(x)
    g .= [div; -logdiv-1; -1];
    g ./= -ψ;

    #then +h'(x)
    g .+= [-1/x1; -1/x2; 0];
end

# Hessian of F(x)
function HessianF(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];

    div = x2/x1
    logdiv = log(div)
    ψ = -x2*logdiv - x3

    #1st and 2nd order infromation of ψ(x)
    ∂ψ = [div; -logdiv-1; -1];
    hat∂ψ = @view ∂ψ[1:2]
    v = [sqrt(x2)/x1; -1/sqrt(x2)]

    #compute the Hessian
    A = ψ^2*Diagonal([1/x1^2; 1/x2^2]) + ψ*v*v'
    H .= [(A + hat∂ψ*hat∂ψ') -hat∂ψ; -hat∂ψ' 1]/(ψ^2)
end

#3rd-order correction (need verification)
function HigherCorrection(η::AbstractVector{T}, x::AbstractVector{T}, u::AbstractVector{T}, v::AbstractVector{T}) where {T}
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];

    u1 = u[1];
    u2 = u[2];
    u3 = u[3];
    v1 = v[1];
    v2 = v[2];
    v3 = v[3];

    div = x2/x1
    logdiv = log(div)
    ψ = -x2*logdiv - x3

    #1st and 2nd order infromation of ψ(x)
    ∂ψ = [div; -logdiv-1; -1];          #ψ'(x)
    hat∂ψ = @view ∂ψ[1:2]
    vx = [sqrt(x2)/x1; -1/sqrt(x2); 0.0]   #ψ''(x) = vx*vx'

    #compute the Hessian
    Hg = ψ*vx*vx' + [hat∂ψ*hat∂ψ' -hat∂ψ; -hat∂ψ' 1]./(ψ^2)     #Hessian g''(x)

    #compute 3rd-order correction
    ∂ψu = dot(∂ψ,u)
    ∂ψv = dot(∂ψ,v)
    vxv = dot(vx,v)
    vxu = dot(u,vx)
    sqx1 = x1^2
    sqx2 = x2^2
    div1 = u1/sqx1
    div2 = u2/sqx1
    div3 = u2/sqx2
    η .= -2*∂ψu*Hg*v/ψ + ∂ψu*vxv*vx/(ψ^2) - [(2*div*div1 - div2)*v1 - div1*v2; -div1*v1 + div3*v2; 0.0]/ψ
            + (-vxu*vxv*∂ψ - vxu*∂ψv*vx)/(ψ^2) -2*[div1*v1/x1; div3*v2/x2; 0.0]
    η .*= -0.5
end
#For the veryfication, we use H = x -> ForwardDiff.hessian(f,x), where f(x) = -log(x[2]*log(x[1]/x[2])-x[3])-log(x[1])-log(x[2])



#compute the dual gradient F*'(x)
# where F*(x) = -2*log(-x3) - log(x1) - log((1-barω)^2/barω) - 3, barω = ω(2 - x2/x3 - log(-x3)+log(x1))
# Instead of the damped Newton method, we use the method in Santiago's thesis to compute the dual gradient, "Algorithms for Unsymmetric Cone Optimization and an Implementation for Problems with the Exponential Cone"
#NB: Numerical accuracy (uncertain)
# YC:the dualgradient computed here is about 1e-5 numerical error to the dualgradient computed via the Fenchel duality definition
function DualGradF(g::AbstractVector{T}, x::AbstractVector{T}) where{T}
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];

    β = 2.0 - x2/x3 - log(-x3) + log(x1)
    barω = WrightOmega(β)
    coef = 1.0/(barω-1)
    g[1] = -1/x1*(1 + coef)
    g[2] = coef/x3
    g[3] = -coef*(x2/(x3^2) - 1.0/x3)
end

#Cross product of 3-dim vectors
function CrossProduct!(q,a,b)
    q[1] = a[2]*b[3] - a[3]*b[2]
    q[2] = a[3]*b[1] - a[1]*b[3]
    q[3] = a[1]*b[2] - a[2]*b[1]
end
