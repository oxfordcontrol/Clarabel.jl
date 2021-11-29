function Base.show(io::IO, x::Union{Float64,Float32})
                 Base.Grisu._show(io, round(x, sigdigits=4), Base.Grisu.SHORTEST, 0, get(io, :typeinfo, Any) !== typeof(x), false)
         end

m = solver.data.m
n = solver.data.n
A = solver.data.A
c = solver.data.c
b = solver.data.b

rx = solver.residuals.rx
rz = solver.residuals.rz
rτ = solver.residuals.rτ

x = solver.variables.x
z = solver.variables.z.vec
s = solver.variables.s.vec
λ = solver.variables.λ.vec
τ = solver.variables.τ
κ = solver.variables.κ
w = sqrt.(s./z)
W = diagm(w)
e = ones(length(s))
WinvT = diagm(1. ./ w)
Λ = diagm(λ)
μ = dot(s,z)/(length(s)+1)
α = 0.74853801  #hard coded since the combined step has overwritten the affine one
σ = (1-α)^3

Δxₐ = solver.step_aff.x
Δsₐ = solver.step_aff.s.vec
Δzₐ = solver.step_aff.z.vec
Δκₐ = solver.step_aff.κ
Δτₐ = solver.step_aff.τ

# zero function
Z(m) = zeros(m,m)
Z(m,n) = zeros(m,n)

C = [Z(n)    A'        c;
     -A      Z(m)      b;
     -c'    -b'        0]

K1 = [Z(n)    A'  ;
     -A      W'*W]

K2 = [Z(n)    A'        c;
      -A      W'*W      b;
      -c'    -b'        κ/τ]

#affine step version
dx = rx
dz = rz
dτ = rτ
ds = λ.^2
dκ = κ*τ
#combined step version
dx = (1-σ)*rx
dτ = (1-σ)*rτ
ds = λ.^2 + (WinvT*Δsₐ).*(W*Δzₐ) - σ*μ.*e
dz = -(1-σ)*rz + W*(ds./λ)
dκ = κ*τ + Δκₐ*Δτₐ - σ*μ


# @printf("\n\nFull system, 5x5\n-----------------\n\n")
#
# M = [Z(n)   -A'        -c       Z(n,m)    Z(n,1);
#      A      Z(m)       -b        I(m)     Z(m,1);
#      c'      b'         0       Z(1,m)        1;
#     Z(m,n)  Λ*W        Z(m,1)   Λ*WinvT   Z(m,1);
#     Z(1,n)  Z(1,m)      κ       Z(1,m)       τ   ]
#
# rhs = -[dx;dz;dτ;ds;dκ ]
# print("rhs=",rhs,"\n")
#
# lhs = M\rhs
# lhs = round.(Int,lhs*100)/100
# print("lhs=",lhs,"\n")
# println()
# println("dx=", dx)
# println("dz=", dz)
# println("dτ=", dτ)
# println("ds=", ds)
# println("dκ=", dκ)
# println()
# println("Δx=", lhs[1:n])
# println("Δz=", lhs[(n+1):(n+m)])
# println("Δτ=", lhs[n+m+1])
# println("Δs=", lhs[(n+m+2):(n+m+1+m)])
# println("Δκ=", lhs[end])

# @printf("Reduced system, 3x3\n-----------------\n\n")
#
# M = [Z(n)    A'          c;
#      -A      W*W        b;
#      -c'     -b'         κ/τ]
#
# rhs = [dx; (dz - W*(ds./λ)); dτ - dκ/τ ]
# print("rhs=",rhs,"\n")
#
# lhs = M\rhs
# lhs = round.(Int,lhs*100)/100
# print("lhs=",lhs,"\n")
# println()
# println("Δx=", lhs[1:n])
# println("Δz=", lhs[(n+1):(n+m)])
# println("Δτ=", lhs[n+m+1])
# println("Δs=", -W*(ds./λ + W*lhs[(n+1):(n+m)]))
# println("Δκ=", -(dκ + κ*lhs[n+m+1])/τ)
# println()


@printf("Reduced system, 3x3, 2 phase+sym\n-----------------\n\n")

M = [Z(n)    A';
      A      -W*W ]

rhs1 = [-c; b]
print("rhs1=",rhs1,"\n")

lhs1 = M\rhs1
lhs1 = round.(Int,lhs1*100)/100
print("lhs1=",lhs1,"\n")
println()
Δx₁=lhs1[1:n]
Δz₁=lhs1[(n+1):(n+m)]
println("Δx₁=", Δx₁)
println("Δz₁=", Δz₁)
println()

rhs2 = [dx; dz]
print("rhs2=",rhs2,"\n")

lhs2 = M\rhs2
lhs2 = round.(Int,lhs2*100)/100
print("lhs2=",lhs2,"\n")
println()
Δx₂=lhs2[1:n]
Δz₂=lhs2[(n+1):(n+m)]
println("Δx₂=", Δx₂)
println("Δz₂=", Δz₂)

Δτ  = dτ - dκ/τ + dot(c,Δx₂) + dot(b,Δz₂)
Δτ /= κ/τ - dot(c,Δx₁) - dot(b,Δz₁)
Δx = Δx₂ + Δτ*Δx₁
Δz = Δz₂ + Δτ*Δz₁

Δκ=-(dκ + κ*Δτ)/τ
Δs=-W*(ds./λ + W*Δz)

println()
println("Δx=", Δx)
println("Δz=", Δz)
println("Δτ=", Δτ)
println("Δs=", Δs)
println("Δκ=", Δκ)
println()
