# -------------------------------------
# Enum and dict for user interface
# -------------------------------------
"""
    SupportedCone
An abstract type use by the Clarabel API used when passing cone specifications to the solver [`setup!`](@ref).
The currently supported concrete types are:

* `ZeroConeT`       : The zero cone.  Used to define equalities.
* `NonnegativeConeT`: The nonnegative orthant.
* `SecondOrderConeT`: The second order / Lorentz / ice-cream cone.
* `PSDTriangleConeT`: The positive semidefinite cone (triangular format).

"""
abstract type SupportedCone end

struct ZeroConeT <: SupportedCone
    dim::DefaultInt
end

struct NonnegativeConeT <: SupportedCone
    dim::DefaultInt
end

struct SecondOrderConeT <: SupportedCone
    dim::DefaultInt
end

struct PSDTriangleConeT <: SupportedCone
    dim::DefaultInt
end

# this reports the number of slack variables that
# will be gnerated by this cone.  Equivalent to
# `numels` for the internal cone representation

function nvars(cone:: SupportedCone)
    if isa(cone, PSDTriangleConeT)
        (cone.dim*(cone.dim+1)) >> 1
    else
        cone.dim
    end
end
