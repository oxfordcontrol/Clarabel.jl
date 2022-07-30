# -------------------------------------
# Enum and dict for user interface
# -------------------------------------
"""
    SupportedCone
An abstract type use by the Clarabel API used when passing cone specifications to the solver [`setup!`](@ref). The currently
supported concrete types are:

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
