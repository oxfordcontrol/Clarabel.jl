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

@enum SupportedCones begin
    ZeroConeT
    NonnegativeConeT
    SecondOrderConeT
    PSDTriangleConeT
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
)
