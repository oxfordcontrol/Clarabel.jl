
function debug_print(s)

print("VARIABLES: ")
dump(s.variables,maxdepth=1)

print("RESIDUALS: ")
dump(s.residuals,maxdepth=1)

end

function debug_rescale(s)

vars = s.variables
τ     = vars.τ
κ     = vars.κ
scale = max(τ,κ)

vars.x ./= scale
vars.z.vec ./= scale
vars.s.vec ./= scale
vars.τ /= scale
vars.κ /= scale

end
