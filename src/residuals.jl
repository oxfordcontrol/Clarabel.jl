function residuals_update!(
    residuals::DefaultResiduals{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where {T}

  # various products used multiple times
  qx  = dot(data.q,variables.x)
  bz  = dot(data.b,variables.z)
  sz  = dot(variables.s,variables.z)
  mul!(residuals.Px,data.Psym,variables.x)
  xPx = dot(variables.x,residuals.Px)

  #partial residual calc so we can check primal/dual
  #infeasibility conditions

  #Same as:
<<<<<<< HEAD
  #residuals.rx_inf .= - data.Psym,variables.x -data.A'* variables.z
  mul!(residuals.rx_inf, data.A', variables.z, -1., 0.)
=======
  #residuals.rx_inf .= -data.A'* variables.z - data.Psym*variables.x
  mul!(residuals.rx_inf, data.A', variables.z, -one(T), zero(T))
>>>>>>> 11812b037ae885122a0eae8a9048a0a6f69bcadf
  residuals.rx_inf .-= residuals.Px

  #Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
  @. residuals.rz_inf = variables.s
  mul!(residuals.rz_inf, data.A, variables.x, one(T), one(T))

  #complete the residuals
  @. residuals.rx = residuals.rx_inf - data.q * variables.τ
  @. residuals.rz = residuals.rz_inf - data.b * variables.τ
  residuals.rτ    = qx + bz + variables.κ + xPx/variables.τ

  #save local versions
  residuals.dot_qx  = qx
  residuals.dot_bz  = bz
  residuals.dot_sz  = sz
  residuals.dot_xPx = xPx

  return nothing
end
