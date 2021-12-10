function residuals_update!(
    residuals::DefaultResiduals{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where {T}

  #scalars used locally more than once
  cx        = dot(data.c,variables.x)
  bz        = dot(data.b,variables.z.vec)
  sz        = dot(variables.s.vec,variables.z.vec)

  #partial residual calc so I can catch the
  #norms of the matrix vector products
  residuals.rx = data.A'* variables.z.vec
  residuals.rz = data.A * variables.x

  #matrix vector product norm (scaled)
  residuals.norm_Ax  = norm(residuals.rz)
  residuals.norm_Atz = norm(residuals.rx)

  #finish the residual calculation
  residuals.rx .= -residuals.rx - data.c * variables.τ
  residuals.rz .= +residuals.rz - data.b * variables.τ + variables.s.vec
  residuals.rτ = cx + bz + variables.κ

  #relative residuals (scaled)
  residuals.norm_rz     = norm(residuals.rz)
  residuals.norm_rx     = norm(residuals.rx)

  #various dot products for later use (all with scaled variables)
  residuals.dot_cx = cx
  residuals.dot_bz = bz
  residuals.dot_sz = sz

  return nothing
end
