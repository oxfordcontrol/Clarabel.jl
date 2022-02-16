function check_termination!(
    info::DefaultInfo{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultScalings{T},
    settings::Settings{T},
    last_iter::Bool
) where {T}

    #optimality termination check should be computed w.r.t
    #the unscaled x and z variables.
    τinv = 1 / variables.τ

    #primal and dual costs
    info.cost_primal =  residuals.dot_cx*τinv + residuals.dot_xPx * τinv * τinv / 2
    info.cost_dual   = -residuals.dot_bz*τinv - residuals.dot_xPx * τinv * τinv / 2

    #primal and dual residuals
    info.res_primal  = norm(residuals.rx) * τinv / max(1,data.norm_c)
    info.res_dual    = norm(residuals.rz) * τinv / max(1,data.norm_b)

    #absolute and relative gaps
    abs_gap   = residuals.dot_sz * τinv * τinv
    if(info.cost_primal > 0 && info.cost_dual < 0)
        rel_gap = 1/eps()
    else
        rel_gap = abs_gap / min(abs(info.cost_primal),abs(info.cost_dual))
    end

    #κ/τ
    info.ktratio = variables.κ / variables.τ

    #check for convergence
    #---------------------
    if( (abs_gap < settings.tol_gap_abs) || (rel_gap < settings.tol_gap_rel)
        && (info.res_primal < settings.tol_feas)
        && (info.res_dual   < settings.tol_feas)
    )
        info.status = SOLVED

    #check for primal infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(residuals.dot_bz < 0 &&
           residuals.norm_pinf/max(1,data.norm_c) < settings.tol_feas)
        info.status = PRIMAL_INFEASIBLE

    #check for dual infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(residuals.dot_cx < 0 &&
           residuals.norm_dinf/max(1,data.norm_b) < settings.tol_feas)
        info.status = DUAL_INFEASIBLE


    #check for last iteration in the absence
    #of any other reason for stopping
    #----------------------
    elseif(last_iter)
        info.status = MAX_ITERATIONS
    end

    #return TRUE if we settled on a final status
    return is_done = info.status != UNSOLVED

end

function info_save_scalars(info,μ,α,σ,iter)

    info.gap = μ  #PJG: this is not the gap, it's gap/(m+1)
    info.step_length = α
    info.sigma = σ
    info.iterations = iter

    return nothing
end


function info_reset!(info)

    info.status     = UNSOLVED
    info.iterations = 0
    info.solve_time = time()

    return nothing
end

function info_finalize!(info)

    info.solve_time = time() - info.solve_time

    return nothing
end
