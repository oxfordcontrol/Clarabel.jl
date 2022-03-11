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
    #the pre-homogenization x and z variables.
    τinv = 1 / variables.τ

    #shortcuts for the equilibration matrices
    D = scalings.D; Dinv = scalings.Dinv
    E = scalings.E; Einv = scalings.Einv
    cscale = scalings.c[]

    #primal and dual costs. do products are invariant w.r.t
    #equilibration, but we still need to back out the overall
    #objective scaling term c
    info.cost_primal =  (+residuals.dot_qx*τinv + residuals.dot_xPx * τinv * τinv / 2)/cscale
    info.cost_dual   =  (-residuals.dot_bz*τinv - residuals.dot_xPx * τinv * τinv / 2)/cscale

    #primal and dual residuals.   Need to invert the equilibration
    info.res_primal  = scaled_norm(Einv,residuals.rz) * τinv
    info.res_dual    = scaled_norm(Dinv,residuals.rx) * τinv

    #primal and dual infeasibility residuals.   Need to invert the equilibration
    #YC: res_primal_inf & res_dual_inf are similar to pinfres & dinfres in ECOS and CVXOPT, but without normalization
    # res_primal_inf = ||Px+A'z||/(-b'z) < 1e-8,  res_dual_inf = max(||Px||, ||Ax+s||)/(-c'x) < 1e-8
    info.res_primal_inf = scaled_norm(Dinv,residuals.rx_inf)/( - residuals.dot_bz)
    info.res_dual_inf   = max(scaled_norm(Dinv,residuals.Px),scaled_norm(Einv,residuals.rz_inf))/( - residuals.dot_qx)

    #absolute and relative gaps
    gap_abs   = residuals.dot_sz * τinv * τinv
    if(info.cost_primal > 0 && info.cost_dual < 0)
        gap_rel = 1/eps()
    else
        gap_rel = gap_abs / min(abs(info.cost_primal),abs(info.cost_dual))
    end

    #κ/τ
    info.ktratio = variables.κ / variables.τ

    #solve time so far (includes setup!)
    info_get_solve_time!(info)
    is_out_of_time = settings.time_limit == 0. ? false : info.solve_time > settings.time_limit

    #check for convergence
    #---------------------
    if( ((gap_abs < settings.tol_gap_abs) || (gap_rel < settings.tol_gap_rel))
        && (info.res_primal < settings.tol_feas)
        && (info.res_dual   < settings.tol_feas)
    )
        info.status = SOLVED

    #check for primal infeasibility
    #PJG: Still not sure how to properly normalize here
    #maybe should be done via cost.   Using RHS is a disaster
    #---------------------
    #DEBUG: Possibly fatal problem here if norm_q is huge
    #YC: change of res_primal_inf and res_dual_inf in line 33,34 as in ECOS and CVXOPT
elseif(residuals.dot_bz < -1e-6 && variables.τ < variables.κ && info.res_primal_inf < 1e-8)
        info.status = PRIMAL_INFEASIBLE

    #check for dual infeasibility
    #---------------------
    #DEBUG: Fatal problem here if norm_b is huge
    elseif(residuals.dot_qx < -1e-6 && variables.τ < variables.κ && info.res_dual_inf < 1e-8)
        info.status = DUAL_INFEASIBLE


    #check for last iteration in the absence
    #of any other reason for stopping
    #----------------------
    elseif(last_iter)
        info.status = MAX_ITERATIONS
    elseif(is_out_of_time)
        info.status = MAX_TIME
    end

    #return TRUE if we settled on a final status
    return is_done = info.status != UNSOLVED

end

function info_save_scalars(info,μ,α,σ,iter)

    info.gap = μ  #DEBUG PJG: this is not the gap, it's gap/(m+1)
    info.step_length = α
    info.sigma = σ
    info.iterations = iter

    return nothing
end


function info_reset!(info)

    info.status     = UNSOLVED
    info.iterations = 0
    info.solve_time = 0

    #reset the solve! timer, but keep the setup!
    reset_timer!(info.timer["solve!"])

    return nothing
end

function info_get_solve_time!(info)

    #TimerOutputs reports in nanoseconds
    info.solve_time = TimerOutputs.tottime(info.timer)*1e-9
    return nothing
end


function info_finalize!(info)

    info_get_solve_time!(info)
    return nothing
end
