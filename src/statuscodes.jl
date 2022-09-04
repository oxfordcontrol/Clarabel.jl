# ---------------
# solver status
# ---------------
"""
    SolverStatus
An Enum of of possible conditions set by [`solve!`](@ref).

If no call has been made to [`solve!`](@ref), then the `SolverStatus`
is:
* `UNSOLVED`: The algorithm has not started.

Otherwise:
* `SOLVED`                      : Solver terminated with a solution.
* `PRIMAL_INFEASIBLE`           : Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
* `DUAL_INFEASIBLE`             : Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
* `ALMOST_SOLVED`               : Solver terminated with a solution (reduced accuracy).
* `ALMOST_PRIMAL_INFEASIBLE`    : Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility (reduced accuracy).
* `ALMOST_DUAL_INFEASIBLE`      : Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility (reduced accuracy).
* `MAX_ITERATIONS`              : Iteration limit reached before solution or infeasibility certificate found.
* `MAX_TIME`                    : Time limit reached before solution or infeasibility certificate found.
* `NUMERICAL_ERROR`             : Solver terminated with a numerical error.
* `INSUFFICIENT_PROGRESS`       : Solver terminated due to lack of progress.
"""
@enum SolverStatus begin
    UNSOLVED           = 0
    SOLVED
    PRIMAL_INFEASIBLE
    DUAL_INFEASIBLE
    ALMOST_SOLVED
    ALMOST_PRIMAL_INFEASIBLE
    ALMOST_DUAL_INFEASIBLE
    MAX_ITERATIONS
    MAX_TIME
    NUMERICAL_ERROR
    INSUFFICIENT_PROGRESS
end

const SolverStatusDict = Dict(
    UNSOLVED            =>  "unsolved",
    SOLVED              =>  "solved",
    PRIMAL_INFEASIBLE   =>  "primal infeasible",
    DUAL_INFEASIBLE     =>  "dual infeasible",
    ALMOST_SOLVED       =>  "solved (reduced accuracy)",
    ALMOST_PRIMAL_INFEASIBLE   =>  "primal infeasible (reduced accuracy)",
    ALMOST_DUAL_INFEASIBLE     =>  "dual infeasible (reduced accuracy)",
    MAX_ITERATIONS      =>  "iteration limit",
    MAX_TIME            =>  "time limit",
    NUMERICAL_ERROR     =>  "numerical error",
    INSUFFICIENT_PROGRESS =>  "insufficient progress"
)

function status_is_infeasible(status::SolverStatus)
    #status is any of the infeasible codes 
    return (
        status == PRIMAL_INFEASIBLE || 
        status == DUAL_INFEASIBLE   ||
        status == ALMOST_PRIMAL_INFEASIBLE || 
        status == ALMOST_DUAL_INFEASIBLE
    )
end 

function status_is_errored(status::SolverStatus)
    #status is any of the error codes 
    return (
        status == NUMERICAL_ERROR || 
        status == INSUFFICIENT_PROGRESS
    )
end 