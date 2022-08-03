using Pkg

function _get_clarabel_version()
    toml_path = joinpath(@__DIR__,"../Project.toml")
    pkg = Pkg.Types.read_package(toml_path)
    string(pkg.version)
end

const SOLVER_NAME    = "Clarabel"
const SOLVER_VERSION = _get_clarabel_version()
const MOI_VERSION    = "0.10.5"

solver_name() = SOLVER_NAME
version()     = SOLVER_VERSION
moi_version() = MOI_VERSION
