using Documenter, DocumenterTools, Clarabel, Literate

# Force use of dev QDLDL package
using Pkg
Pkg.dev("QDLDL")
Pkg.status("QDLDL")

# This file borrows heavily from the one in COSMO.jl

@info "Building example problems..."

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")
fix_suffix(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
      """
      The source files for all examples can be found in [/examples](https://github.com/oxfordcontrol/Clarabel.jl/tree/main/examples/).
      """ * cont
end


# find all example source files
exclude_files = String[];
example_path = joinpath(@__DIR__, "../examples/")
build_path =  joinpath(@__DIR__, "src", "examples/")
files = readdir(example_path)
filter!(x -> endswith(x, ".jl"), files)
filter!(x -> !in(x, exclude_files), files)

for file in files
      Literate.markdown(example_path * file, build_path; preprocess = fix_math_md, postprocess = postprocess, documenter = true, credit = true)
end

examples_nav = fix_suffix.("./examples/" .* files)

# find all other documentation source files that are built with Literate
example_path = joinpath(@__DIR__, "src", "literate/")
build_path =  joinpath(@__DIR__, "src", "literate", "build/")
files = readdir(example_path)
filter!(x -> endswith(x, ".jl"), files)
for file in files
      Literate.markdown(example_path * file, build_path; preprocess = fix_math_md, documenter = true, credit = true)
end


@info "Making documentation..."
makedocs(
  sitename="Clarabel.jl",
  authors = "Paul Goulart",
  format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://oxfordcontrol.github.io/Clarabel.jl/stable/",
        assets = ["assets/favicon.ico"; "assets/github_buttons.js"; "assets/custom.css"],
        analytics = "G-FP3WPEJMVX",
  ),
  pages = [
        "Home" => "index.md",
        "User Guide" => Any[
        "Getting Started" =>  "getting_started.md",
        "JuMP Interface" => "jump.md",
        "Convex.jl Interface" => "./literate/build/convex_jl.md",
        "Arbitrary Precision Arithmetic" => "./literate/build/arbitrary_precision.md",
        "Linear Solvers" => "linear_solvers.md",
        ],
        #"Method" => "method.md",
        "Examples" => examples_nav,
        "Citing Clarabel" => "citing.md",
        "Contributing" => "contributing.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    devbranch = "main",
    devurl    = "dev",
    repo      = "github.com/oxfordcontrol/Clarabel.jl.git")
