using Documenter

using HawkesSimulator ; global const H = HawkesSimulator

DocMeta.setdocmeta!(HawkesSimulator, :DocTestSetup, :(using HawkesSimulator); recursive=true)

makedocs(;
    modules=[HawkesSimulator],
    authors="Dylan Festa <dylan.festa@gmail.com>",
    repo="https://github.com/dylanfesta/HawkesSimulator.jl/blob/{commit}{path}#{line}",
    sitename="HawkesSimulator.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/HawkesSimulator.jl",
        repolink="https://github.com/dylanfesta/HawkesSimulator.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Exponential kernels" => "exp_1and2D.md",
            "Delayed kernels" => "alphadelay.md",
            "Hawkes and linear networks" => "hawkes_vs_2D_linear.md",
            "Pairwise and triplet STDP" => "plasticity_STDP.md",
            "Rate-dependent STDP" => "plasticity_rate_based.md",
            "Comparing STDP rate components" => "plasticity_STDP_ratecompare.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/HawkesSimulator.jl",
    devbranch="main",
)
