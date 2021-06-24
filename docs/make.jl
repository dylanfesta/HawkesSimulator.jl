using HawkesSimulator
using Documenter

DocMeta.setdocmeta!(HawkesSimulator, :DocTestSetup, :(using HawkesSimulator); recursive=true)

makedocs(;
    modules=[HawkesSimulator],
    authors="Dylan Festa <dylan.festa@gmail.com>",
    repo="https://github.com/dylanfesta/HawkesSimulator.jl/blob/{commit}{path}#{line}",
    sitename="HawkesSimulator.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/HawkesSimulator.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/HawkesSimulator.jl",
    devbranch="main",
)
