push!(LOAD_PATH,"../src/")

using Documenter

using HawkesProcesses ; global const HP = HawkesProcesses

DocMeta.setdocmeta!(HawkesProcesses, :DocTestSetup, :(using HawkesProcesses); recursive=true)

makedocs(;
    modules=[HawkesProcesses],
    authors="Dylan Festa <dylan.festa@gmail.com>",
    repo="https://github.com/dylanfesta/HawkesProcesses.jl/blob/{commit}{path}#{line}",
    sitename="HawkesProcesses.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/HawkesProcesses.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/HawkesProcesses.jl",
    devbranch="main",
)