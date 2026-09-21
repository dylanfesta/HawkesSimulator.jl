# Building the documentation

From the repository root, use Julia 1.9 or newer (CI uses Julia 1.12):

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
GKSwstype=100 julia --project=docs docs/make.jl
```

The build executes the examples and doctests. `GKSwstype=100` lets GR render
plots without a display. Open `docs/build/index.html` to view the result.
