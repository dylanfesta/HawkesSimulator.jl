# Exponential-kernel rate-path benchmarks

The current benchmark compares a self-contained copy of the legacy scalar
rate path with the optimized package implementation. Both variants use the
same deterministic four-population network:

- four dense populations of 200 neurons;
- two excitatory and two inhibitory populations;
- connection magnitude `0.2`, normalized by presynaptic population size;
- no plasticity and no spike recording;
- single-threaded BLAS.

From the repository root, prepare the benchmark environment once:

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Run the comparison with:

```sh
julia --project=benchmarks benchmarks/benchmark_expkernel_rate_path.jl
```

Set `HAWKES_BENCHMARK_SECONDS` to change the time spent on each benchmark.
The script validates numerical and seeded proposal agreement before timing.
Record representative results in `results.md`, including the Julia and BLAS
configuration printed by the script.

The upper-rate correctness question is isolated from the optimization:

```sh
julia benchmarks/investigate_upper_bound.jl
```
