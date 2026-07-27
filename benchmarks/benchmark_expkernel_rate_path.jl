
##

using BenchmarkTools
using LinearAlgebra
using Random
using Statistics
using HawkesSimulator

const H = HawkesSimulator
include("legacy_expkernel_rate_path.jl")
const Legacy = LegacyExpKernelRatePath

BLAS.set_num_threads(1)

##

const POPULATION_SIZE = 200
const NUM_SPIKES = 5_000
const BENCHMARK_SEED = 20260723

function connection_weights(n_post::Int,n_pre::Int,magnitude::Float64;
    remove_autapses::Bool=false)
  denominator = remove_autapses ? n_pre-1 : n_pre
  weights = fill(magnitude/denominator,n_post,n_pre)
  if remove_autapses
    weights[diagind(weights)] .= 0.0
  end
  return weights
end

function make_network()
  n = POPULATION_SIZE
  states_traces = (
    H.population_state_exp_and_trace(n,0.2;label="e1"),
    H.population_state_exp_and_trace(n,0.2;label="e2"),
    H.population_state_exp_and_trace_inhibitory(n,0.1;label="i1"),
    H.population_state_exp_and_trace_inhibitory(n,0.1;label="i2"),
  )
  states = map(first,states_traces)
  traces = map(last,states_traces)
  population_indices = (1,2,3,4)
  populations = map(population_indices) do idx_post
    conn_pre = map(population_indices) do idx_pre
      weights = connection_weights(n,n,0.2;
        remove_autapses=idx_post==idx_pre)
      (H.ConnectionExpKernel(weights,traces[idx_pre]),states[idx_pre])
    end
    H.PopulationExpKernel(states[idx_post],fill(20.0,n),conn_pre...;
      nonlinearity=H.NLRmax(300.0))
  end
  return H.RecurrentNetworkExpKernel(populations,(H.RecNothing(),))
end

function reset_network!(network)
  H.reset!(network)
  for population in network.populations
    H.set_initial_rates!(population,fill(20.0,H.nneurons(population)))
  end
  return nothing
end

function simulate_optimized!(network,num_spikes::Int,seed::Int)
  reset_network!(network)
  rng = MersenneTwister(seed)
  t_now = 0.0
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(rng,t_now,network)
  end
  return t_now
end

function simulate_legacy!(network,workspaces,num_spikes::Int,seed::Int)
  reset_network!(network)
  rng = MersenneTwister(seed)
  t_now = 0.0
  for _ in 1:num_spikes
    t_now = Legacy.dynamics_step!(rng,t_now,network,workspaces)
  end
  return t_now
end

function validate_implementations!()
  legacy_network = make_network()
  optimized_network = make_network()
  legacy_workspaces = map(Legacy.PopulationWorkspace,legacy_network.populations)
  reset_network!(legacy_network)
  reset_network!(optimized_network)

  legacy_rates = similar(first(legacy_network.populations).spike_proposals)
  optimized_rates = similar(first(optimized_network.populations).spike_proposals)
  Legacy.compute_rates!(legacy_rates,0.01,first(legacy_network.populations),
    first(legacy_workspaces))
  H.compute_rates!(optimized_rates,0.01,first(optimized_network.populations))
  @assert isapprox(legacy_rates,optimized_rates;rtol=1E-12,atol=1E-12)

  Legacy.compute_rates_upper!(legacy_rates,0.01,
    first(legacy_network.populations),first(legacy_workspaces))
  H.compute_rates_upper!(optimized_rates,0.01,first(optimized_network.populations))
  @assert isapprox(legacy_rates,optimized_rates;rtol=1E-12,atol=1E-12)

  legacy_proposal = Legacy.call_for_compute_next_spike(
    MersenneTwister(BENCHMARK_SEED),0.0,
    legacy_network.populations,legacy_workspaces)
  optimized_proposal = H.call_for_compute_next_spike(
    MersenneTwister(BENCHMARK_SEED),0.0,optimized_network.populations)
  @assert legacy_proposal[2:4] == optimized_proposal[2:4]
  @assert isapprox(legacy_proposal[1],optimized_proposal[1];
    rtol=1E-12,atol=1E-12)
  return nothing
end

function build_suite()
  legacy_network = make_network()
  optimized_network = make_network()
  legacy_workspaces = map(Legacy.PopulationWorkspace,legacy_network.populations)
  reset_network!(legacy_network)
  reset_network!(optimized_network)
  legacy_rates = similar(first(legacy_network.populations).spike_proposals)
  optimized_rates = similar(first(optimized_network.populations).spike_proposals)

  suite = BenchmarkGroup()
  suite["rates"] = BenchmarkGroup()
  suite["rates"]["legacy"] = @benchmarkable Legacy.compute_rates!(
    $legacy_rates,0.01,$(first(legacy_network.populations)),
    $(first(legacy_workspaces))) evals=1
  suite["rates"]["optimized"] = @benchmarkable H.compute_rates!(
    $optimized_rates,0.01,$(first(optimized_network.populations))) evals=1

  suite["selector"] = BenchmarkGroup()
  suite["selector"]["legacy"] = @benchmarkable(
    Legacy.call_for_compute_next_spike(rng,0.0,
      $(legacy_network.populations),$legacy_workspaces);
    setup=(rng=MersenneTwister($BENCHMARK_SEED)),evals=1)
  suite["selector"]["optimized"] = @benchmarkable(
    H.call_for_compute_next_spike(rng,0.0,$(optimized_network.populations));
    setup=(rng=MersenneTwister($BENCHMARK_SEED)),evals=1)

  suite["simulation"] = BenchmarkGroup()
  suite["simulation"]["legacy"] = @benchmarkable simulate_legacy!(
    $legacy_network,$legacy_workspaces,$NUM_SPIKES,$BENCHMARK_SEED) evals=1
  suite["simulation"]["optimized"] = @benchmarkable simulate_optimized!(
    $optimized_network,$NUM_SPIKES,$BENCHMARK_SEED) evals=1
  return suite
end

function print_results(results)
  println("| benchmark | implementation | median | minimum | memory | allocations |")
  println("|---|---:|---:|---:|---:|---:|")
  for benchmark in ("rates","selector","simulation")
    for implementation in ("legacy","optimized")
      trial = results[benchmark][implementation]
      median_estimate = median(trial)
      minimum_estimate = minimum(trial)
      println("| ",benchmark," | ",implementation," | ",
        BenchmarkTools.prettytime(median_estimate.time)," | ",
        BenchmarkTools.prettytime(minimum_estimate.time)," | ",
        BenchmarkTools.prettymemory(median_estimate.memory)," | ",
        median_estimate.allocs," |")
    end
    ratio = median(results[benchmark]["legacy"]).time /
      median(results[benchmark]["optimized"]).time
    println("Speedup for ",benchmark,": ",round(ratio;digits=2),"x")
  end
  return nothing
end

validate_implementations!()
suite = build_suite()
seconds = parse(Float64,get(ENV,"HAWKES_BENCHMARK_SECONDS","5"))
results = run(suite;seconds=seconds,verbose=true)
println()
println("Julia: ",VERSION)
println("CPU: ",first(Sys.cpu_info()).model)
println("Julia threads: ",Threads.nthreads())
println("BLAS: ",BLAS.get_config())
println("BLAS threads: ",BLAS.get_num_threads())
println("Topology: 4 populations × 200 neurons (2 excitatory, 2 inhibitory)")
println("Simulation length: ",NUM_SPIKES," spikes")
print_results(results)
