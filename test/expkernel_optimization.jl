@testset "Exponential-kernel trace handling" begin
  τ = 0.5
  trace = H.Trace(τ,2,H.ForDynamics())
  trace.val .= [2.0,4.0]
  trace.t_last = 0.25

  @test !hasfield(typeof(trace),:purpose)
  @test isapprox(H.trace_decay(0.75,trace),exp(-1.0);rtol=1E-14)

  proposal = fill(NaN,2)
  H.trace_proposal!(proposal,0.75,trace)
  @test all(isapprox.(proposal,[2.0,4.0].*exp(-1.0);rtol=1E-14))

  state = H.PopulationStateExpKernel(2,trace;label="trace_test")
  values_before = copy(trace.val)
  H.burn_spike!(0.75,state)
  @test trace.val == values_before
  @test trace.t_last == 0.25

  H.burn_spike!(0.75,state,2)
  @test isapprox(trace.val[1],2.0exp(-1.0);rtol=1E-14)
  @test isapprox(trace.val[2],4.0exp(-1.0)+inv(τ);rtol=1E-14)
  @test trace.t_last == 0.75
end

@testset "Vectorized exponential-kernel rates" begin
  post_state,_ = H.population_state_exp_and_trace(2,0.4;label="post")
  pre_e,trace_e = H.population_state_exp_and_trace(3,0.2;label="pre_e")
  pre_i,trace_i = H.population_state_exp_and_trace_inhibitory(2,0.1;label="pre_i")

  trace_e.val .= [1.0,2.0,4.0]
  trace_i.val .= [3.0,5.0]
  trace_e.t_last = 0.1
  trace_i.t_last = 0.15

  weights_e = [0.2 0.1 0.3; 0.4 0.2 0.1]
  weights_i = [0.5 0.1; 0.2 0.6]
  conn_e = H.ConnectionExpKernel(weights_e,trace_e)
  conn_i = H.ConnectionExpKernel(weights_i,trace_i)
  conn_none = H.ConnectionNonInteracting(zeros(2,3))

  @test !hasfield(typeof(conn_e),:trace_proposal)
  @test_throws AssertionError H.ConnectionExpKernel(zeros(2,2),trace_e)

  input = [2.0,3.0]
  pop = H.PopulationExpKernel(post_state,input,
    (conn_e,pre_e),(conn_i,pre_i),(conn_none,pre_e))
  t_now = 0.3
  decay_e = exp(-(t_now-trace_e.t_last)/trace_e.τ)
  decay_i = exp(-(t_now-trace_i.t_last)/trace_i.τ)
  expected_raw = input + decay_e*weights_e*trace_e.val -
    decay_i*weights_i*trace_i.val .+ 1E-9
  expected = max.(expected_raw,0.0)

  contribution = zeros(2)
  H.accumulate_signal!(contribution,t_now,post_state,conn_e,pre_e)
  @test all(isapprox.(contribution,decay_e*weights_e*trace_e.val;
    rtol=1E-12,atol=1E-12))
  H.accumulate_signal!(contribution,t_now,post_state,conn_i,pre_i)
  @test all(isapprox.(contribution,
    decay_e*weights_e*trace_e.val-decay_i*weights_i*trace_i.val;
    rtol=1E-12,atol=1E-12))
  H.accumulate_signal!(contribution,t_now,post_state,conn_none,pre_e)
  @test all(isapprox.(contribution,
    decay_e*weights_e*trace_e.val-decay_i*weights_i*trace_i.val .+ 1E-9;
    rtol=1E-12,atol=1E-12))

  rates = similar(input)
  H.compute_rates!(rates,t_now,pop)
  @test all(isapprox.(rates,expected;rtol=1E-12,atol=1E-12))
  scalar_rates = [H.compute_rate(t_now,input[i],pop,i) for i in eachindex(input)]
  @test all(isapprox.(rates,scalar_rates;rtol=1E-12,atol=1E-12))

  upper_rates = similar(input)
  H.compute_rates_upper!(upper_rates,t_now,pop)
  expected_upper = max.(max.(input,eps(Float64)),expected)
  @test all(isapprox.(upper_rates,expected_upper;rtol=1E-12,atol=1E-12))
  scalar_upper = [H.compute_rate_upper(t_now,input[i],pop,i) for i in eachindex(input)]
  @test all(isapprox.(upper_rates,scalar_upper;rtol=1E-12,atol=1E-12))

  empty_state,_ = H.population_state_exp_and_trace(0,0.2;label="empty")
  empty_pop = H.PopulationExpKernel(empty_state,Float64[])
  empty_rates = Float64[]
  @test H.compute_rates!(empty_rates,0.0,empty_pop) === nothing
  @test isempty(empty_rates)
end

@testset "RNG-aware exponential-kernel dynamics" begin
  function make_rng_test_network()
    state_e,trace_e = H.population_state_exp_and_trace(3,0.2;label="rng_e")
    state_i,trace_i = H.population_state_exp_and_trace_inhibitory(2,0.1;label="rng_i")
    conn_ee = H.ConnectionExpKernel(zeros(3,3),trace_e)
    conn_ie = H.ConnectionExpKernel(zeros(2,3),trace_e)
    conn_ei = H.ConnectionExpKernel(zeros(3,2),trace_i)
    conn_ii = H.ConnectionExpKernel(zeros(2,2),trace_i)
    pop_e = H.PopulationExpKernel(state_e,[10.0,20.0,30.0],
      (conn_ee,state_e),(conn_ei,state_i))
    pop_i = H.PopulationExpKernel(state_i,[15.0,25.0],
      (conn_ie,state_e),(conn_ii,state_i))
    return H.RecurrentNetworkExpKernel((pop_e,pop_i),(H.RecNothing(),))
  end

  network_a = make_rng_test_network()
  network_b = make_rng_test_network()
  rng_a = MersenneTwister(123)
  rng_b = MersenneTwister(123)
  proposal_a = H.call_for_compute_next_spike(rng_a,0.0,network_a.populations)
  proposal_b = H.call_for_compute_next_spike(rng_b,0.0,network_b.populations)
  @test proposal_a == proposal_b
  @test proposal_a[3] isa Symbol
  @test rand(rng_a) == rand(rng_b)

  H.reset!(network_a)
  H.reset!(network_b)
  rng_a = MersenneTwister(456)
  rng_b = MersenneTwister(456)
  t_a = H.dynamics_step!(rng_a,0.0,network_a)
  t_b = H.dynamics_step!(rng_b,0.0,network_b)
  @test t_a == t_b
  for (pop_a,pop_b) in zip(network_a.populations,network_b.populations)
    @test pop_a.state.traces[1].val == pop_b.state.traces[1].val
    @test pop_a.state.traces[1].t_last == pop_b.state.traces[1].t_last
  end

  Random.seed!(789)
  default_a = H.call_for_compute_next_spike(0.0,network_a.populations)
  Random.seed!(789)
  default_b = H.call_for_compute_next_spike(0.0,network_b.populations)
  @test default_a == default_b

  single_source_a = make_rng_test_network()
  single_source_b = make_rng_test_network()
  single_a = H.RecurrentNetworkExpKernel(
    (first(single_source_a.populations),),(H.RecNothing(),))
  single_b = H.RecurrentNetworkExpKernel(
    (first(single_source_b.populations),),(H.RecNothing(),))
  t_single_a = H.dynamics_step_singlepopulation!(
    MersenneTwister(987),0.0,single_a)
  t_single_b = H.dynamics_step_singlepopulation!(
    MersenneTwister(987),0.0,single_b)
  @test t_single_a == t_single_b
  @test single_a.populations[1].state.traces[1].val ==
    single_b.populations[1].state.traces[1].val

  forced_trains = [[0.1,0.2],[0.15,0.25]]
  mixed_trace = H.Trace(0.2,2,H.ForDynamics())
  mixed_state = H.PopulationStateMixedExp(forced_trains,mixed_trace)
  mixed_conn = H.ConnectionExpKernel(zeros(2,2),mixed_trace)
  mixed_pop = H.PopulationMixedExp(mixed_state,mixed_conn,[1.0,1.0])
  mixed_a = H.compute_next_spike(MersenneTwister(321),0.0,mixed_pop)
  mixed_b = H.compute_next_spike(MersenneTwister(321),0.0,mixed_pop)
  @test mixed_a == mixed_b
end
