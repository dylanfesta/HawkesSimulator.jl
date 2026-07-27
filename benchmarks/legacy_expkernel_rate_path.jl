module LegacyExpKernelRatePath

using LinearAlgebra
using Random
using HawkesSimulator

const H = HawkesSimulator

struct PopulationWorkspace{T}
  trace_proposals::T
end

function PopulationWorkspace(pop)
  proposals = map(pop.connections) do conn
    if conn isa H.ConnectionExpKernel
      Vector{Float64}(undef,size(conn.weights,2))
    else
      nothing
    end
  end
  return PopulationWorkspace{typeof(proposals)}(proposals)
end

function trace_proposal!(proposal::Vector{Float64},t_now::Real,
    conn::H.ConnectionExpKernel)
  copyto!(proposal,conn.pre_trace.val)
  rmul!(proposal,exp(-(t_now-conn.pre_trace.t_last)/conn.pre_trace.τ))
  return nothing
end

function propagated_signal(t_now::Real,idx_post::Integer,
    conn::H.ConnectionExpKernel,::H.PopulationStateMarkovian,
    proposal::Vector{Float64})
  trace_proposal!(proposal,t_now,conn)
  return dot(view(conn.weights,idx_post,:),proposal)
end

function propagated_signal(t_now::Real,idx_post::Integer,
    conn::H.ConnectionExpKernel,::H.PopulationStateExpKernelInhibitory,
    proposal::Vector{Float64})
  trace_proposal!(proposal,t_now,conn)
  return -dot(view(conn.weights,idx_post,:),proposal)
end

function propagated_signal(::Real,::Integer,::H.ConnectionNonInteracting,
    ::H.PopulationStateMarkovian,::Nothing)
  return 1E-9
end

function compute_signal(ret,t_now,idx_post,connections,pre_states,proposals)
  ret += propagated_signal(t_now,idx_post,first(connections),
    first(pre_states),first(proposals))
  return compute_signal(ret,t_now,idx_post,Base.tail(connections),
    Base.tail(pre_states),Base.tail(proposals))
end
function compute_signal(ret,::Real,::Integer,::Tuple{},::Tuple{},::Tuple{})
  return ret
end

function compute_rate(t_now::Real,external_input::Float64,pop,idx_post::Integer,
    workspace::PopulationWorkspace)
  rate = compute_signal(external_input,t_now,idx_post,pop.connections,
    pop.pre_states,workspace.trace_proposals)
  return H.apply_nonlinearity(rate,pop.nonlinearity)
end

function compute_rate_upper(t_now::Real,external_input::Float64,pop,
    idx_post::Integer,workspace::PopulationWorkspace)
  external_input_nz = max(external_input,eps(Float64))
  rate = compute_signal(external_input_nz,t_now,idx_post,pop.connections,
    pop.pre_states,workspace.trace_proposals)
  return max(external_input_nz,H.apply_nonlinearity(rate,pop.nonlinearity))
end

function compute_rates!(rates::Vector{Float64},t_now::Real,pop,
    workspace::PopulationWorkspace)
  for idx_post in eachindex(rates)
    @inbounds rates[idx_post] =
      compute_rate(t_now,pop.input[idx_post],pop,idx_post,workspace)
  end
  return nothing
end

function compute_rates_upper!(rates::Vector{Float64},t_now::Real,pop,
    workspace::PopulationWorkspace)
  for idx_post in eachindex(rates)
    @inbounds rates[idx_post] =
      compute_rate_upper(t_now,pop.input[idx_post],pop,idx_post,workspace)
  end
  return nothing
end

function compute_next_spike(rng::AbstractRNG,t_now::Real,pop,
    workspace::PopulationWorkspace;Tmax::Real=100.0)
  t_start = t_now
  t = t_now
  rates = pop.spike_proposals
  while (t-t_start)<Tmax
    compute_rates_upper!(rates,t,pop,workspace)
    rate_upper = sum(rates)
    if iszero(rate_upper)
      break
    end
    t += -log(rand(rng))/rate_upper
    uniform_rate = rand(rng)*rate_upper
    compute_rates!(rates,t,pop,workspace)
    cumsum!(rates,rates)
    if uniform_rate < rates[end]
      return (t,searchsortedfirst(rates,uniform_rate))
    end
  end
  return (Tmax+t_start,1)
end

function call_for_compute_next_spike(rng::AbstractRNG,t_now,populations,workspaces,
    currentpop,bestspiketime,bestpop,bestpoplabel,bestneuron)
  pop = first(populations)
  workspace = first(workspaces)
  currentpop += 1
  bestspiketime_here,bestneuron_here =
    compute_next_spike(rng,t_now,pop,workspace)
  if bestspiketime_here < bestspiketime
    bestspiketime = bestspiketime_here
    bestpop = currentpop
    bestpoplabel = pop.state.label
    bestneuron = bestneuron_here
  end
  tailpopulations = Base.tail(populations)
  if isempty(tailpopulations)
    return (bestspiketime,bestpop,bestpoplabel,bestneuron)
  end
  return call_for_compute_next_spike(rng,t_now,tailpopulations,
    Base.tail(workspaces),currentpop,bestspiketime,bestpop,bestpoplabel,bestneuron)
end

function call_for_compute_next_spike(rng::AbstractRNG,t_now,populations,workspaces)
  return call_for_compute_next_spike(rng,t_now,populations,workspaces,
    0,Inf,-1,-1,-1)
end

function burn_spikes!(currentpop::Integer,tfire,popfire,neufire,populations)
  currentpop += 1
  population = first(populations)
  if currentpop == popfire
    H.burn_spike!(tfire,population.state,neufire)
  else
    for trace in population.state.traces
      H.propagate_for_dynamics!(tfire,trace)
    end
  end
  tailpopulations = Base.tail(populations)
  if !isempty(tailpopulations)
    burn_spikes!(currentpop,tfire,popfire,neufire,tailpopulations)
  end
  return nothing
end

function dynamics_step!(rng::AbstractRNG,t_now::Real,network,workspaces)
  tfire,popfire,labelfire,neufire =
    call_for_compute_next_spike(rng,t_now,network.populations,workspaces)
  burn_spikes!(0,tfire,popfire,neufire,network.populations)
  H.multipop_call_for_plasticity_update!(
    tfire,labelfire,neufire,network.populations)
  H.call_for_each_record_stuff!(
    network.recorders,tfire,popfire,neufire,labelfire,network)
  return tfire
end

end
