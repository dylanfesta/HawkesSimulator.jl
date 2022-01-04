# Mixed model neurons. In part they produce a spike train **fully determined beforehand** in part they can interact (thus producing additional spikes)


struct PopulationStateMixedExp{N}  <: PopulationStateMarkovian
  label::Symbol
  n::Int64
  traces::NTuple{N,Trace}
  forced_trains::Vector{Vector{Float64}}
end
nneurons(ps::PopulationStateMixedExp) = ps.n

function PopulationStateMixedExp(forced_trains::Vector{Vector{Float64}},traces...;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  n = length(forced_trains)
  PopulationStateMixedExp(label,n,traces,forced_trains)
end
function reset!(ps::PopulationStateMixedExp)
  for tra in ps.traces
    reset!(tra)
  end
  return nothing
end

@inline function get_next_forced_spike(t_now,ps::PopulationStateMixedExp,idx_neu::Integer)
  return get_next_spike(t_now,ps.forced_trains[idx_neu])
end

@inline function get_next_forced_spike!(t_alloc::Vector{Float64},t_now,ps::PopulationStateMixedExp)
  for i in eachindex(t_alloc)
    t_alloc[i] = get_next_spike(t_now,ps.forced_trains[i])
  end
  return findmin(t_alloc)
end

# same as compute_next_spike in src/inputs.jl
@inline function get_next_spike(t_now::R,train::Vector{R}) where R<:Real
  t_now_plus = t_now+eps(10*t_now) # add an increment to move to next element
  idx = searchsortedfirst(train,t_now_plus)
  if idx > length(train)
    return Inf
  else
    return train[idx]
  end
end

struct PopulationMixedExp{N,PS<:PopulationStateMixedExp,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationStateMarkovian},
    NL<:AbstractNonlinearity} <: AbstractPopulation
  state::PS
  connections::TC
  pre_states::TP
  input::Vector{Float64} # input only used for external currents 
  nonlinearity::NL # nonlinearity if present
  spike_proposals::Vector{Float64} # memory allocation 
end
function PopulationMixedExp(state::PopulationStateMixedExp,input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationStateMixedExp})...;
      nonlinearity::AbstractNonlinearity=NLRelu())
  connections = Tuple(getindex.(conn_pre,1))
  typeassert.(connections,ConnectionExpKernel)
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(Inf,nneurons(state))
  return PopulationMixedExp(state,connections,pre_states,input,nonlinearity,spike_proposals) 
end
# one population only!
function PopulationMixedExp(state::PopulationStateMixedExp,conn::Connection,input::Vector{Float64}; 
    nonlinearity::AbstractNonlinearity=NLRelu())
  return PopulationMixedExp(state,input,(conn,state);nonlinearity=nonlinearity)
end


function compute_rate(t_now::Real,external_input::Real,
    pop::PopulationMixedExp, idxneu::Integer)
  ret = external_input
  ps_post = pop.state
  for (conn,ps_pre) in zip(pop.connections,pop.pre_states)
    ret += propagated_signal(t_now,idxneu,ps_post,conn,ps_pre)
  end
  return apply_nonlinearity(ret,pop.nonlinearity)
end
function compute_rates!(r_alloc::Vector{Float64},t_now::Real,pop::PopulationMixedExp)
  inputs = pop.input
  for i in eachindex(r_alloc)
    r_alloc[i] = compute_rate(t_now,inputs[i],pop,i)
  end
  return nothing
end

# Multivariate thinning algorithf(Y. Chen, 2016) plus forced spiketimes
function compute_next_spike(t_now::Real,pop::PopulationMixedExp;Tmax::Real=100.0)
  t_start = t_now
  t = t_now
  n = nneurons(pop)
  # compute forced value here
  nneu = nneurons(pop)
  t_alloc = Vector{Float64}(undef,nneu)
  (t_forced,k_forced) = get_next_forced_spike!(t_alloc,t_now,pop.state)
  rates = pop.spike_proposals # recycle & reuse  # Vector{Float64}(undef,n)
  dorates!(t) = compute_rates!(rates,t,pop)
  while (t-t_start)<Tmax 
    dorates!(t)
    M = sum(rates)
    Δt =  -log(rand())/M # rand(Exponential())/M
    t = t+Δt
    if t >= t_forced # if in the future, return forced spike instead
      return (t_forced,k_forced)
    else
      u = rand()*M # random between 0 and M
      dorates!(t)
      cumsum!(rates,rates)
      k = searchsortedfirst(rates,u)
      if k <= n
        return (t,k)
      end
    end
  end
  @warn "Population did not spike ! Returning fake spike at t=$(Tmax+t_start) (is this a test?)"
  return (Tmax + t_start,1)
end
