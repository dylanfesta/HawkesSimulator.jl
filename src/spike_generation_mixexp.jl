# Mixed model neurons. In part they produce a spike train **fully determined beforehand** in part they can interact (thus producing additional spikes)



struct PopulationStateMixedExp{N}  <: PopulationStateMarkovian
  label::Symbol
  n::Int64
  traces::NTuple{N,Trace}
  forced_trains::Vector{Vector{Float64}}
end
nneurons(ps::PopulationStateMixedExp) = ps.n

function PopulationStateMixedExp(n::Integer,forced_trains,traces...;
    label::Union{String,Nothing}=nothing)
  @assert length(forced_trains) == n  "Trains not set correctly!"  
  label = something(label,rand_label()) 
  PopulationStateMixedExp(label,n,traces,forced_trains)
end
function reset!(ps::PopulationStateMixedExp)
  for tra in ps.traces
    reset!(tra)
  end
  return nothing
end

@inline function get_next_forced_spike(t_now,ps::PopulationStateMixedExp,idx_neu)
  return get_next_spike(t_now,ps.forced_trains[idx_neu])
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

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function compute_next_spike(t_now::Real,pop::PopulationMixedExp,ineu::Integer;Tmax::Real=100.0)
  t_start = t_now
  t = t_now 
  # compute forced value here
  t_forced = get_next_forced_spike(t_now,pop.state,ineu)
  ext_inp = pop.input[ineu]
  freq(t) = compute_rate(t,ext_inp,pop,ineu)
  while (t-t_start)<Tmax 
    (M::Float64) = freq(t)
    Δt =  -log(rand())/M # rand(Exponential())/M
    t = t+Δt
    u = rand()*M # random between 0 and M
    (u_up::Float64) = freq(t) 
    if (u <= u_up) || t > t_forced
      break # and return t
    end
  end
  return min(t,t_forced)
end
