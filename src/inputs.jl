

# input population is a simplified version of
# PopulationHawkes. Does not connect to anything

struct PopulationInput{PS<:PopulationState} <: AbstractPopulation
  state::PS
  spike_proposals::Vector{Float64} # memory allocation 
  function PopulationInput(ps::PS) where PS<:PopulationState
    new{PS}(ps,fill(0.0,nneurons(ps)))
  end
end


abstract type SpikeGenerator end
struct InputUnit{SG<:SpikeGenerator} <: UnitType
  spike_generator::SG
end
@inline function compute_next_spike(t_now::Real,pop::PopulationInput,ineu::Integer)
  return compute_next_spike(t_now,pop.state.unittype.spike_generator,ineu)
end

# simplified constructor for population object
# (because it has no inputs or incoming weights, so it is pretty much a 
# wrapper around its population state)
function PopulationInput(sg::SpikeGenerator,n::Integer)
  return PopulationInput(PopulationState(InputUnit(sg),n))
end

############
# A population with dummy weights, to test plasticity functions.
# N.B. those weights have no influence whatsoever on the dynamics!
struct PopulationInputTestWeights{N,PS<:PopulationState,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationState},
    NL<:AbstractNonlinearity} <: AbstractPopulation
  state::PS
  connections::TC
  pre_states::TP
  spike_proposals::Vector{Float64} # memory allocation 
end
# and constructors
function PopulationInputTestWeights(state::PopulationState,
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationState})...)
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(-Inf,nneurons(state))
  return PopulationInputTestWeights(state,connections,pre_states,spike_proposals) 
end
# one population only!
function PopulationInputTestWeights(state::PopulationState,conn::Connection)
  return PopulationInputTestWeights(state,(conn,state))
end

##############
# Different types of inputs :

##############
# Poisson with given rate
struct SGPoisson <: SpikeGenerator
  rates::Vector{Float64}
end
@inline function compute_next_spike(t_now::Float64,sg::SGPoisson,idxneu::Integer)
  Δt = -log(rand())/sg.rates[idxneu]
  return t_now+Δt
end

##############
# Spiketrains set externally 
struct SGTrains <: SpikeGenerator
  trains::Vector{Vector{Float64}}
end
@inline function compute_next_spike(t_now::Float64,sg::SGTrains,idxneu::Integer)
  train = sg.trains[idxneu]
  idx = searchsortedfirst(train,t_now)
  if idx > length(train)
    return Inf
  else
    return train[idx]
  end
end

##############
# Poisson varying according to a function
# for exact spike generation, a non-decreasing upper limit is needed, too
struct SGPoissonFunction <: SpikeGenerator
  ratefunction::Function # (t::Float64,idx::Int64) -> rate::Float64
  ratefunction_upper::Function # (t::Float64,idx::Int64) -> rate::Float64
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
# pretty much the same as in `spike_generation_hawkes` ...
function _rand_by_thinning(t_start::Real,idx_neu::Integer,
    get_rate::Function,get_rate_upper::Function;
    Tmax=50.0,nowarning::Bool=false)
  t = t_start 
  while (t-t_start)<Tmax # Tmax is upper limit, if rate too low 
    (rup::Float64) = get_rate_upper(t,idx_neu)
    Δt = -log(rand())/rup # rand(Exponential())/rup
    t = t+Δt
    u = rand()*rup # rand(Uniform(0.0,rup))
    (_r::Float64) = get_rate(t,idx_neu) 
    if u <= _r
      return t
    end
  end
  # if we are above Tmax, just return upper limit
  if !nowarning
    @warn "Upper limit reached, input firing below $(inv(Tmax)) Hz"
  end
  return Tmax + t_start
end
function compute_next_spike(t_now::Float64,sg::SGPoissonFunction,idx_neu::Integer)
  return _rand_by_thinning(t_now,idx_neu,sg.ratefunction,sg.ratefunction_upper)
end
