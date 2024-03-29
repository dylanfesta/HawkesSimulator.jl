

# input population is a simplified version of
# PopulationHawkes. Does not connect to anything

# I need to abstract the type, because there is a variant to test weights
abstract type AbstractPopulationInput <: AbstractPopulation end

struct PopulationInput{PS<:PopulationState} <: AbstractPopulationInput
  state::PS
  spike_proposals::Vector{Float64} # memory allocation 
  function PopulationInput(ps::PS) where PS<:PopulationState
    new{PS}(ps,fill(+Inf,nneurons(ps)))
  end
end

abstract type SpikeGenerator end
struct InputUnit{SG<:SpikeGenerator} <: UnitType
  spike_generator::SG
end
@inline function compute_next_spike(t_now::Real,pop::AbstractPopulationInput,ineu::Integer;Tmax::Float64=100.0)
  return compute_next_spike(t_now,pop.state.unittype.spike_generator,ineu;Tmax=Tmax)
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
    TP<:NTuple{N,PopulationState}} <: AbstractPopulationInput
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
  spike_proposals = fill(+Inf,nneurons(state))
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
@inline function compute_next_spike(t_now::Float64,sg::SGPoisson,idxneu::Integer;Tmax::Float64=100.0)
  Δt = -log(rand())/sg.rates[idxneu]
  return t_now + min(Δt,Tmax)
end

# this is here mostly for testing
function make_poisson_samples(rate::R,t_tot::R) where R
  ret = Vector{R}(undef,round(Integer,1.3*rate*t_tot+10)) # preallocate
  t_curr = zero(R)
  k_curr = 1
  while t_curr <= t_tot
    Δt = -log(rand())/rate
    t_curr += Δt
    ret[k_curr] = t_curr
    k_curr += 1
  end
  return keepat!(ret,1:k_curr-2)
end


##############
# Spiketrains set externally 
struct SGTrains <: SpikeGenerator
  trains::Vector{Vector{Float64}}
end
@inline function compute_next_spike(t_now::Float64,sg::SGTrains,idxneu::Integer;Tmax::Float64=100.0)
  train = sg.trains[idxneu]
  t_now_plus = t_now+eps(10*t_now) # add an increment to move to next element
  idx = searchsortedfirst(train,t_now_plus)
  if idx > length(train)
    return Inf
  else
    return min(train[idx],t_now+Tmax)
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
    Tmax=100.0,nowarning::Bool=false)
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
function compute_next_spike(t_now::Float64,sg::SGPoissonFunction,idx_neu::Integer;Tmax::Float64=100.0)
  return _rand_by_thinning(t_now,idx_neu,sg.ratefunction,sg.ratefunction_upper;Tmax=Tmax)
end


# compatibility with ExpKernel stuff
# let's just do nothing, because inputs never depend on past states
function burn_spike!(::Real,ps::PopulationState,::Integer)
  typeassert(ps.unittype,InputUnit)
  return nothing
end



########### 
# 2022-05-24 -- I entirely forgot what I did before, so 
# let's start again

struct InputCurrentFun <: AbstractPopulationState
  label::Symbol
  n::Int64
  f::Function # f(t::Float64,neu::Integer) -> Float64
  f_upper::Function  # same signature
  function InputCurrentFun(n::Integer,f::Function,f_upper::Function;
      label::Union{Symbol,String}=rand_label())
    return new(Symbol(label),n,f,f_upper)
  end
end

reset!(::InputCurrentFun) = nothing

@inline function propagated_signal(t_now::Real,idx_post::Integer,
    ::Union{PopulationStateExpKernel,PopulationStateExpKernelInhibitory},
    ::ConnectionVoid,in::InputCurrentFun)
  return in.f(t_now,idx_post)
end
@inline function propagated_signal_upper(t_now::Real,idx_post::Integer,
    ::Union{PopulationStateExpKernel,PopulationStateExpKernelInhibitory},
    ::ConnectionVoid,in::InputCurrentFun)
  return in.f_upper(t_now,idx_post)
end