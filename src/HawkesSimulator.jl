module HawkesSimulator
using StatsBase,Statistics,Distributions,LinearAlgebra,Random
using FFTW
using Colors # to save rasters as png

abstract type UnitType end
abstract type Connection end
abstract type PlasticityRule end
struct NoPlasticity <: PlasticityRule end

struct ConnectionWeights{N,PL<:NTuple{N,PlasticityRule}} <: Connection
  weights::Matrix{Float64}
  plasticities::PL
end

function ConnectionWeights(weights::Matrix{Float64})
  plast = NTuple{0,NoPlasticity}()
  return ConnectionWeights(weights,plast)
end

abstract type AbstractNonlinearity end
# this one is the default
struct NLRelu <: AbstractNonlinearity end


# for now, inputs are stationary and constant
# this can be generalized later
# abstract type ExternalInput end

struct PopulationState{UT <:UnitType}
  label::Symbol
  n::Int64
  unittype::UT
  trains::Vector{Vector{Float64}}
  trains_history::Vector{Vector{Float64}}
end
nneurons(ps::PopulationState) = ps.n

# constructor

function rand_label()
  return Symbol(randstring(3))
end
function PopulationState(unit_type::UnitType,n::Int64;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  trains = [Float64[] for _ in 1:n] # empty vectors
  trainsh = [Float64[] for _ in 1:n] 
  PopulationState(label,n,unit_type,trains,trainsh)
end

function reset!(ps::PopulationState)
  for train in ps.trains
    if !isempty(train)
      deleteat!(train,1:length(train))
    end
  end
  for trainh in ps.trains_history
    if !isempty(trainh)
      deleteat!(trainh,1:length(trainh))
    end
  end
  return nothing
end


struct Population{N,PS<:PopulationState,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationState},
    NL<:AbstractNonlinearity}
  state::PS
  connections::TC
  pre_states::TP
  input::Vector{Float64} # input only used for external currents 
  nonlinearity::NL # nonlinearity if present
  spike_proposals::Vector{Float64} # allocation (probably useless)
end
nneurons(p::Population) = nneurons(p.state)

function Population(state::PopulationState,input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationState})...;
      nonlinearity::AbstractNonlinearity=NLRelu())
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(-Inf,nneurons(state))
  return Population(state,connections,pre_states,input,nonlinearity,spike_proposals) 
end

# one population only!
function Population(state::PopulationState,conn::Connection,input::Vector{Float64}; 
    nonlinearity::AbstractNonlinearity=NLRelu())
  return Population(state,input,(conn,state);nonlinearity=nonlinearity)
end

# recurrent network is just a tuple of (input) populations 
struct RecurrentNetwork{N,TP<:NTuple{N,Population}}
  populations::TP
end
function reset!(rn::RecurrentNetwork)
  for pop in rn.populations
    reset!(pop.state)
  end
  return nothing
end
@inline function npopulations(ntw::RecurrentNetwork)
  return length(ntw.populations)
end

# more constructors
function RecurrentNetwork((pops::P where P<:Population)...)
  RecurrentNetwork(pops)
end
# one population simplification
function RecurrentNetwork(state::PopulationState,conn::Connection,
    input::Vector{Float64};
    nonlinearity::AbstractNonlinearity=NLRelu())
  return RecurrentNetwork(Population(state,conn,input;nonlinearity=nonlinearity))
end
# or even simpler constructor
function RecurrentNetwork(state::PopulationState,weights::Matrix{Float64},
    input::Vector{Float64};
    nonlinearity=NLRelu())
  @assert size(weights,2) == length(input) 
  return RecurrentNetwork(
    Population(state,ConnectionWeights(weights),input;
    nonlinearity=nonlinearity))
end

# everything else

include("spikes_gen.jl")
# the interaction_kernel and interaction_kernel_upper come from here : 
include("kernels.jl")
# apply_nonlinearity defined here
include("nonlinearities.jl")
# analyze the output: mean, covariance, etc
include("spike_analysis.jl")
# apply_plasticities defined here
include("plasticity_rules.jl")

##


end # of module
