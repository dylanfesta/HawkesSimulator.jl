module HawkesSimulator
using StatsBase,Statistics,Distributions,LinearAlgebra,Random
using FFTW
using Colors # to save rasters as png

#=
UnitType : specifies the kernel for Hawkes processes, or 
the input type for input units. 
PopulationState  : contains UnitType, n, which is the population size,
then trains contains n arrays, each is the spike of past trains 
for that single unit.  trains_history are non-interactive spikes 
AbstractPopulation : contains one postsynaptic (or input) population state,
which is called `state`.Contains `spike_proposals` for memory allocation
PopulationHawkes : contains one postsynaptic population, plus the connecting from
other populations into it.
PopulationInput : used for input types. 

=#


abstract type UnitType end
abstract type Connection end
abstract type PlasticityRule end
struct NoPlasticity <: PlasticityRule end
reset!(::NoPlasticity) = nothing

struct ConnectionWeights{N,PL<:NTuple{N,PlasticityRule}} <: Connection
  weights::Matrix{Float64}
  plasticities::PL
end
function ConnectionWeights(weights::Matrix{Float64})
  plast = NTuple{0,NoPlasticity}()
  return ConnectionWeights(weights,plast)
end
function ConnectionWeights(weights::Matrix{Float64},
     (plasticity_rules::PL where PL<:PlasticityRule)...)
  return ConnectionWeights(weights,plasticity_rules)
end
function reset!(conn::ConnectionWeights)
  reset!.(conn.plasticities)
  return nothing
end

# Neurons do not interact through this connection
# Should be used to study plasticity of weights
struct ConnectionNonInteracting{N,PL<:NTuple{N,PlasticityRule}} <: Connection
  weights::Matrix{Float64}
  plasticities::PL
end
function ConnectionNonInteracting(weights::Matrix{Float64})
  plast = NTuple{0,NoPlasticity}()
  return ConnectionNonInteracting(weights,plast)
end
function ConnectionNonInteracting(weights::Matrix{Float64},
     (plasticity_rules::PL where PL<:PlasticityRule)...)
  return ConnectionNonInteracting(weights,plasticity_rules)
end
function reset!(conn::ConnectionNonInteracting)
  reset!.(conn.plasticities)
  return nothing
end



# useful for the input
struct ConnectionVoid <: Connection 
  plasticities::Tuple{NoPlasticity}
  function ConnectionVoid()
    return new((NoPlasticity(),))
  end
end
reset!(::ConnectionVoid) = nothing 

abstract type AbstractNonlinearity end
# this one is the default
struct NLRelu <: AbstractNonlinearity end

# between 0 and rmax
struct NLRmax <: AbstractNonlinearity
  rmax::Float64
end


# for now, inputs are stationary and constant
# this can be generalized later
# abstract type ExternalInput end

abstract type AbstractPopulationState end
# includes the special case of ExpKernel, which has no unit type
# must still have n and label.

struct PopulationState{UT <:UnitType} <: AbstractPopulationState
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

# this will cover Hawkes populations but also input populations
abstract type AbstractPopulation end
nneurons(p::AbstractPopulation) = nneurons(p.state)

struct PopulationHawkes{N,PS<:PopulationState,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationState},
    NL<:AbstractNonlinearity} <: AbstractPopulation
  state::PS
  connections::TC
  pre_states::TP
  input::Vector{Float64} # input only used for external currents 
  nonlinearity::NL # nonlinearity if present
  spike_proposals::Vector{Float64} # memory allocation 
end

function PopulationHawkes(state::PopulationState,input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationState})...;
      nonlinearity::AbstractNonlinearity=NLRelu())
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(-Inf,nneurons(state))
  return PopulationHawkes(state,connections,pre_states,input,nonlinearity,spike_proposals) 
end

# one population only!
function PopulationHawkes(state::PopulationState,conn::Connection,input::Vector{Float64}; 
    nonlinearity::AbstractNonlinearity=NLRelu())
  return PopulationHawkes(state,input,(conn,state);nonlinearity=nonlinearity)
end

# recurrent network is just a tuple of (input) populations 
struct RecurrentNetwork{N,TP<:NTuple{N,AbstractPopulation}}
  populations::TP
end
# base constructor
function RecurrentNetwork((pops::P where P<:AbstractPopulation)...)
  RecurrentNetwork(pops)
end
# reset
function reset!(rn::RecurrentNetwork)
  for pop in rn.populations
    reset!(pop.state)
    # reset plasticity operators if present
    if isdefined(pop,:connections) # because input has no connection field
      for conn in pop.connections
        for plast in conn.plasticities
          reset!(plast)
        end
      end
    end
  end
  return nothing
end
@inline function npopulations(ntw::RecurrentNetwork)
  return length(ntw.populations)
end

# more constructors
# one population simplification (assuming it is an Hawkes type!)
function RecurrentNetwork(state::PopulationState,conn::Connection,
    input::Vector{Float64};
    nonlinearity::AbstractNonlinearity=NLRelu())
  return RecurrentNetwork(PopulationHawkes(state,conn,input;nonlinearity=nonlinearity))
end
# or even simpler constructor
function RecurrentNetwork(state::PopulationState,weights::Matrix{Float64},
    input::Vector{Float64};
    nonlinearity=NLRelu())
  @assert size(weights,2) == length(input) 
  return RecurrentNetwork(
    PopulationHawkes(state,ConnectionWeights(weights),input;
    nonlinearity=nonlinearity))
end



###################
# traces as structs
# (after all, why not?  ... why shouldn't I use a struct?)

# trace for dynamics : all receive an update 
# as soon as the spike is produced. The update is normed
# so that the integral of kernel is one

# trace for plasticity : ignored by the dynamics, 
# plasticity rules should take care of it.

abstract type TracePurpose end
struct ForDynamics <: TracePurpose end
struct ForPlasticity <: TracePurpose end

struct Trace{P<:TracePurpose}
  val::Vector{Float64}
  τ::Float64
  t_last::Ref{Float64}
  purpose::P
  function Trace(τ::Real,n::Integer,
      (purpose::P)=ForPlasticity()) where {P<:TracePurpose}
    val = fill(0.0,n)
    t_last = Ref(0.0)
    return new{P}(val,τ,t_last,purpose)
  end
end

function nneurons(tra::Trace)
  return length(tra.val)
end

function propagate!(tnow::Float64,tra::Trace)
  Δt = tnow - tra.t_last[]
  tra.val .*= exp(-Δt/tra.τ)
  tra.t_last[]=tnow
  return nothing
end
# updates single element of trace, unless it is nothing
function update_now!(tra::Trace,idx_update::Int64,up_val::Float64=1.0)
  if idx_update != 0
    tra.val[idx_update] += up_val
  end
  return nothing
end

# updates only if the trace is specifically for dynamics
function update_for_dynamics!(::Trace,::Int64)
  return nothing
end
function update_for_dynamics!(tra::Trace{ForDynamics},idx_update::Int64)
  tra.val[idx_update] += inv(tra.τ)
  return nothing
end
# same with propagate
function propagate_for_dynamics!(::Float64,::Trace)
  return nothing
end
function propagate_for_dynamics!(tnow::Float64,tra::Trace{ForDynamics})
  Δt = tnow - tra.t_last[]
  tra.val .*= exp(-Δt/tra.τ)
  tra.t_last[]=tnow
  return nothing
end

function reset!(tra::Trace)
  fill!(tra.val,0.0) 
  tra.t_last[]=0.0
  return nothing
end

# proposal of future trace. Useful to compute quantities without advancing the trace
function trace_proposal!(proposal::Vector{R},tnow::R,tra::Trace) where R
  Δt = tnow - tra.t_last[]
  @. proposal = tra.val * exp(-Δt/tra.τ)
  return proposal
end


#### Small general utility functions
@inline hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))


#############
# everything else

# record spiketrains etc
include("recorders.jl")
# general Hawkes for any type of kernel
include("spike_generation_hawkes.jl")
# the interaction_kernel and interaction_kernel_upper come from here : 
include("kernels.jl")
# special case optimized for exponential kernels
include("spike_generation_expkernel.jl")
include("spike_generation_mixexp.jl")
# apply_nonlinearity defined here
include("nonlinearities.jl")
# input-neurons 
include("inputs.jl")
# apply_plasticities defined here
include("plasticity_rules.jl")
include("plasticity_heterosynaptic.jl")
# analyze the output: mean, covariance, etc
include("spike_analysis.jl")

end # of module
