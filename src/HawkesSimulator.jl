module HawkesSimulator
using StatsBase,Statistics,Distributions,LinearAlgebra,Random
using FFTW

abstract type Population end
abstract type AbstractInteraction end
# abstract type PopulationState end
# abstract type InputNetwork end

# for now, inputs are stationary and constant
# this can be generalized later
# abstract type ExternalInput end


struct PopulationState{P}
  pop::P
  trains::Vector{Vector{Float64}}
  trains_history::Vector{Vector{Float64}}
  input::Vector{Float64} # input only used for external currents 
  spike_proposals::Vector{Float64}
end
Base.getindex(ps::PopulationState,j) = ps.trains[j]
Base.size(ps::PopulationState) = length(ps.input)

# constructor
function PopulationState(pop::Population,input::Vector{Float64})
  n = length(input)
  trains = [Float64[] for _ in 1:n ] # empty vectors
  trainsh = [ [0.0,] for _ in 1:n ] 
  spike_proposals = fill(Inf,n)
  PopulationState(pop,trains,trainsh,input,spike_proposals)
end

function reset!(ps::PopulationState)
  for train in ps.trains
    if !isempty(train)
      deleteat!(train,1:length(train))
    end
  end
  for trainh in ps.trains_history
    h = length(trainh)
    if h>2
      deleteat!(trainh,2:h)
    end
  end
  return nothing
end

# fallback
interaction_kernel(t::Real,ps::PopulationState) = interaction_kernel(t,ps.pop)
interaction_kernel_upper(t::Real,ps::PopulationState) = interaction_kernel_upper(t,ps.pop)

function numerical_rates(ps::PopulationState)
  return numerical_rate.(ps.trains_history)
end
function numerical_rate(train::Vector{Float64})
  Δt = train[end]-train[1]
  return length(train)/Δt
end

"""
    clear_trains!(ps::PopulationState,tlim=30.0)
Moves the spike train older than `tlim` to history. The spikes in history are not considered for 
interaction. This is necessary to make the dynamics efficient.
"""
function clear_trains!(ps::PopulationState,tlim=30.0)
  for ineu in 1:size(ps)
    train = ps.trains[ineu]
    trainh = ps.trains_history[ineu]
    t_last = train[end]
    th_last = trainh[end]
    t_elapsed = t_last - th_last
    if t_elapsed > tlim
      t_forh = filter(t -> t<t_last-tlim,train)
      push!(trainh,t_forh...)
      filter!(t-> t >= t_last-tlim,train)
    end
  end
  return nothing
end


abstract type AbstracInteraction end
struct NoInteraction <: AbstractInteraction end

struct InputNetwork{P,I<:AbstractInteraction}
  postpops::PopulationState
  prepops::Vector{PopulationState{P}}
  weights::Vector{Matrix{Float64}}  # perhaps implement SparseArrays later
  autapses::I
end

function InputNetwork(postps,prepops,weights)
  InputNetwork(postps,prepops,weights,NoInteraction())
end

function reset!(in::InputNetwork)
  reset!(in.postpops)
  reset!.(in.prepops)
  return nothing
end

struct Autapses <: AbstractInteraction
  weights_self::Vector{Float64}
  popself::Population
end


# input from a single presynaptic neuron and its spike train
# train, weight, prepop (kernel)
@inline function interaction(t::R,train::Vector{R},w::R,prepop::Population,upperbound::Bool) where R
  if (iszero(w) || isempty(train))
    return zero(R)
  elseif upperbound
    return  w * mapreduce(tk -> interaction_kernel_upper(t-tk,prepop),+,train)
  else
    return w * mapreduce(tk -> interaction_kernel(t-tk,prepop),+,train)
  end
end
# input from multiple presynaptic neurons
# pop-state with multiple trains, multiple weights
function interaction(t::R,weights_in::AbstractVector{<:R},prepop::PopulationState,
    upperbound::Bool) where R
  ret = 0.0
  for (j,w) in enumerate(weights_in)
    train = prepop[j]
    ret += interaction(t,train,w,prepop.pop,upperbound)
  end
  return ret
end

# autapses interaction
function interaction(t::R,pops::PopulationState,aut::NoInteraction,ineu,upperbound) where R
  return zero(R)
end

# autapses interaction : interacts only with its own train
@inline function interaction(t::R,pops::PopulationState,aut::Autapses,
    ineu::Integer,upperbound::Bool) where R
  train = pops[ineu]
  return interaction(t,train,aut.weights_self[ineu],aut.popself,upperbound)
end


function compute_rate(t_now::Real,inp::InputNetwork,ineu::Integer;upperbound::Bool=false)
  ret = 0.0
  for (w,prepop) in zip(inp.weights,inp.prepops)
    w_in = view(w,ineu,:)
    ret += interaction(t_now,w_in,prepop,upperbound)
  end
  # add autapses, if they exist
  ret += interaction(t_now,inp.postpops,inp.autapses,ineu,upperbound)
  return ret
end


# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function hawkes_next_spike(t_now::Real,inp::InputNetwork,ineu::Integer;Tmax=100.0)
  t_start = t_now
  t = t_now 
  ext = inp.postpops.input[ineu]
  freq(t) = ext + compute_rate(t,inp,ineu;upperbound=false)
  freq_up(t) = max( ext + compute_rate(t,inp,ineu;upperbound=true), eps(100.)) # cannot be negative
  while (t-t_start)<Tmax # Tmax is upper limit, if rate too low 
    M = freq_up(t)
    Δt = rand(Exponential(inv(M)))
    t = t+Δt
    u = rand(Uniform(0,M))
    if u <= freq(t) 
      return t
    end
  end
  return Tmax + t_start
end


function update_proposals!(t_now::Real,in_net::InputNetwork)
  nneu = size(in_net.postpops)
  for ineu in 1:nneu
    in_net.postpops.spike_proposals[ineu] = hawkes_next_spike(t_now,in_net,ineu) 
  end
  return nothing
end


function dynamics_step!(t_now::Real,input_networks::Vector{<:InputNetwork})
  nntws = length(input_networks)
  proposals_best = Vector{Float64}(undef,nntws)
  neuron_best = Vector{Int64}(undef,nntws)
  # for each postsynaptic network, compute spike proposals 
  for (kn,ntw) in enumerate(input_networks)
    update_proposals!(t_now,ntw)
    (proposals_best[kn], neuron_best[kn]) = findmin(ntw.postpops.spike_proposals) 
  end 
  # select smallest , update train for that specific neuron
  (tbest,kbest) = findmin(proposals_best)
  # add the spiking time to the neuron that is firing
  best_neu_train = input_networks[kbest].postpops[neuron_best[kbest]]
  push!(best_neu_train,tbest)
  # update t_now 
  return tbest
end


### Interaction kernels here


# Negative exponential

struct PopulationExp <: Population
  τ::Float64
end

@inline function interaction_kernel(t::R,pop::PopulationExp) where R<:Real
  return t < zero(R) ? zero(R) : exp(-t/pop.τ) / pop.τ
end
interaction_kernel_upper(t::Real,pop::PopulationExp) = interaction_kernel(t,pop) + eps(100.0)

interaction_kernel_fourier(ω::Real,pop::PopulationExp) =  inv( 1 + im*2*π*ω*pop.τ)

# Alpha-shape with delay

struct PopulationAlphaDelay <: Population
  τ::Float64
  τdelay::Float64
end

@inline function interaction_kernel(t::Real,pop::PopulationAlphaDelay)
  Δt = t - pop.τdelay
  if Δt <= 0.0
    return 0.0
  else
    return Δt/pop.τ^2 * exp(-Δt/pop.τ)
  end
end
@inline function interaction_kernel_upper(t::Real,pop::PopulationAlphaDelay)
  Δt = t - pop.τdelay
  if Δt <= pop.τ
    return  inv(pop.τ*ℯ) + eps(100.)
  else
    return  interaction_kernel(t,pop) + eps(100.0)
  end
end

@inline function interaction_kernel_fourier(ω::Real,pop::PopulationAlphaDelay)
  return exp(-im*2*π*ω*pop.τdelay) / (1+im*2*π*ω*pop.τ)^2
end


##########################
## covariance density

function bin_spikes(Y::Vector{R},dt::R,Tmax::R) where R
  h = fit(Histogram,Y,0.0:dt:Tmax,closed=:left)
  return h.weights
end


# time starts at 0, ends at T-dt, there are T/dt steps in total
@inline function get_times(dt::Real,T::Real)
  return (0.0:dt:(T-dt))
end

# frequencies for Fourier transform.
# from -1/dt to 1/dt - 1/T in steps of 1/T
function get_frequencies_centerzero(dt::Real,T::Real)
  dω = inv(T)
  ωmax = 0.5/dt
  f = dω:dω:ωmax
  ret = vcat(-reverse(f),0.,f[1:end-1])
  return ret
end

function get_frequencies(dt::Real,T::Real)
  dω = inv(T)
  ωmax = inv(dt)
  ret = 0:dω:ωmax-dω
  return ret
end

function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tmax::Union{R,Nothing}=nothing) where R
  ret = covariance_density_numerical([Y,],dτ,τmax,Tmax;verbose=false)
  return ret[1,1,:]
end

function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R,
   Tmax::Union{R,Nothing}=nothing ; verbose::Bool=false) where R
  Tmax = something(Tmax, maximum(x->x[end],Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,n,n,ndt)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tmax/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tmax)
    fmi = length(Ys[i]) / Tmax # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tmax)
      fmj = length(Ys[j]) / Tmax # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = _ret_alloc / (ndt_tot*dτ^2) - fmi*fmj
      ret[i,j,:] = _ret_alloc
    end
  end
  return ret
end



end # of module
