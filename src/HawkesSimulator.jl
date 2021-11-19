module HawkesSimulator
using StatsBase,Statistics,Distributions,LinearAlgebra,Random
using FFTW
using Colors # to save rasters as png

abstract type UnitType end

abstract type Connection end

abstract type AbstractNonlinearity end
struct NLRelu <: AbstractNonlinearity end

struct ConnectionWeights <: Connection
  weights::Matrix{Float64} # dense weight matrix
  # might want to add plasticity rules in the future...
end

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


# input from a single presynaptic neuron and its spike train, with given pre/post weight
@inline function interaction(t::R,train::Vector{R},w::R,prepop::UnitType) where R
  return w * mapreduce(tk -> interaction_kernel(t-tk,prepop),+,train)
end
@inline function interaction_upperbound(t::R,train::Vector{R},w::R,prepop::UnitType) where R
  return  w * mapreduce(tk -> interaction_kernel_upper(t-tk,prepop),+,train)
end

function compute_rate(t_now::Real,external_input::Real,
    pop::Population, idxneu::Integer)
  ret = external_input
  for (conn,pre) in zip(pop.connections,pop.pre_states)
    weights = conn.weights
    npre = size(weights,2)
    for j in 1:npre 
      wij = weights[idxneu,j]
      pre_train = pre.trains[j]
      if ! (iszero(wij) || isempty(pre_train))
        ret += interaction(t_now,pre_train,wij,pre.unittype)
      end
    end
  end
  return apply_nonlinearity(ret,pop.nonlinearity)
end
# same as above, but interaction upperbound (for thinning algorithm)
function compute_rate_upperbound(t_now::Real,external_input::Real,
    pop::Population, idxneu::Integer)
  ret = external_input
  for (conn,pre) in zip(pop.connections,pop.pre_states)
    weights = conn.weights
    npre = size(weights,2)
    for j in 1:npre 
      wij = weights[idxneu,j]
      pre_train = pre.trains[j]
      if ! (iszero(wij) || isempty(pre_train))
        ret += interaction_upperbound(t_now,pre_train,wij,pre.unittype)
      end
    end
  end
  return apply_nonlinearity(ret,pop.nonlinearity)
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function hawkes_next_spike(t_now::Real,pop::Population,ineu::Integer;Tmax::Real=100.0)
  t_start = t_now
  t = t_now 
  ext_inp = pop.input[ineu]
  expdistr=Exponential()
  freq(t) = compute_rate(t,ext_inp,pop,ineu)
  freq_up(t) = compute_rate_upperbound(t,ext_inp,pop,ineu)
  while (t-t_start)<Tmax # Tmax is upper limit, if rate too low 
    M = freq_up(t)
    Δt = rand(expdistr)/M
    t = t+Δt
    u = rand()*M # random between 0 and M
    if u <= freq(t) 
      return t
    end
  end
  return Tmax + t_start
end

function dynamics_step!(t_now::Real,ntw::RecurrentNetwork)
  npops = npopulations(ntw)
  proposals_best = Vector{Float64}(undef,npops)
  neuron_best = Vector{Int64}(undef,npops)
  # for each postsynaptic network, compute spike proposals 
  for (kn,pop) in enumerate(ntw.populations)
    # update proposed next spike for each postsynaptic neuron
    nneu = nneurons(pop)
    for ineu in 1:nneu
      pop.spike_proposals[ineu] = hawkes_next_spike(t_now,pop,ineu) 
    end
    # best candidate and neuron that fired it for one input network
    (proposals_best[kn], neuron_best[kn]) = findmin(pop.spike_proposals) 
  end 
  # select next spike (best across all input_networks)
  (tbest,kbest) = findmin(proposals_best)
  # update train for that specific neuron :
  # add the spiking time to the neuron that fired it
  best_neu_train = ntw.populations[kbest].state.trains[neuron_best[kbest]]
  push!(best_neu_train,tbest)
  # update t_now 
  return tbest
end

"""
    dynamics_step_singlepopulation!(t_now::Real,ntw::RecurrentNetwork)

Iterates a *one-population* network up until its next spike time.
This is done by computing a next spike proposal for each neuron, and then
picking the one that happens sooner. This spike is then added to the 
spiketrain for that neuron. The fundtion returns the new current time
of the simulation.

For long simulations, this functions should be called jointly with 
`flush_trains!`. Otherwise the spike trains will keep growing, making the 
propagation of signals extremely cumbersome.

# Arguments
+ `t_now` - Current time of the simulation
+ `ntw`   - The network

# Returns   
+ `t_now_new` - the new current time of the simulation

"""
function dynamics_step_singlepopulation!(t_now::Real,ntw::RecurrentNetwork)
  # for each postsynaptic network, compute spike proposals 
  @assert npopulations(ntw) == 1 "This function is only for 1-population networks"
  pop = ntw.populations[1]
  nneu = nneurons(pop)
  for ineu in 1:nneu
    pop.spike_proposals[ineu] = hawkes_next_spike(t_now,pop,ineu) 
  end
  (tbest, neuron_best) = findmin(pop.spike_proposals) 
  # select train for that specific neuron
  # and add the spike time to it
  push!(pop.state.trains[neuron_best],tbest)
  # return the new current time, corresponding to the spiketime 
  return tbest
end



## Nonlinearities here

@inline function apply_nonlinearity(x::Real,::NLRelu)
  return max(x,0.0)
end
struct NLRectifiedQuadratic <: AbstractNonlinearity end
@inline function apply_nonlinearity(x::R,::NLRectifiedQuadratic) where R<:Real
  if x <= 0.0
    return 0.0
  else 
    return x*x
  end
end

### Unit types (interaction kernels) here

@inline function interaction_kernel_fourier(ω::Real,popstate::PopulationState)
  return interaction_kernel_fourier(ω,popstate.unittype)
end

# a cluncky sharp step !
struct KernelStep <: UnitType
  τ::Float64
end
@inline function interaction_kernel(t::R,ker::KernelStep) where R<:Real
  return (t < zero(R)) || (t > ker.τ) ? zero(R) :  inv(ker.τ)
end
interaction_kernel_upper(t::Real,pop::KernelStep) = interaction_kernel(t,pop) + eps(100.0)

@inline function interaction_kernel_fourier(ω::R,pop::KernelStep) where R<:Real
  if iszero(ω)
    return one(R)
  end
  cc = 2π*pop.τ*ω
  return (sin(cc)-2*im*sin(0.5cc)^2)/cc
end

# Negative exponential
struct KernelExp <: UnitType
  τ::Float64
end
@inline function interaction_kernel(t::R,ker::KernelExp) where R<:Real
  return t < zero(R) ? zero(R) : exp(-t/ker.τ) / ker.τ
end
interaction_kernel_upper(t::Real,ker::KernelExp) = interaction_kernel(t,ker) + eps(100.0)
interaction_kernel_fourier(ω::Real,ker::KernelExp) =  inv( 1 + im*2*π*ω*ker.τ)

# Alpha-shape with delay

struct KernelAlphaDelay <: UnitType
  τ::Float64
  τdelay::Float64
end

@inline function interaction_kernel(t::Real,ker::KernelAlphaDelay)
  Δt = t - ker.τdelay
  if Δt <= 0.0
    return 0.0
  else
    return Δt/ker.τ^2 * exp(-Δt/ker.τ)
  end
end
@inline function interaction_kernel_upper(t::Real,ker::KernelAlphaDelay)
  Δt = t - ker.τdelay
  if Δt <= ker.τ
    return  inv(ker.τ*ℯ) + eps(100.)
  else
    return  interaction_kernel(t,ker) + eps(100.0)
  end
end
@inline function interaction_kernel_fourier(ω::Real,ker::KernelAlphaDelay)
  return exp(-im*2*π*ω*ker.τdelay) / (1+im*2*π*ω*ker.τ)^2
end

## flush trains into history to speed up calculation 

"""
    flush_trains!(ps::PopulationState,Ttrigger::Real;
        Tflush::Union{Real,Nothing}=nothing)
 
  Spike history is spiketimes that do not interact with the kernel (because too old)      
  This function compares most recent spike with spike history, if enough time has passed
  (measured by Ttrigger) it flushes the spiketrain up to Tflush into the history.
"""
function flush_trains!(ps::PopulationState,Ttrigger::Real;
    Tflush::Union{Real,Nothing}=nothing)
  Tflush=something(Tflush,0.5*Ttrigger)
  @assert Tflush < Ttrigger
  for neu in 1:nneurons(ps)
    train = ps.trains[neu]
    history = ps.trains_history[neu]
    if isempty(train) # no spikes, then nothing to do
      continue
    end
    t_last = train[end]
    th_last = isempty(history) ? 0.0 : history[end]
    if (t_last - th_last) > Ttrigger
      (ps.trains_history[neu],ps.trains[neu]) = 
        _flush_train!(history,train,t_last+Tflush)
    end
  end
  return nothing
end
function flush_trains!(ntw::RecurrentNetwork,Ttrigger::Real;
  Tflush::Union{Real,Nothing}=nothing)
  for pop in ntw.populations
    flush_trains!(pop.state,Ttrigger;Tflush=Tflush)
  end
end 

# does the flushing, returning two (not necessarily new) vectors
function _flush_train!(history::Vector{R},train::Vector{R},Tflush::R) where R
  idx = searchsortedfirst(train,Tflush)
  idx_tohistory = 1:(idx-1)
  history_new = vcat(history,view(train,idx_tohistory))
  deleteat!(train,idx_tohistory)
  return history_new,train
end

# withouth other arguments, flushes everything into history!
function flush_trains!(ps::PopulationState)
  for neu in 1:nneurons(ps)
    (ps.trains_history[neu],ps.trains[neu]) = 
        _flush_train!( ps.trains_history[neu],ps.trains[neu],Inf)
  end
  return nothing
end
function flush_trains!(ntw::RecurrentNetwork)
  for pop in ntw.populations
    flush_trains!(pop.state)
  end
end 



###### mean rates and other measures

function numerical_rates(ps::PopulationState;
    Tstart::Real=0.0,Tend::Real=Inf)
  return numerical_rate.(ps.trains_history;Tstart=Tstart,Tend=Tend)
end
function numerical_rates(pop::Population;
    Tstart::Real=0.0,Tend::Real=Inf)
  return numerical_rates(pop.state;Tstart=Tstart,Tend=Tend)
end

function numerical_rate(train::Vector{Float64};
    Tstart::Real=0.0,Tend::Real=Inf)
  isempty(train) && return 0.0  
  Tend = min(Tend,train[end])
  Δt = Tend - Tstart
  return length(train)/Δt
end

##########################
## covariance density

function bin_spikes(Y::Vector{R},dt::R,Tend::R;
    Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  ret = fill(0,length(times)-1)
  for y in Y
    if Tstart < y <= Tend
      k = searchsortedfirst(times,y)-1
      ret[k] += 1
    end
  end
  return ret
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

#### warmup phase!

"""
  warmup_step!(t_now::Real,ntw::RecurrentNetwork,
    warmup_rates::Union{Vector{Float64},Vector{Vector{Float64}}}) -> t_end

In the warmup phase, all neurons fire as independent Poisson process 
with a rate set by `warmup_rates`.  
This is useful to quick-start the network, or set itinitial
conditions that are far from the stable point.

# Arguments
+ `t_now` - current time
+ `ntw`  - the network (warning: weights and kernels are entirely ignored here)
+ `warmup_rates` - the desired stationary rates. In a one-population network,
    it is a vector with the desired rates. In a multi-population network,
    is a collection of vectors, where each vector refers to one population.
"""
function warmup_step!(t_now::Real,ntw::RecurrentNetwork,
    warmup_rates::Union{Vector{Float64},Vector{Vector{Float64}}})
  npops = npopulations(ntw)
  isonepop = npops == 1
  expdistr = Exponential()
  proposals_best = Vector{Float64}(undef,npops)
  neuron_best = Vector{Int64}(undef,npops)
  # for each postsynaptic network, compute spike proposals 
  for (kn,pop) in enumerate(ntw.populations)
    # update proposed next spike for each postsynaptic neuron
    nneu = nneurons(pop)
    _rates =  isonepop ?  warmup_rates : warmup_rates[kn]
    for ineu in 1:nneu
      pop.spike_proposals[ineu] = t_now + rand(expdistr)/_rates[ineu] 
    end
    # best candidate and neuron that fired it for one input network
    (proposals_best[kn], neuron_best[kn]) = findmin(pop.spike_proposals) 
  end 
  # select next spike (best across all input_networks)
  (tbest,kbest) = findmin(proposals_best)
  # update train for that specific neuron :
  # add the spiking time to the neuron that fired it
  best_neu_train = ntw.populations[kbest].state.trains[neuron_best[kbest]]
  push!(best_neu_train,tbest)
  # update t_now 
  return tbest
end


"""
    do_warmup!(Twarmup::Real,ntw::RecurrentNetwork,
        warmup_rates::Union{Vector{Float64},Vector{Vector{Float64}}}) -> t_end

In the warmup phase, all neurons fire as independent Poisson process 
with a rate set by `warmup_rates`.  
This is useful to quick-start the network, or set itinitial
conditions that are far from the stable point.

# Arguments
+ `Twarmup` - total warmup time
+ `ntw`  - the network (warning: weights and kernels are entirely ignored here)
+ `warmup_rates` - the desired stationary rates. In a one-population network,
    it is a vector with the desired rates. In a multi-population network,
    is a collection of vectors, where each vector refers to one population.

# In-place changes
The `trains` in the `populations` inside `ntw` will contain the warmup spiking.

# Returns
+ `t_end` the time after warmum (the earliest spike after Twarmup). Note that 
 if you want to run a full network simulations, you need to use this as staring time.
"""

function do_warmup!(Twarmup::Real,ntw::RecurrentNetwork,
    warmup_rates::Union{Vector{Float64},Vector{Vector{Float64}}})
  t_now = 0.0 
  npops = npopulations(ntw)
  isonepop = npops == 1
  reset!(ntw) # clear spike trains etc
  # sanity checks
  if isonepop
    @assert eltype(warmup_rates) <: Number
    @assert nneurons(ntw.populations[1]) == length(warmup_rates)
  else
    @assert eltype(warmup_rates) <: Vector
    @assert length(warmup_rates) == npops
    @assert all(nneurons.(ntw.populations) .== length.(warmup_rates)) 
  end
  while t_now <= Twarmup
    t_now =  warmup_step!(t_now,ntw,warmup_rates)
  end
  return t_now
end



###




"""
     draw_spike_raster(trains::Vector{Vector{Float64}},
      dt::Real,Tend::Real;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      background_color::Color=RGB(1.,1.,1.),
      spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
      max_size::Real=1E4) where C<:Color

Draws a matrix that contains the raster plot of the spike train.

# Arguments
+ `Trains` :  Vector of spike trains. The order of the vector corresponds to 
the order of the plot. First element is at the top, second is second row, etc.
+ `dt` : time interval representing one horizontal pixel  
+ `Tend` : final time to be considered

# Optional arguments
+ `Tstart` : starting time
+ `max_size` : throws an error if image is larger than this number (in pixels)
+ `spike_size` : heigh of spike (in pixels)
+ `spike_separator` : space between spikes, and vertical padding
+ `background_color` : self-explanatory
+ `spike_colors` : if a single color, color of all spikes, if vector of colors, 
color for each neuron (length should be same as number of neurons)

# returns
`raster_matrix::Matrix{Color}` you can save it as a png file

"""
function draw_spike_raster(trains::Vector{Vector{Float64}},
  dt::Real,Tend::Real;
    Tstart::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where C<:Color
  nneus = length(trains)
  if typeof(spike_colors) <: Color
    spike_colors = repeat([spike_colors,];outer=nneus)
  else
    @assert length(spike_colors) == nneus "error in setting colors"
  end
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,Tend;Tstart=Tstart))
  end
  ntimes = length(binned_binary[1])
  ret = fill(background_color,
    (nneus*spike_size + # spike sizes
      spike_separator*nneus + # spike separators (incl. top/bottom padding) 
      spike_separator),ntimes)
  @assert all(size(ret) .< max_size ) "The image is too big! Please change time limits"  
  for (neu,binv,col) in zip(1:nneus,binned_binary,spike_colors)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    ret[y_color,spk_idx] .= col
  end
  return ret
end



end # of module
