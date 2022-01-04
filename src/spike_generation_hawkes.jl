


# warmup phase!

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
function warmup_step_onepopulation!(t_now::Real,ntw::RecurrentNetwork,
    warmup_rates::Vector{Float64})
  proposals_best = Vector{Float64}(undef,npops)
  neuron_best = Vector{Int64}(undef,npops)
  # compute spike proposals 
  pop = ntw.populations[1]
  # update proposed next spike for each postsynaptic neuron
  nneu = nneurons(pop)
  for ineu in 1:nneu
    pop.spike_proposals[ineu] = t_now + -log(rand())/warmup_rates[ineu] 
  end
  # best candidate and neuron that fired it for one input network
  (tbest, neuron_best) = findmin(pop.spike_proposals) 
  # update train for that specific neuron :
  # add the spiking time to the neuron that fired it
  best_neu_train = pop.state.trains[neuron_best]
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
    while t_now <= Twarmup
      t_now =  warmup_step_onepopulation!(t_now,ntw,warmup_rates)
    end
    return t_now
  else
    @assert eltype(warmup_rates) <: Vector
    @assert length(warmup_rates) == npops
    @assert all(nneurons.(ntw.populations) .== length.(warmup_rates)) 
    while t_now <= Twarmup
      t_now =  warmup_step!(t_now,ntw,warmup_rates)
    end
    return t_now
  end
end

# now consider interactive network

# input from a single presynaptic neuron and its spike train, with given pre/post weight
@inline function interaction(t::R,train::Vector{R},w::R,prepop::UnitType) where R
  return w * mapreduce(tk -> interaction_kernel(t-tk,prepop),+,train)
end
@inline function interaction_upperbound(t::R,train::Vector{R},w::R,prepop::UnitType) where R
  return  w * mapreduce(tk -> interaction_kernel_upper(t-tk,prepop),+,train)
end

function compute_rate(t_now::Real,external_input::Real,
    pop::PopulationHawkes, idxneu::Integer)
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

function compute_rates!(r_alloc::Vector{Float64},t_now::Real,pop::PopulationHawkes)
  inputs = pop.input
  for i in eachindex(r_alloc)
    r_alloc[i] = compute_rate(t_now,inputs[i],pop,i)
  end
  return nothing
end


# same as above, but interaction upperbound (for thinning algorithm)
function compute_rate_upperbound(t_now::Real,external_input::Real,
    pop::PopulationHawkes, idxneu::Integer)
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
function compute_ratesum_upperbound(t_now::Real,pop::PopulationHawkes)
  neus = 1:nneurons(pop)
  inputs = pop.input
  return mapreduce(neu->compute_rate_upperbound(t_now,inputs[neu],pop,neu),+,neus)
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
# function compute_next_spike(t_now::Real,pop::PopulationHawkes,ineu::Integer;Tmax::Real=100.0)
#   t_start = t_now
#   t = t_now 
#   ext_inp = pop.input[ineu]
#   freq(t) = compute_rate(t,ext_inp,pop,ineu)
#   freq_up(t) = compute_rate_upperbound(t,ext_inp,pop,ineu)
#   while (t-t_start)<Tmax # Tmax is upper limit, if rate too low 
#     (M::Float64) = freq_up(t)
#     Δt =  -log(rand())/M # rand(Exponential())/M
#     t = t+Δt
#     u = rand()*M # random between 0 and M
#     (u_up::Float64) = freq(t) 
#     if u <= u_up
#       return t
#     end
#   end
#   return Tmax + t_start
# end


# Multivariate thinning algorith. From  Y. Chen, 2016
function compute_next_spike(t_now::Real,pop::PopulationHawkes;Tmax::Real=100.0)
  t_start = t_now
  t = t_now
  n = nneurons(pop)
  rates = pop.spike_proposals # recycle & reuse  # Vector{Float64}(undef,n)
  rates_up = similar(rates)
  dorates!(t) = compute_rates!(rates,t,pop)
  ratesup(t) = compute_ratesum_upperbound(t,pop)
  while (t-t_start)<Tmax 
    M = ratesup(t)
    Δt =  -log(rand())/M # rand(Exponential())/M
    t = t+Δt
    u = rand()*M # random between 0 and M
    dorates!(t)
    cumsum!(rates,rates)
    k = searchsortedfirst(rates,u)
    if k <= n
      return (t,k)
    end
  end
  @warn "Population did not spike ! Returning fake spike at t=$(Tmax+t_start) (is this a test?)"
  return (Tmax + t_start,1)
end

# for population, just pick the one that happens earlier
function compute_next_spike(t_now::Real,pop::AbstractPopulation;Tmax::Float64=100.0)
  nneu = nneurons(pop)
  for ineu in 1:nneu
    # update proposed next spike for each postsynaptic neuron
    pop.spike_proposals[ineu] = compute_next_spike(t_now,pop,ineu;Tmax=Tmax) 
  end
  # best candidate and neuron that fired it for one input network
  return findmin(pop.spike_proposals)  # (neuron_spike,t_spike)
end

function dynamics_step!(t_now::Real,ntw::RecurrentNetwork)
  npops = npopulations(ntw)
  proposals_best = Vector{Float64}(undef,npops)
  neuron_best = Vector{Int64}(undef,npops)
  # for each postsynaptic network, compute spike proposals 
  for (kn,pop) in enumerate(ntw.populations)
    (proposals_best[kn], neuron_best[kn]) = compute_next_spike(t_now,pop)
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
    pop.spike_proposals[ineu] = compute_next_spike(t_now,pop,ineu) 
  end
  (tbest, neuron_best) = findmin(pop.spike_proposals) 
  # select train for that specific neuron
  # and add the spike time to it
  push!(pop.state.trains[neuron_best],tbest)
  # return the new current time, corresponding to the spiketime 
  return tbest
end


# flush trains into history to speed up calculation 

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

