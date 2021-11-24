
# define a special case of exponential Kernel, because Claudia said so

####
# Population and connection constructors

# the taus are in the trace, now. No unittype or kernel object
# this is pretty much a wrapper around neural traces
struct PopulationStateExpKernel{N}  <: AbstractPopulationState
  label::Symbol
  n::Int64
  traces::NTuple{N,Trace}
end
nneurons(ps::PopulationStateExpKernel) = ps.n

function PopulationStateExpKernel(n::Int64,traces...;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  PopulationStateExpKernel(label,n,traces)
end
function reset!(ps::PopulationStateExpKernel)
  for tra in ps.traces
    reset!(tra)
  end
  return nothing
end

struct PopulationExpKernel{N,PS<:PopulationStateExpKernel,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationStateExpKernel},
    NL<:AbstractNonlinearity} <: AbstractPopulation
  state::PS
  connections::TC
  pre_states::TP
  input::Vector{Float64} # input only used for external currents 
  nonlinearity::NL # nonlinearity if present
  spike_proposals::Vector{Float64} # memory allocation 
end
function PopulationExpKernel(state::PopulationStateExpKernel,input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationStateExpKernel})...;
      nonlinearity::AbstractNonlinearity=NLRelu())
  connections = Tuple(getindex.(conn_pre,1))
  typeassert.(connections,ConnectionExpKernel)
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(Inf,nneurons(state))
  return PopulationExpKernel(state,connections,pre_states,input,nonlinearity,spike_proposals) 
end
# one population only!
function PopulationExpKernel(state::PopulationStateExpKernel,conn::Connection,input::Vector{Float64}; 
    nonlinearity::AbstractNonlinearity=NLRelu())
  return PopulationExpKernel(state,input,(conn,state);nonlinearity=nonlinearity)
end

struct ConnectionExpKernel{N,PL<:NTuple{N,PlasticityRule}} <: Connection
  weights::Matrix{Float64}
  pre_trace::Trace
  trace_proposal::Vector{Float64}
  plasticities::PL
end
function ConnectionExpKernel(weights::Matrix{Float64},pre_trace::Trace)
  plast = NTuple{0,NoPlasticity}()
  _,npre = size(weights)
  @assert npre == nneurons(pre_trace)
  tra_prop = Vector{Float64}(undef,npre)
  return ConnectionExpKernel(weights,pre_trace,tra_prop,plast)
end
function ConnectionExpKernel(weights::Matrix{Float64},pre_trace::Trace,
     (plasticity_rules::PL where PL<:PlasticityRule)...)
  _,npre = size(weights)
  @assert npre == nneurons(pre_trace)
  tra_prop = Vector{Float64}(undef,npre)
  return ConnectionExpKernel(weights,pre_trace,tra_prop,plasticity_rules)
end

@inline function trace_proposal!(t_now::Real,conn::ConnectionExpKernel)
  return trace_proposal!(conn.trace_proposal,t_now,conn.pre_trace)
end

abstract type Recorder end

struct RecurrentNetworkExpKernel{N,TP<:NTuple{N,AbstractPopulation},TR<:NTuple{N,Recorder}}
  populations::TP
  recorders::TR
end
# one population constructor
function RecurrentNetworkExpKernel(pop::AbstractPopulation,recorders...)
  if isempty(recorders)
    recorders=NTuple{0,RecNothing}()
  end
  return RecurrentNetworkExpKernel((pop,),recorders)
end

# reset
function reset!(rn::RecurrentNetworkExpKernel)
  # reset internal variables
  for pop in rn.populations
    reset!(pop.state)
    fill!(pop.spike_proposals,Inf)
    # reset plasticity operators if present
    for conn in pop.connections
      for plast in conn.plasticities
        reset!(plast)
      end
    end
  end
  # reset recorders
  for rec in rn.recorders
    reset!(rec)
  end
  return nothing
end
@inline function npopulations(ntw::RecurrentNetworkExpKernel)
  return length(ntw.populations)
end

function burn_spike!(t_spike::Real,ps::PopulationStateExpKernel,idx_update::Integer)
  # take care of traces  
  for tra in ps.traces
    propagate_for_dynamics!(t_spike,tra) # update full trace to t_spike 
    update_for_dynamics!(tra,idx_update) # add pulse at index
  end
  return nothing
end

function compute_rate(t_now::Real,external_input::Real,
    pop::PopulationExpKernel, idxneu::Integer)
  ret = external_input
  for conn in pop.connections
    tra_tnow = trace_proposal!(t_now,conn)
    wij_all = view(conn.weights,idxneu,:)
    ret += dot(wij_all,tra_tnow)
  end
  return apply_nonlinearity(ret,pop.nonlinearity)
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function compute_next_spike(t_now::Real,pop::PopulationExpKernel,ineu::Integer;Tmax::Real=100.0)
  t_start = t_now
  t = t_now 
  ext_inp = pop.input[ineu]
  freq(t) = compute_rate(t,ext_inp,pop,ineu)
  while (t-t_start)<Tmax 
    (M::Float64) = freq(t)
    Δt =  -log(rand())/M # rand(Exponential())/M
    t = t+Δt
    u = rand()*M # random between 0 and M
    (u_up::Float64) = freq(t) 
    if u <= u_up
      return t
    end
  end
  return Tmax + t_start
end



###
# recorders methods
struct RecNothing end

function reset!(::RecNothing)
  return nothing
end


struct RecFullTrain{N} <: Recorder
  nrec::Int64
  timesneurons::NTuple{N,Tuple{Vector{Float64},Vector{Int64}}}
  k_rec::Vector{Int64}
end
function RecFullTrain(nrec::Integer,(npops::Integer)=1)
  timesneurons=ntuple(_-> (fill(NaN,nrec),fill(-1,nrec)),npops)
  k_rec = fill(0,nrec)
  RecFullTrain(nrec,timesneurons,k_rec)
end
function reset!(rec::RecFullTrain)
  fill!(rec.k_rec,0)
  for (spktim,spkneu) in rec.timesneurons
    fill!(spktim,NaN)
    fill!(spkneu,-1)
  end
  return nothing
end


# general signature: record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)

function record_stuff!(rec::RecFullTrain,tfire::Real,
    popfire::Integer,neufire::Integer,::Symbol,
    ::RecurrentNetworkExpKernel)
  k = rec.k_rec[neufire]+1
  spiketimes,spikeneurons = rec.timesneurons[popfire]
  if !checkbounds(Bool,spiketimes,k)
    @error "recorder full!"
    return nothing
  end
  @inbounds spiketimes[k] = tfire
  @inbounds spikeneurons[k] = neufire
  rec.k_rec[neufire] = k
end

function dynamics_step!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  npops = npopulations(ntw)
  proposals_best = Vector{Float64}(undef,npops)
  neuron_best = Vector{Int64}(undef,npops)
  # for each postsynaptic network, compute spike proposals 
  for (kn,pop) in enumerate(ntw.populations)
    # update proposed next spike for each postsynaptic neuron
    nneu = nneurons(pop)
    for ineu in 1:nneu
      pop.spike_proposals[ineu] = compute_next_spike(t_now,pop,ineu) 
    end
    # best candidate and neuron that fired it for one input network
    (proposals_best[kn], neuron_best[kn]) = findmin(pop.spike_proposals) 
  end 
  # select next spike (best across all input_networks)
  (tfire,popfire) = findmin(proposals_best)
  neufire = neuron_best[popfire]
  psfire = ntw.populations[popfire].state
  label_fire = psfire.label
  # update stuff for that specific neuron/population state :
  burn_spike!(tfire,psfire,neufire)
  # apply plasticity rules ! Each rule in each connection in each population
  for pop in enumerate(ntw.populations)
    post = pop.state
    for (conn,pre) in zip(pop.connections,pop.pre_states)
      for plast in conn.plasticities
        plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)
      end
    end
  end
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  for rec in ntw.recorders
    record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)
  end
  # update t_now 
  return tfire
end

function dynamics_step_singlepopulation!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  # update proposed next spike for each postsynaptic neuron
  popfire = 1 
  pop = ntw.populations[1]
  nneu = nneurons(pop)
  for ineu in 1:nneu
    pop.spike_proposals[ineu] = compute_next_spike(t_now,pop,ineu) 
  end
  (tfire, neufire) = findmin(pop.spike_proposals) 
  psfire = pop.state
  label_fire = psfire.label
  # update stuff for that specific neuron/population state :
  burn_spike!(tfire,psfire,neufire)
  # apply plasticity rules ! Each rule in each connection in each population
  post = psfire
  for (conn,pre) in zip(pop.connections,pop.pre_states)
    for plast in conn.plasticities
      plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)
    end
  end
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  for rec in ntw.recorders
    record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)
  end
  # update t_now 
  return tfire
end