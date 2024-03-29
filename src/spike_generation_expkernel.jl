
# define a special case of exponential Kernel, because Claudia said so

####
# Population and connection constructors

# the taus are in the trace, now. No unittype or kernel object
# this is pretty much a wrapper around neural traces

abstract type PopulationStateMarkovian <: AbstractPopulationState end

struct PopulationStateExpKernel  <: PopulationStateMarkovian
  label::Symbol
  n::Int64
  traces::NTuple{N,Trace} where N
end
nneurons(ps::PopulationStateExpKernel) = ps.n

function PopulationStateExpKernel(n::Int64,(traces::Trace)...;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  label = Symbol(label)
  PopulationStateExpKernel(label,n,traces)
end
function reset!(ps::PopulationStateExpKernel)
  for tra in ps.traces
    reset!(tra)
  end
  return nothing
end
@inline function interaction_kernel(t::Real,psker::PopulationStateExpKernel)
  τ = psker.traces[1].τ
   return interaction_kernel(t,KernelExp(τ))
end
@inline function interaction_kernel_upper(t::Real,psker::PopulationStateExpKernel)
  τ = psker.traces[1].τ
   return interaction_kernel_upper(t,KernelExp(τ))
end
@inline function interaction_kernel_fourier(ω::Real,psker::PopulationStateExpKernel)
  τ = psker.traces[1].τ
   return interaction_kernel_fourier(ω,KernelExp(τ))
end



struct PopulationStateExpKernelInhibitory  <: PopulationStateMarkovian
  label::Symbol
  n::Int64
  traces::NTuple{N,Trace} where N
end
nneurons(ps::PopulationStateExpKernelInhibitory) = ps.n
function PopulationStateExpKernelInhibitory(n::Int64,(traces::Trace)...;
    label::Union{String,Nothing} = nothing)
  label = something(label,rand_label()) 
  label = Symbol(label)
  return PopulationStateExpKernelInhibitory(label,n,traces)
end
function reset!(ps::PopulationStateExpKernelInhibitory)
  for tra in ps.traces
    reset!(tra)
  end
  return nothing
end
@inline function interaction_kernel(t::Real,psker::PopulationStateExpKernelInhibitory)
  τ = psker.traces[1].τ
   return interaction_kernel(t,KernelExp(τ))
end
@inline function interaction_kernel_upper(t::Real,psker::PopulationStateExpKernelInhibitory)
  τ = psker.traces[1].τ
   return interaction_kernel_upper(t,KernelExp(τ))
end
@inline function interaction_kernel_fourier(ω::Real,psker::PopulationStateExpKernelInhibitory)
  τ = psker.traces[1].τ
   return interaction_kernel_fourier(ω,KernelExp(τ))
end


function population_state_exp_and_trace(n::Integer,τ::Float64; label::Union{String,Nothing}=nothing)
  trace = Trace(τ,n,ForDynamics())
  return PopulationStateExpKernel(n,trace;label=label),trace
end
function population_state_exp_and_trace_inhibitory(n::Integer,τ::Float64; 
    label::Union{String,Nothing}=nothing)
  trace = Trace(τ,n,ForDynamics())
  return PopulationStateExpKernelInhibitory(n,trace;label=label),trace
end


# global inhibition as a population state
struct PopulationStateGlobalStabilization
  label::Symbol
  Aglo::Float64
  Aloc::Float64
  tra_glo::Trace
  tra_loc::Trace
  function PopulationStateGlobalStabilization(n_post::Integer,
      τglo::Real,τloc::Real,
      Aglo::Real,Aloc::Real,
      label::Union{String,Nothing}=nothing)
    label = something(label,rand_label()) 
    tra_glo = Trace(τglo,1)
    tra_loc = Trace(τloc,n_post)
    return PopulationStateGlobalStabilization(label,Aglo,Aloc,tra_glo,tra_loc)
  end
end

function reset!(ps::PopulationStateGlobalStabilization)
  reset!((ps.tra_glo,ps.tra_loc))
end

struct PopulationExpKernel{N,
    PS<:Union{PopulationStateExpKernel,PopulationStateExpKernelInhibitory},
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,AbstractPopulationState},
    NL<:AbstractNonlinearity} <: AbstractPopulation
  state::PS
  connections::TC
  pre_states::TP
  input::Vector{Float64} # input only used for external currents 
  nonlinearity::NL # nonlinearity if present
  spike_proposals::Vector{Float64} # memory allocation 
end
function PopulationExpKernel(state::PopulationStateMarkovian,input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:AbstractPopulationState})...;
      nonlinearity::AbstractNonlinearity=NLRelu())
  connections = Tuple(getindex.(conn_pre,1))
  #typeassert.(connections,ConnectionExpKernel)
  pre_states = Tuple(getindex.(conn_pre,2))
  spike_proposals = fill(Inf,nneurons(state))
  return PopulationExpKernel(state,connections,pre_states,input,nonlinearity,spike_proposals) 
end
# one population only!
function PopulationExpKernel(state::PopulationStateMarkovian,conn::Connection,input::Vector{Float64}; 
    nonlinearity::AbstractNonlinearity=NLRelu())
  return PopulationExpKernel(state,input,(conn,state);nonlinearity=nonlinearity)
end

function set_initial_rates!(pop::PopulationExpKernel,rates::Union{Nothing,Vector{<:Real}})
  if !isnothing(rates)
    @assert nneurons(pop) == length(rates) "Dimensions wrong!"
    pop.state.traces[1].val .= rates
  end
  return nothing
end


struct ConnectionExpKernel{W,NPL,PL<:NTuple{NPL,PlasticityRule},T} <: ConnectionWithWeights{W,NPL,PL}
  weights::W
  pre_trace::T
  trace_proposal::Vector{Float64}
  plasticities::PL
end
function ConnectionExpKernel(weights::Matrix{Float64},pre_trace::Trace)
  plast = NTuple{0,NoPlasticity}()
  _,npre = size(weights)
  @assert npre == nneurons(pre_trace)
  tra_prop = Vector{Float64}(undef,npre)
  @assert(all(weights .>= 0.0), 
  """
    Weights must be positive! 
    For inhibitory connections use PopulationStateExpKernelInhibitory
  """)
  return ConnectionExpKernel(weights,pre_trace,tra_prop,plast)
end
function ConnectionExpKernel(weights::Matrix{Float64},pre_trace::Trace,
     (plasticity_rules::PL where PL<:PlasticityRule)...)
  _,npre = size(weights)
  @assert npre == nneurons(pre_trace)
  tra_prop = Vector{Float64}(undef,npre)
  @assert(all(weights .>= 0.0), 
  """
    Weights must be positive! 
    For inhibitory connections use PopulationStateExpKernelInhibitory
  """)
  return ConnectionExpKernel(weights,pre_trace,tra_prop,plasticity_rules)
end
function reset!(conn::ConnectionExpKernel)
  reset!(conn.pre_trace)
  fill!(conn.trace_proposal,NaN)
  reset!.(conn.plasticities)
  return nothing
end

@inline function trace_proposal!(t_now::Real,conn::ConnectionExpKernel)
  trace_proposal!(conn.trace_proposal,t_now,conn.pre_trace)::Nothing
  return conn.trace_proposal::Vector{Float64}
end

# if connection admits no inteaction, returns a vector of zeros
@inline function trace_proposal!(::Real,conn::ConnectionNonInteracting)
  npre = size(conn.weights,2)
  return zeros(npre) 
end

struct RecurrentNetworkExpKernel{N,TP<:NTuple{N,AbstractPopulation},NR,TR<:NTuple{NR,Recorder}}
  populations::TP
  recorders::TR
  alloc_spikeproposals::Vector{Float64}
  alloc_bestneurons::Vector{Int64}
  function RecurrentNetworkExpKernel(populations::TP,recorders::TR) where {TP,TR}
    npops = length(populations)
    nrec = length(recorders)
    alloc_spikeproposals = Vector{Float64}(undef,npops)
    alloc_bestneurons = Vector{Int64}(undef,npops)
    return new{npops,TP,nrec,TR}(populations,recorders,alloc_spikeproposals,alloc_bestneurons)
  end
end
# one population constructor
function RecurrentNetworkExpKernel(pop::AbstractPopulation,recorders...)
  if isempty(recorders)
    recorders=(RecNothing(),)
  end
  return RecurrentNetworkExpKernel((pop,),recorders)
end
# multiple pops, but no recorders
function RecurrentNetworkExpKernel(pops::Tuple)
  recorders=(RecNothing(),)
  return RecurrentNetworkExpKernel(pops,recorders)
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

function burn_spike!(t_spike::Real,ps::PopulationStateMarkovian,idx_update::Integer)
  # take care of traces  
  for tra in ps.traces
    propagate_for_dynamics!(t_spike,tra) # update full trace to t_spike 
    update_for_dynamics!(tra,idx_update) # add pulse at index
  end
  return nothing
end
function burn_spike!(t_spike::Real,ps::PopulationStateMarkovian)
  # take care of traces  
  for tra in ps.traces
    propagate_for_dynamics!(t_spike,tra) # update full trace to t_spike 
  end
  return nothing
end


# trick to update the global stabilization
function plasticity_update!(t_spike::Real,label_spike::Symbol,
    neufire::Integer,ps_post::PopulationStateMarkovian,
    ::Connection,ps_pre::PopulationStateGlobalStabilization,
    ::PlasticityRule)
  k_post,_ = find_which_spiked(label_spike,neufire,ps_post,ps_pre)
  if k_post == 0
    return nothing
  end
  # move traces to t_spike
  propagate!(t_spike,ps_pre.tra_loc)
  propagate!(t_spike,ps_pre.tra_glo)
  # add 1 to traces
  update_now!(ps_pre.tra_loc,k_post)
  update_now!(ps_pre.tra_glo,1)
  return nothing
end

@inline function propagated_signal(t_now::Real,idx_post::Integer,
    ::PopulationStateMarkovian,conn::Connection,::PopulationStateMarkovian)
  tra_tnow = trace_proposal!(t_now,conn)
  wij_all = view(conn.weights,idx_post,:)
  return dot(wij_all,tra_tnow)
end
propagated_signal_upper(a::Real,b::Integer,c::PopulationStateMarkovian,
  d::Connection,e::PopulationStateMarkovian) = propagated_signal(a,b,c,d,e)

# if inhibitory, same as above ,but all weights are considered negative
@inline function propagated_signal(t_now::Real,idx_post::Integer,
    ::PopulationStateMarkovian,conn::Connection,::PopulationStateExpKernelInhibitory)
  tra_tnow = trace_proposal!(t_now,conn)
  wij_all = view(conn.weights,idx_post,:)
  return -dot(wij_all,tra_tnow)
end
propagated_signal_upper(a::Real,b::Integer,c::PopulationStateMarkovian,d::Connection,
  e::PopulationStateExpKernelInhibitory) = propagated_signal(a,b,c,d,e)

# this one below is proably not needed
propagated_signal_upper(::Real,::Integer,::PopulationState,::Connection,
  ::PopulationStateExpKernelInhibitory) = 0.0

# when the connection is non-interacting, the propagated signal is ALWAYS zero
@inline function propagated_signal(::Real,::Integer,
    ::PopulationStateMarkovian,::ConnectionNonInteracting,::PopulationStateMarkovian)
  return 1E-9
end
@inline function propagated_signal(::Real,::Integer,
    ::PopulationStateMarkovian,::ConnectionNonInteracting,::PopulationStateExpKernelInhibitory)
  return 1E-9
end
  

function trace_proposals(t_now::Real,idx_neu::Integer,
    ps::PopulationStateGlobalStabilization)
  Δt = t_now - ps.tra_loc.t_last[]
  glo = ps.tra_glo[idx_neu] *  exp(-Δt/ps.tra_glo.τ)
  loc = ps.tra_loc.val[1] * exp(-Δt/ps.tra_loc.τ)
  return glo,loc
end

@inline function propagated_signal(t_now::Real,idx_post::Integer,
    ::PopulationState,conn::Connection,ps_pre::PopulationStateGlobalStabilization)
  tra_glo,tra_loc = trace_proposals(t_now,idx_post,ps_pre)
  return  -(ps_pre.Aglo*tra_glo + ps_pre.Aloc*tra_loc)
end
propagated_signal_upper(::Real,::Integer,::PopulationState,::Connection,
  ::PopulationStateGlobalStabilization) = 0.0




function call_for_each_compute_signal(ret,t_now,idxneu,ps_post,connections,pre_states)
    ret += propagated_signal(t_now,idxneu,ps_post,first(connections),first(pre_states)) 
    return call_for_each_compute_signal(ret,t_now,idxneu,ps_post,
      Base.tail(connections),Base.tail(pre_states)) 
end
function call_for_each_compute_signal(ret,t_now,idxneu,ps_post,::Tuple{},::Tuple{})
  return ret
end
  
function compute_rate(t_now::Float64,external_input::Float64,
    pop::PopulationExpKernel, idxneu::Integer)::Float64
  ps_post = pop.state
  ret = call_for_each_compute_signal(external_input,t_now,idxneu,ps_post,pop.connections,pop.pre_states)
  #nc = length(pop.connections)
  #for i in 1:nc
  #  ret += propagated_signal(t_now,idxneu,ps_post,pop.connections[i] ,pop.pre_states[i])::Float64
  #end
  ret = apply_nonlinearity(ret,pop.nonlinearity)
  return ret
end
function compute_rates!(r_alloc::Vector{Float64},t_now::Real,pop::PopulationExpKernel)
  inputs = pop.input
  for i in eachindex(r_alloc)
    r_alloc[i] = compute_rate(t_now,inputs[i],pop,i)
  end
  return nothing
end


# upper boundary ignores inhibitory component, because it's increasing!


function compute_rates_upper!(r_alloc::Vector{Float64},t_now::Real,pop::PopulationExpKernel)
  inputs = pop.input
  for i in eachindex(r_alloc)
    r_alloc[i] = compute_rate_upper(t_now,inputs[i],pop,i)
  end
  return nothing
end

function call_for_each_compute_signal_upper(ret,t_now,idxneu,ps_post,connections,pre_states)
    ret += propagated_signal_upper(t_now,idxneu,ps_post,first(connections),first(pre_states)) 
    return call_for_each_compute_signal_upper(ret,t_now,idxneu,ps_post,
      Base.tail(connections),Base.tail(pre_states)) 
end
function call_for_each_compute_signal_upper(ret,t_now,idxneu,ps_post,::Tuple{},::Tuple{})
  return ret
end

function compute_rate_upper(t_now::R,external_input::R,pop::PopulationExpKernel, 
    idxneu::Integer) where {R<:Real}
  external_input_nz = max(external_input,eps(R)) # deals with negative inputs
  ps_post = pop.state
  ret = call_for_each_compute_signal_upper(external_input_nz,t_now,idxneu,ps_post,
    pop.connections,pop.pre_states)
  ret_nonlin = apply_nonlinearity(ret,pop.nonlinearity)
  ret = max(external_input_nz,ret_nonlin)
  return ret
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
# replaced by....
# multivariate thinning algorith. From  Y. Chen, 2016
function compute_next_spike(t_now::Real,pop::PopulationExpKernel;Tmax::Real=100.0)
  t_start = t_now
  t = t_now
  rates = pop.spike_proposals # recycle & reuse  # Vector{Float64}(undef,n)
  dorates_upper!(t) = compute_rates_upper!(rates,t,pop)
  dorates!(t) = compute_rates!(rates,t,pop)
  while (t-t_start)<Tmax 
    dorates_upper!(t)
    R_up = sum(rates)
    if R_up==0
      @error "Something wrong with neural inputs!"
      break
    end
    Δt =  -log(rand())/R_up # rand(Exponential())/R_up
    t = t+Δt
    u = rand()*R_up # random between 0 and R_up
    dorates!(t)
    cumsum!(rates,rates)
    if u < rates[end] # else, continue
      k = searchsortedfirst(rates,u)
      return (t,k)
    end
  end
  @warn "Population did not spike ! Returning fake spike at t=$(Tmax+t_start) (is this a test? or too much inh?)" maxlog=20
  return (Tmax + t_start,1)
end

###

# general signature: record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)
# @inline function record_stuff!(::RecNothing,::Real,
#     ::Integer,::Integer,::Symbol,
#     ::RecurrentNetworkExpKernel)
#   return nothing  
# end

struct RecSomeTrain{I,R} <: Recorder
  nmaxrec::I
  poprec::I
  neusrec::Vector{I}
  timesneurons::Tuple{Vector{R},Vector{I}}
  krec::Ref{I}
  Tstart::R
  Tend::R
end
function RecSomeTrain(nrec,neurons_to_record;population_rec=1,Tstart=0.0,Tend=Inf)
  krec = Ref(0)
  alltimes = fill(NaN,nrec)
  allneus = fill(-1,nrec)
  return RecSomeTrain(nrec,population_rec,neurons_to_record,(alltimes,allneus),krec,Tstart,Tend)
end
struct RecSomeTrainContent
  timesneurons::NTuple{N,Tuple{Vector{Float64},Vector{Int64}}} where N
  Tstart::Float64
  Tend::Float64
end
function RecSomeTrainContent(r::RecSomeTrain)
  (_times,_neus) = r.timesneurons
  idx_keep = isfinite.(_times)
  keepat!(_times,idx_keep)
  keepat!(_neus,idx_keep)
  return RecSomeTrainContent(r.timesneurons,r.Tstart,r.Tend)
end
function get_content(rec::RecSomeTrain)
  return RecSomeTrainContent(rec)
end



struct RecFullTrain{N} <: Recorder
  nrec::Int64
  timesneurons::NTuple{N,Tuple{Vector{Float64},Vector{Int64}}}
  k_rec::Vector{Int64}
  Tstart::Float64
  Tend::Float64
end
function RecFullTrain(nrec::Integer,(npops::Integer)=1;Tstart=0.0,Tend=Inf)
  timesneurons=ntuple(_-> (fill(NaN,nrec),fill(-1,nrec)),npops)
  k_rec = fill(0,npops)
  RecFullTrain(nrec,timesneurons,k_rec,Tstart,Tend)
end
function reset!(rec::RecFullTrain)
  fill!(rec.k_rec,0)
  for (spktim,spkneu) in rec.timesneurons
    fill!(spktim,NaN)
    fill!(spkneu,-1)
  end
  return nothing
end

struct RecFullTrainContent
  timesneurons::NTuple{N,Tuple{Vector{Float64},Vector{Int64}}} where N
  Tstart::Float64
  Tend::Float64
end
function RecFullTrainContent(r::RecFullTrain;Tstart=0.0,Tend=Inf)
  npops = length(r.timesneurons)
  for p in 1:npops
    _times = r.timesneurons[p][1]
    _neus = r.timesneurons[p][2]
    idx_keep = isfinite.(_times)
    keepat!(_times,idx_keep)
    keepat!(_neus,idx_keep)
  end
  return RecFullTrainContent(r.timesneurons,r.Tstart,r.Tend)
end

function get_content(rec::RecFullTrain)
  return RecFullTrainContent(rec)
end

function Base.vcat((trains::RecFullTrainContent)...)
  npops = length(trains[1].timesneurons)
  ret = map(1:npops) do p
    alltimes = cat([tr.timesneurons[p][1] for tr in trains]...;dims=1)
    allneus = cat([tr.timesneurons[p][2] for tr in trains]...;dims=1)
    (alltimes,allneus)
  end
  if npops == 1
    return RecFullTrainContent((Tuple(ret...),))
  else
    # Warning! Not tested!
    rt = (ret...,)
    rntups = convert( NTuple{npops,Tuple{Vector{Float64},Vector{Int64}}},rt)
    return RecFullTrainContent(rntups)
  end
end

function numerical_rates(rec::Union{RecFullTrain,RecFullTrainContent})
  @error "This might be the wrong function!"
  rates = Vector{Float64}[]
  for p in 1:N
    (spkt,spkneu) = rec.timesneurons[p]
    idx_good = isfinite.(spkt)
    if !any(idx_good)
      return 0.0
    end
    t_good = spkt[idx_good]
    neu_good = spkneu[idx_good]
    neu_tot = maximum(neu_good)
    Ttot = t_good[end]
    rates_neu = map(1:neu_tot) do idx_neu
      return count(==(idx_neu),neu_good) / Ttot
    end
    push!(rates,rates_neu)
  end
  return rates
end

"""
  function get_trains(rec::RecFullTrain{N}; 
      Nneus::Integer = 0,
      pop_idx::Integer = 1) where N

Read full recorded train from the population selected 
by `pop_idx` (default 1).
"""
function get_trains(rec::Union{RecFullTrain,RecFullTrainContent}; 
    Nneus::Integer = 0,
    pop_idx::Integer = 1)
  (spkt,spkneu) = rec.timesneurons[pop_idx]
  if Nneus == 0
    Nneus = maximum(spkneu)
  end
  trains = Vector{Vector{Float64}}(undef,Nneus)
  for neu in 1:Nneus
    _idx = findall(==(neu),spkneu)
    trains[neu] = isempty(_idx) ? Float64[] : spkt[_idx]
  end
  return trains
end

# general signature: record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)

function record_stuff!(rec::RecSomeTrain,tfire::Real,popfire::Integer,
    neufire::Integer,::Symbol,::RecurrentNetworkExpKernel)
  # check all conditions
  if ( (popfire != rec.poprec) || 
      !(neufire in rec.neusrec) ||
       (tfire < rec.Tstart) ||
       (tfire > rec.Tend ))
    return nothing
  end
  k = rec.krec[]+1
  spiketimes,spikeneurons = rec.timesneurons
  if !checkbounds(Bool,spiketimes,k)
    @error "recorder full!"
    return nothing
  end
  @inbounds spiketimes[k] = tfire
  @inbounds spikeneurons[k] = neufire
  rec.krec[] = k
  return nothing
end

function record_stuff!(rec::RecFullTrain,tfire::Real,popfire::Integer,
    neufire::Integer,::Symbol,::RecurrentNetworkExpKernel)
  # still checks time boundaries! 
  if (tfire < rec.Tstart) || (tfire > rec.Tend )
    return nothing
  end
  k = rec.k_rec[popfire]+1
  spiketimes,spikeneurons = rec.timesneurons[popfire]
  if !checkbounds(Bool,spiketimes,k)
    @error "recorder full!"
    return nothing
  end
  @inbounds spiketimes[k] = tfire
  @inbounds spikeneurons[k] = neufire
  rec.k_rec[popfire] = k
  return nothing
end


# To fix memory leaks
function call_for_each_record_stuff!(recorders,tfire,popfire,neufire,label_fire,ntw)
  record_stuff!(first(recorders),tfire,popfire,neufire,label_fire,ntw)
  call_for_each_record_stuff!(Base.tail(recorders),tfire,popfire,neufire,label_fire,ntw)
  return nothing
end
function call_for_each_record_stuff!(::Tuple{},args...)
  return nothing
end

# Also to avoid memory leaks
# sorry it is ugly :'-(
# replacing this
#  for pop in ntw.populations
#    post = pop.state
#    for (conn,pre) in zip(pop.connections,pop.pre_states)
#      for plast in conn.plasticities
#        plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)
#      end
#    end
#  end
function multipop_call_for_plasticity_update!(tfire,labelfire,neufire,populations)
  pop = first(populations)
  tailpopulations = Base.tail(populations)  
  post = pop.state
  connections = pop.connections
  pre_states  = pop.pre_states
  @assert length(connections) == length(pre_states) # just for debugging
  singlepop_call_for_plasticity_update!(tfire,labelfire,neufire,post,connections,pre_states)
  if isempty(tailpopulations)
    return nothing
  else
    multipop_call_for_plasticity_update!(tfire,labelfire,neufire,tailpopulations)
  end
end
function singlepop_call_for_plasticity_update!(tfire,labelfire,neufire,post,
    connections,pre_states)
  connection = first(connections)
  pre_state = first(pre_states)
  tailpre_states = Base.tail(pre_states)
  tailconnections = Base.tail(connections)  
  plasticities = connection.plasticities
  if !isempty(plasticities)
    subcall_for_plasticity_update!(tfire,labelfire,neufire,post,
      connection,pre_state,plasticities)
  end
  if isempty(tailconnections)
    return nothing
  else
    singlepop_call_for_plasticity_update!(tfire,labelfire,neufire,post,
      tailconnections,tailpre_states)
  end
end
function subcall_for_plasticity_update!(tfire,labelfire,neufire,post,
    conn,pre,plasticities)
  plasticity = first(plasticities)
  tailplasticities = Base.tail(plasticities)
  plasticity_update!(tfire,labelfire,neufire,post,conn,pre,plasticity)
  if isempty(tailplasticities)
    return nothing
  else
    subcall_for_plasticity_update!(tfire,labelfire,neufire,post,conn,pre,tailplasticities)
  end
end

# no memory no leak!
# for (kn,pop) in enumerate(ntw.populations)
#   nextspk =   compute_next_spike(t_now,pop)
#   (proposals_best[kn],neuron_best[kn]) = nextspk
# end 
# # select next spike (best across all input_networks)
# (tfire,popfire) = findmin(proposals_best)
# neufire = neuron_best[popfire]
function call_for_compute_next_spike(t_now,populations,
    currentpop,bestspiketime,bestpop,bestpoplabel,bestneuron)
  pop = first(populations)
  currentpop = currentpop+1
  (bestspiketime_here,bestneuron_here) = compute_next_spike(t_now,pop)
  if bestspiketime_here < bestspiketime
    bestspiketime = bestspiketime_here
    bestpop = currentpop
    bestpoplabel = pop.state.label
    bestneuron = bestneuron_here
  end
  tailpopulations = Base.tail(populations)
  if isempty(tailpopulations)
    return (bestspiketime,bestpop,bestpoplabel,bestneuron)
  else
    return call_for_compute_next_spike(t_now,tailpopulations,
      currentpop,bestspiketime,bestpop,bestpoplabel,bestneuron)
  end
end

# Thou shall not leak!
# for (kn,pop) in enumerate(ntw.populations)
#   if kn==popfire
#     burn_spike!(tfire,pop.state,neufire)
#   else
#     burn_spike!(tfire,pop.state)
#   end
# end
function call_for_multipop_burn_spike!(currentpop::Integer,tfire,popfire,neufire,populations)
  currentpop += 1
  population = first(populations)
  tailpopulations = Base.tail(populations)
  if currentpop == popfire
    burn_spike!(tfire,population.state,neufire)
  else
    burn_spike!(tfire,population.state)
  end
  if isempty(tailpopulations)
    return nothing
  else
    return call_for_multipop_burn_spike!(currentpop,tfire,popfire,neufire,tailpopulations)
  end
end


function dynamics_step!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  (tfire,popfire,labelfire,neufire) = call_for_compute_next_spike(t_now,ntw.populations,0,Inf,-1,-1,-1)
  # update stuff for that specific neuron/population state :
  call_for_multipop_burn_spike!(0,tfire,popfire,neufire,ntw.populations)
  # apply plasticity rules ! Each rule in each connection in each population
  multipop_call_for_plasticity_update!(tfire,labelfire,neufire,ntw.populations)
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  call_for_each_record_stuff!(ntw.recorders,tfire,popfire,neufire,labelfire,ntw)
  # update t_now 
  return tfire
end

function dynamics_step_singlepopulation!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  # update proposed next spike for each postsynaptic neuron
  popfire = 1 
  pop = ntw.populations[1]
  (tfire,neufire) = compute_next_spike(t_now,pop)
  psfire = pop.state
  label_fire = psfire.label
  # update stuff for that specific neuron/population state :
  burn_spike!(tfire,psfire,neufire)
  # apply plasticity rules ! Each rule in each connection in each population
  post = psfire
  singlepop_call_for_plasticity_update!(tfire,label_fire,neufire,post,
    pop.connections,pop.pre_states)
  # for (conn,pre) in zip(pop.connections,pop.pre_states)
  #   for plast in conn.plasticities
  #     plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)
  #   end
  # end
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  call_for_each_record_stuff!(ntw.recorders,tfire,popfire,neufire,label_fire,ntw)
  # update t_now 
  return tfire
end
