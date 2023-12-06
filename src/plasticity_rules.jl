

# General application of plasticity rule on network

function plasticity_update!(t_now::Real,ntw::RecurrentNetwork)
  for pop in ntw.populations
    post = pop.state
    for (conn,pre) in zip(pop.connections,pop.pre_states)
      for plast in conn.plasticities
        plasticity_update!(t_now,pre,conn,post,plast)
      end
    end
  end
  return nothing
end


# bounds are structures too!
# because!

abstract type PlasticityBounds{R} end
struct PlasticityBoundsNonnegative{R} <: PlasticityBounds{R} end

function PlasticityBoundsNonnegative()
  return PlasticityBoundsNonnegative{Float64}()
end

struct PlasticityBoundsLowHigh{R} <: PlasticityBounds{R} 
  low::R
  high::R
end
@inline function (plb::PlasticityBoundsNonnegative{R})(w::R,Δw::R)::R where R<:Real
  ret = w+Δw
  return max(100*eps(R),ret)::R
end
# @inline function (plb::PlasticityBoundsLowHigh{R})(w::R,Δw::R)::R where R<:Real
#   ret = w+Δw
#   return min(plb.high,max(plb.low,ret))::R
# end

@inline function (plb::PlasticityBoundsLowHigh{R})(w::R,Δw::R) where R<:Real
  ret = w+Δw
  if ret < plb.low
    return plb.low
  elseif ret > plb.high
    return plb.high
  else
    return ret
  end
end

function apply_bounds(plb::PlasticityBoundsLowHigh{R},w::R,Δw::R) where R<:Real
  ret = w+Δw
  if ret < plb.low
    return plb.low
  elseif ret > plb.high
    return plb.high
  else
    return ret
  end
end


# utility function (ALWAYS follow post<-pre , never pre,post)
@inline function find_which_spiked(t_spike::Real,pspost::PopulationState,pspre::PopulationState)
  return find_which_spiked(t_spike,pspost.trains,pspre.trains)
end
function find_which_spiked(t_spike::R,trains_post::Vector{Vector{R}},trains_pre::Vector{Vector{R}}) where R
  is_t_now(train) = isempty(train) ? false : last(train) == t_spike
  k_post::Int64 = something(findfirst(is_t_now,trains_post),0)
  k_pre::Int64  = something(findfirst(is_t_now,trains_pre),0)
  return (k_post,k_pre)
end
function find_which_spiked(label_spike::Symbol,idx_spike::Int64,
    pspost::AbstractPopulationState,pspre::AbstractPopulationState)
  k_post = label_spike == pspost.label ? idx_spike : 0  
  k_pre = label_spike == pspre.label ? idx_spike : 0  
  return (k_post,k_pre)
end

# two different signatures, but unified base plasticity update

# signature for ExpKernel: plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)
@inline function plasticity_update!(tfire::Real,label_spike::Symbol,
    neufire::Integer,
    pspost::AbstractPopulationState,
    conn::Connection,
    pspre::AbstractPopulationState,
    plast::PlasticityRule)
  k_post,k_pre = find_which_spiked(label_spike,neufire,pspost,pspre)
  plasticity_update!(tfire,k_post,k_pre,pspost,conn,pspre,plast)
  return nothing
end

# signature for general Hawkes
@inline function plasticity_update!(t_spike::Real,
    pspost::PopulationState,conn::Connection,pspre::PopulationState,
    plast::PlasticityRule)
  k_post,k_pre = find_which_spiked(t_spike,pspost,pspre)
  plasticity_update!(t_spike,k_post,k_pre,pspost,conn,pspre,plast)
  return nothing
end

# no plasticity

plasticity_update!(::Real,::Integer,::Integer,::AbstractPopulationState,::Connection,
  ::AbstractPopulationState,::NoPlasticity) = nothing

# Pairwise plasticity 
struct PairSTDP{R} <: PlasticityRule
  Aplus::R
  Aminus::R
  traceplus::Trace{ForPlasticity,R}
  traceminus::Trace{ForPlasticity,R}
  bounds::PlasticityBounds{R}
  function PairSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
       plasticity_bounds=PlasticityBoundsNonnegative())
    traceplus = Trace(τplus,n_pre) # (r) add on post firing based on pre trace
    traceminus = Trace(τminus,n_post) # (o) subtract on pre firing based on post trace 
    new{Float64}(Aplus,Aminus,traceplus,traceminus,plasticity_bounds)
  end
end
function reset!(pl::PairSTDP)
  reset!(pl.traceplus)
  reset!(pl.traceminus)
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PairSTDP{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    @inbounds Δw = plast.traceminus.val[i_post]*plast.Aminus
    @inbounds weight_matrix[i_post,k_pre] = plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PairSTDP{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre] 
  if !iszero(wkj)
    @inbounds Δw = plast.traceplus.val[j_pre]*plast.Aplus
    @inbounds weight_matrix[k_post,j_pre] = plast.bounds(wkj,Δw)
  end
  return nothing
end


function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,::AbstractPopulationState,
    conn::ConnectionWithWeights{M,NPL,PL},::AbstractPopulationState,plast::PairSTDP{R}) where {NPL,PL,R,M<:Matrix{R}}
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  weights = conn.weights
  npost,npre = size(weights)
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.traceplus)
  propagate!(t_spike,plast.traceminus)
  # update the plasticity trace variables, if needed
  update_now!(plast.traceplus,k_pre_spike)
  update_now!(plast.traceminus,k_post_spike)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      wkj = weights[k_post_spike,j] 
      if !iszero(wkj)
        weight_update_postfired!(weights,k_post_spike,j,plast)
      end
    end
  end
  return nothing
end

## visualize STDP curve
function get_stdp_curve(stdp::PairSTDP,times::AbstractVector{Float64})
  τplus = stdp.traceplus.τ
  τminus = stdp.traceminus.τ
  stdpfun(t) = ifelse(t >= 0,stdp.Aplus*exp(-t/τplus) , stdp.Aminus*exp(t/τminus))
  return map(stdpfun,times)
end


### Triplets rule
# WARNING : for standard case, A2minus and A3 minus parameters should be NEGATIVE
struct PlasticityTriplets <: PlasticityRule
  A2plus::Float64
  A3plus::Float64
  A2minus::Float64
  A3minus::Float64
  o1::Trace # pOst tau_minus
  o2::Trace # pOst tau_y
  r1::Trace # pRe  tau_plus
  r2::Trace # pRe  tau_y
  bounds::PlasticityBounds
end
function PlasticityTriplets(τplus::R,τminus::R,τx::R,τy::R,
    A2plus::R,A3plus::R,A2minus::R,A3minus::R,n_post::I,n_pre::I;
      plasticity_bounds=PlasticityBoundsNonnegative()) where {R,I}
  o1 = Trace(τminus,n_post,ForPlasticity())    
  o2 = Trace(τy,n_post,ForPlasticity())    
  r1 = Trace(τplus,n_pre,ForPlasticity())    
  r2 = Trace(τx,n_pre,ForPlasticity())    
  PlasticityTriplets(A2plus,A3plus,A2minus,A3minus,
    o1,o2,r1,r2,plasticity_bounds)
end
function reset!(pl::PlasticityTriplets)
  reset!.((pl.o1,pl.o2,pl.r1,pl.r2))
  return nothing
end


function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityTriplets)
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!.(t_spike,(plast.o1,plast.o2,plast.r1,plast.r2))
  # update the plasticity trace variables for o1 and r1
  if !iszero(k_post_spike)
    update_now!(plast.o1,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.r1,k_pre_spike)
  end
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: move vertically along k_pre column 
    for i_post in 1:npost
      wik = weights[i_post,k_pre_spike] 
      if wik > 0
        Δw = plast.o1.val[i_post]*(plast.A2minus+plast.A3minus*plast.r2.val[k_pre_spike])
        weights[i_post,k_pre_spike] =  plast.bounds(wik,Δw)
      end
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: move horizontally along k_post row
    for j_pre in 1:npre
      wkj = weights[k_post_spike,j_pre] 
      if wkj > 0
        Δw = plast.r1.val[j_pre]*(plast.A2plus+plast.A3plus*plast.o2.val[k_post_spike])
        weights[k_post_spike,j_pre] =  plast.bounds(wkj,Δw)
      end
    end
  end
  # update the plasticity trace variables for o2 and r2
  if !iszero(k_post_spike)
    update_now!(plast.o2,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.r2,k_pre_spike)
  end
  return nothing
end

# Symmetric STDP

# Pairwise plasticity 
struct SymmetricSTDP{R} <: PlasticityRule
  Aplus::R
  Aminus::R
  post_plus_trace::Trace{ForPlasticity,R}
  post_minus_trace::Trace{ForPlasticity,R}
  pre_plus_trace::Trace{ForPlasticity,R}
  pre_minus_trace::Trace{ForPlasticity,R}
  bounds::PlasticityBounds
  function SymmetricSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
       plasticity_bounds=PlasticityBoundsNonnegative())
    post_plus_t = Trace(τplus,n_post)
    post_minus_t = Trace(τminus,n_post)
    pre_plus_t = Trace(τplus,n_pre)
    pre_minus_t = Trace(τminus,n_pre)
    new{Float64}(Aplus,Aminus,post_plus_t,post_minus_t,
     pre_plus_t,pre_minus_t,plasticity_bounds)
  end
end
function reset!(pl::SymmetricSTDP)
  reset!(pl.pre_plus_trace)
  reset!(pl.pre_minus_trace)
  reset!(pl.post_plus_trace)
  reset!(pl.post_minus_trace)
  return nothing
end

function plasticity_update!(t_spike::R,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::SymmetricSTDP{R}) where R
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_plus_trace)
  propagate!(t_spike,plast.pre_minus_trace)
  propagate!(t_spike,plast.post_plus_trace)
  propagate!(t_spike,plast.post_minus_trace)
  # increase the plasticity trace variables
  if !iszero(k_post_spike)
    update_now!(plast.post_plus_trace,k_post_spike)
    update_now!(plast.post_minus_trace,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.pre_plus_trace,k_pre_spike)
    update_now!(plast.pre_minus_trace,k_pre_spike)
  end
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      wik = weights[i,k_pre_spike] 
      if !iszero(wik)
        Δw = ( plast.post_minus_trace.val[i]*plast.Aminus +
               plast.post_plus_trace.val[i]*plast.Aplus) 
        weights[i,k_pre_spike] =  plast.bounds(wik,Δw)
      end
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      wkj = weights[k_post_spike,j] 
      if !iszero(wkj)
        Δw = ( plast.pre_minus_trace.val[j]*plast.Aminus +
               plast.pre_plus_trace.val[j]*plast.Aplus) 
        weights[k_post_spike,j] =  plast.bounds(wkj,Δw)
      end
    end
  end
  return nothing
end

## visualize STDP curve
function get_stdp_curve(stdp::SymmetricSTDP,times::AbstractVector{Float64})
  τplus = stdp.traceplus.τ
  τminus = stdp.traceminus.τ
  stdpfun(t) =  stdp.Aplus*exp(-abs(t)/τplus) + stdp.Aminus*exp(-abs(t)/τminus)
  return map(stdpfun,times)
end


# Inhibitory stabilization, Vogels-Sprekeler 2011

# Warning : this formulation assumes all positive weights !

struct PlasticityInhibitory{R} <: PlasticityRule
  τ::R
  η::R
  α::R
  o::Trace{ForPlasticity,R} # pOst
  r::Trace{ForPlasticity,R} # pRe
  bounds::PlasticityBounds
  function PlasticityInhibitory(τ,η,n_post,n_pre;
      r_target=5.0,
      plasticity_bounds::PlasticityBounds=PlasticityBoundsNonnegative())
    α = 2*r_target*τ
    o = Trace(τ,n_post,ForPlasticity())    
    r = Trace(τ,n_pre,ForPlasticity())    
    new{Float64}(τ,η,α,o,r,plasticity_bounds)
  end
end
function reset!(pl::PlasticityInhibitory)
  reset!.((pl.o,pl.r))
  return nothing
end


# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PlasticityInhibitory{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    @inbounds Δw = plast.η*(plast.o.val[i_post]-plast.α)
    @inbounds weight_matrix[i_post,k_pre] = plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PlasticityInhibitory{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre]
  if !iszero(wkj)
    @inbounds Δw = plast.η*plast.r.val[j_pre]
    @inbounds weight_matrix[k_post,j_pre] = plast.bounds(wkj,Δw)
  end
  return nothing
end


function plasticity_update!(t_spike::R,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityInhibitory{R}) where R<:Real
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  weights = conn.weights
  npost,npre = size(weights)

  # update all pre and post traces to t_now
  propagate!.(t_spike,(plast.o,plast.r))
  # spike update, pre or post
  if !iszero(k_post_spike)
    update_now!(plast.o,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.r,k_pre_spike)
  end
  if !iszero(k_pre_spike)
    # k is presynaptic: move vertically along k_pre column 
    for i_post in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i_post,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is presynaptic: move vertically along k_pre column 
    for j_pre in 1:npre
      weight_update_postfired!(weights,k_post_spike,j_pre,plast)
    end
  end
  return nothing
end

# Generalized STDP, symmetric
struct PlasticitySymmetricSTDPX{R} <: PlasticityRule
  A::R      # global scaling
  θ::R      # bias term  (total bias is 1+θ)
  τ::R      # time constant 
  γ::R      # time constant minus scaling
  αpre::R   # presynaptic firing bias
  αpost::R  # postsynaptic firing bias
  Aplus::R  # A/τ  normalized "potentiation"
  Aminus::R # θ*A/(γ*τ) normalized "depression"
  post_plus_trace::Trace{ForPlasticity,R}
  post_minus_trace::Trace{ForPlasticity,R}
  pre_plus_trace::Trace{ForPlasticity,R}
  pre_minus_trace::Trace{ForPlasticity,R}
  bounds::PlasticityBounds{R}
  function PlasticitySymmetricSTDPX(A::R,θ::R,τ::R,γ::R,
      αpre::R,αpost::R,n_post::Integer,n_pre::Integer;
      plasticity_bounds=PlasticityBoundsNonnegative{R}()) where R
    post_plus_t = Trace(τ,n_post)
    post_minus_t = Trace(τ*γ,n_post)
    pre_plus_t = Trace(τ,n_pre)
    pre_minus_t = Trace(τ*γ,n_pre)
    Aplus = A/τ
    Aminus = θ*A/(γ*τ)
    @assert γ>0 "Something wrong with parameter γ, should be >0"
    new{Float64}(A,θ,τ,γ,αpre,αpost,
      Aplus,Aminus,
      post_plus_t,post_minus_t,
      pre_plus_t,pre_minus_t,plasticity_bounds)
  end
end
function reset!(pl::PlasticitySymmetricSTDPX)
  reset!(pl.pre_plus_trace)
  reset!(pl.pre_minus_trace)
  reset!(pl.post_plus_trace)
  reset!(pl.post_minus_trace)
  return nothing
end

function area_under_curve(plast::PlasticitySymmetricSTDPX)
  return 2*plast.A*(1+plast.θ)
end


# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PlasticitySymmetricSTDPX{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    Aabs = abs(plast.A)
    @inbounds Δw = (  Aabs*plast.αpre + # add presynaptic firing bias
            plast.post_plus_trace.val[i_post]*plast.Aplus +
            plast.post_minus_trace.val[i_post]*plast.Aminus)
    @inbounds weight_matrix[i_post,k_pre] =  plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PlasticitySymmetricSTDPX{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre] 
  if !iszero(wkj)
    Aabs = abs(plast.A)
    @inbounds Δw = ( Aabs*plast.αpost + 
            plast.pre_plus_trace.val[j_pre]*plast.Aplus +
            plast.pre_minus_trace.val[j_pre]*plast.Aminus) 
    @inbounds weight_matrix[k_post,j_pre] =  plast.bounds(wkj,Δw)
  end
  return nothing
end


function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticitySymmetricSTDPX{R}) where R<:Real
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_plus_trace)
  propagate!(t_spike,plast.pre_minus_trace)
  propagate!(t_spike,plast.post_plus_trace)
  propagate!(t_spike,plast.post_minus_trace)
  # increase the plasticity trace variables
  if !iszero(k_post_spike)
    update_now!(plast.post_plus_trace,k_post_spike)
    update_now!(plast.post_minus_trace,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.pre_plus_trace,k_pre_spike)
    update_now!(plast.pre_minus_trace,k_pre_spike)
  end
  # used for scaling the alphas
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      weight_update_postfired!(weights,k_post_spike,j,plast)
    end
  end
  return nothing
end



# Generalized STDP, symmetric
struct PlasticitySymmetricSTDPGL{R} <: PlasticityRule
  Azero::R      # global scaling
  Aplus::R
  Aminus::R
  τ::R      # time constant 
  γ::R      # time constant minus scaling
  αzero::R  # constant leack term
  αpre::R   # presynaptic firing bias
  αpost::R  # postsynaptic firing bias
  _Cplus::R  #  normalized potentiation
  _Cminus::R #  normalized depression
  post_plus_trace::Trace{ForPlasticity,R}
  post_minus_trace::Trace{ForPlasticity,R}
  pre_plus_trace::Trace{ForPlasticity,R}
  pre_minus_trace::Trace{ForPlasticity,R}
  t_last_posts::Vector{R}
  bounds::PlasticityBounds{R}
  function PlasticitySymmetricSTDPGL(Azero::R,Aplus::R,Aminus::R,τ::R,γ::R,
      αzero::R,αpre::R,αpost::R,
      n_post::Integer,n_pre::Integer;
      t_start::Real=0.0,
      plasticity_bounds=PlasticityBoundsNonnegative{R}()) where R
    @assert Azero >= 0 "Azero should be >= 0"
    @assert Aplus*Aminus <= 0 "Aplus and Aminus should have opposite signs"
    post_plus_t = Trace(τ,n_post)
    post_minus_t = Trace(τ*γ,n_post)
    pre_plus_t = Trace(τ,n_pre)
    pre_minus_t = Trace(τ*γ,n_pre)
    t_last_posts = fill(t_start,n_post)
    _Cplus = 0.5*Aplus/τ
    _Cminus = 0.5*Aminus/(γ*τ)
    @assert γ>0 "Something wrong with parameter γ, should be >0"
    new{Float64}(Azero,Aplus,Aminus,τ,γ,
      αzero, αpre,αpost,
      _Cplus,_Cminus,
      post_plus_t,post_minus_t,pre_plus_t,pre_minus_t,
      t_last_posts,
      plasticity_bounds)
  end
end
function reset!(pl::PlasticitySymmetricSTDPGL;t_start::Float64=0.0)
  reset!(pl.pre_plus_trace)
  reset!(pl.pre_minus_trace)
  reset!(pl.post_plus_trace)
  reset!(pl.post_minus_trace)
  pl.t_last_posts .= t_start
  return nothing
end

function area_under_curve(plast::PlasticitySymmetricSTDPGL)
  return plast.Azero*(plast.Aplus+plast.Aminus)
end


# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PlasticitySymmetricSTDPGL{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    @inbounds Δw = plast.Azero* ( plast.αpre +
            plast.post_plus_trace.val[i_post]*plast._Cplus +
            plast.post_minus_trace.val[i_post]*plast._Cminus)
    @inbounds weight_matrix[i_post,k_pre] =  plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(Δt_last_post::Float64,
    weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PlasticitySymmetricSTDPGL{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre] 
  if !iszero(wkj)
    @inbounds Δw = 
            plast.Azero*(
              plast.αzero*Δt_last_post +
              plast.αpost + 
              plast.pre_plus_trace.val[j_pre]*plast._Cplus +
              plast.pre_minus_trace.val[j_pre]*plast._Cminus) 
    @inbounds weight_matrix[k_post,j_pre] =  plast.bounds(wkj,Δw)
  end
  return nothing
end


function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticitySymmetricSTDPGL{R}) where R<:Real
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_plus_trace)
  propagate!(t_spike,plast.pre_minus_trace)
  propagate!(t_spike,plast.post_plus_trace)
  propagate!(t_spike,plast.post_minus_trace)
  # increase the plasticity trace variables
  if !iszero(k_post_spike)
    update_now!(plast.post_plus_trace,k_post_spike)
    update_now!(plast.post_minus_trace,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.pre_plus_trace,k_pre_spike)
    update_now!(plast.pre_minus_trace,k_pre_spike)
  end
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    Δt_last_post = t_spike-plast.t_last_posts[k_post_spike]
    for j in 1:npre
      weight_update_postfired!(Δt_last_post,weights,k_post_spike,j,plast)
    end
    plast.t_last_posts[k_post_spike] = t_spike
  end
  return nothing
end


# Generalized STDP, asymmetric (classical STDP Hebbian shape)
struct PlasticityAsymmetricX{R} <: PlasticityRule
  A::R# scaling
  θ::R# bias term  (total bias is 1+θ)
  τ::R# time constant 
  γ::R# time constant minus scaling
  αpre::R# presynaptic firing bias
  αpost::R# postsynaptic firing bias
  Aplus::R# A/τ  normalized "potentiation"
  Aminus::R# θ*A/(γ*τ) normalized "depression"
  post_trace::Trace{ForPlasticity,R}
  pre_trace::Trace{ForPlasticity,R}
  bounds::PlasticityBounds{R}
  function PlasticityAsymmetricX(A::R,θ::R,τ::R,γ::R,
      αpre::R,αpost::R,n_post::Integer,n_pre::Integer;
      plasticity_bounds=PlasticityBoundsNonnegative()) where R<:Real
    trace_post = Trace(τ*γ,n_post) # this is the A minus side
    trace_pre = Trace(τ,n_pre) # this is the A plus side
    Aplus = A/τ
    Aminus = θ*A/(γ*τ)
    @assert γ>0 "Something wrong with parameter γ, should be >0"
    new{R}(A,θ,τ,γ,αpre,αpost,
      Aplus,Aminus,trace_post,trace_pre,plasticity_bounds)
  end
end
function reset!(pl::PlasticityAsymmetricX)
  reset!(pl.pre_trace)
  reset!(pl.post_trace)
  return nothing
end

function area_under_curve(plast::PlasticityAsymmetricX)
  return plast.A*(1+plast.θ)
end


# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PlasticityAsymmetricX{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    Aabs = abs(plast.A)
    @inbounds Δw = ( plast.Aminus*plast.post_trace.val[i_post]
               + Aabs*plast.αpre) # add presynaptic firing bias
    @inbounds weight_matrix[i_post,k_pre] =  plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PlasticityAsymmetricX{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre] 
  if !iszero(wkj)
    Aabs = abs(plast.A)
    @inbounds Δw = (plast.Aplus*plast.pre_trace.val[j_pre]
               + Aabs*plast.αpost) # add postsynaptic firing bias
    @inbounds weight_matrix[k_post,j_pre] =  plast.bounds(wkj,Δw)
  end
  return nothing
end



function plasticity_update!(t_spike::R,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityAsymmetricX{R}) where R<:Real
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_trace)
  propagate!(t_spike,plast.post_trace)
  # increase the plasticity trace variables (if not zero)
  update_now!(plast.post_trace,k_post_spike)
  update_now!(plast.pre_trace,k_pre_spike)
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      weight_update_postfired!(weights,k_post_spike,j,plast)
    end
  end
  return nothing
end


# Generalized STDP, symmetric
struct PlasticityAsymmetricSTDPGL{R} <: PlasticityRule
  Azero::R      # global scaling
  Aplus::R
  Aminus::R
  τ::R      # time constant 
  γ::R      # time constant minus scaling
  αzero::R  # constant leack term
  αpre::R   # presynaptic firing bias
  αpost::R  # postsynaptic firing bias
  _Cplus::R  #  normalized potentiation
  _Cminus::R #  normalized depression
  post_trace::Trace{ForPlasticity,R}
  pre_trace::Trace{ForPlasticity,R}
  t_last_posts::Vector{R}
  bounds::PlasticityBounds{R}
  function PlasticityAsymmetricSTDPGL(Azero::R,Aplus::R,Aminus::R,τ::R,γ::R,
      αzero::R,αpre::R,αpost::R,
      n_post::Integer,n_pre::Integer;
      t_start = 0.0,
      plasticity_bounds=PlasticityBoundsNonnegative{R}()) where R
    @assert Azero >= 0 "Azero should be >= 0"
    @assert Aplus*Aminus <= 0 "Aplus and Aminus should have opposite signs"
    post_t = Trace(τ*γ,n_post) # this is the A minus side
    pre_t = Trace(τ,n_pre) # this is the A plus side
    t_last_posts = fill(t_start,n_post)
    _Cplus = Aplus/τ
    _Cminus = Aminus/(γ*τ)
    @assert γ>0 "Something wrong with parameter γ, should be >0"
    new{Float64}(Azero,Aplus,Aminus,τ,γ,
      αzero, αpre,αpost,
      _Cplus,_Cminus,
      post_t,pre_t,t_last_posts,
      plasticity_bounds)
  end
end
function reset!(pl::PlasticityAsymmetricSTDPGL;t_start::Real=0.0)
  reset!(pl.pre_trace)
  reset!(pl.post_trace)
  pl.t_last_posts .= t_start
  return nothing
end

function area_under_curve(plast::PlasticityAsymmetricSTDPGL)
  return plast.Azero*(plast.Aplus+plast.Aminus)
end


# needs to stay outside for optimization purposes
@inline function weight_update_prefired!(weight_matrix::Matrix{Float64},
    k_pre::Integer,i_post::Integer,plast::PlasticityAsymmetricSTDPGL{Float64})::Nothing
  wik = weight_matrix[i_post,k_pre] 
  if !iszero(wik)
    @inbounds Δw = plast.Azero*( plast.αpre +
            plast.post_trace.val[i_post]*plast._Cminus)
    @inbounds weight_matrix[i_post,k_pre] =  plast.bounds(wik,Δw)
  end
  return nothing
end

# needs to stay outside for optimization purposes
@inline function weight_update_postfired!(Δt_last_post::Float64,
    weight_matrix::Matrix{Float64},
    k_post::Integer,j_pre::Integer,plast::PlasticityAsymmetricSTDPGL{Float64})::Nothing
  wkj = weight_matrix[k_post,j_pre] 
  if !iszero(wkj)
    @inbounds Δw = plast.Azero*(
            plast.αzero*Δt_last_post +
            plast.αpost + 
            plast.pre_trace.val[j_pre]*plast._Cplus)
    @inbounds weight_matrix[k_post,j_pre] =  plast.bounds(wkj,Δw)
  end
  return nothing
end


function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityAsymmetricSTDPGL{R}) where R<:Real
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_trace)
  propagate!(t_spike,plast.post_trace)
  # increase the plasticity trace variables
  if !iszero(k_post_spike)
    update_now!(plast.post_plus_trace,k_post_spike)
    update_now!(plast.post_minus_trace,k_post_spike)
    # propagate the decay trace
    propagate!(t_spike,plast.weight_decay_post_trace)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.pre_plus_trace,k_pre_spike)
    update_now!(plast.pre_minus_trace,k_pre_spike)
  end
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      weight_update_prefired!(weights,k_pre_spike,i,plast)
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    Δt_last_post = t_spike-plast.t_last_posts[k_post_spike]
    for j in 1:npre
      weight_update_postfired!(Δt_last_post,weights,k_post_spike,j,plast)
    end
    # update to current spike
    t_last_posts[k_post_spike] = t_spike
  end
  return nothing
end





#=
struct PlasticityAsymmetricAdaptiveX <: PlasticityRule
  A::Float64  # overall scaling
  θ::Float64  # bias term  (total bias is 1+θ)
  τ::Float64  # time constant 
  γ::Float64  # scaling of post-pre time constant
  τadapt::Float64 # adaptation time constant (should be high) 
  αpre::Float64 # postsynaptic firing bias
  αadapt::Float64 # mismatch between target rate and presynaptic rate
  Aplus::Float64 # A/τ  normalized "potentiation"
  Aminus::Float64 # θ*A/(γ*τ) normalized "depression"
  pre_plus_trace::Trace
  pre_minus_trace::Trace
  post_plus_trace::Trace
  post_minus_trace::Trace
  adaptive_trace::Trace
  bounds::PlasticityBounds
  function PlasticityAsymmetricAdaptiveX(A,θ,τ,γ,τadapt,
      αpre,αadapt,n_post,n_pre;
      plasticity_bounds=PlasticityBoundsNonnegative())
    pre_plus_tr = Trace(τ,n_pre)
    pre_minus_tr = Trace(τ*γ,n_pre)
    post_plus_tr = Trace(τ,n_post)
    post_minus_tr = Trace(τ*γ,n_post)
    adapt_pre_tr = Trace(τadapt,n_pre)
    Aplus = A/τ
    Aminus = θ*A/(γ*τ)
    @assert γ>0 "Something wrong with parameter γ, should be >0"
    new(A,θ,τ,γ,γadapt,αpre,αadapt, 
        Aplus,Aminus,
        pre_plus_tr,pre_minus_tr,post_plus_tr,post_minus_tr,adapt_pre_tr,plasticity_bounds)
  end
end



function reset!(pl::PlasticityAsymmetricAdaptiveX)
  reset!(pl.pre_plus_trace)
  reset!(pl.pre_minus_trace)
  reset!(pl.post_plus_trace)
  reset!(pl.post_minus_trace)
  reset!(pl.adaptive_trace)
  return nothing
end

function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityAsymmetricAdaptiveX)
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_plus_trace)
  propagate!(t_spike,plast.pre_minus_trace)
  propagate!(t_spike,plast.post_plus_trace)
  propagate!(t_spike,plast.post_minus_trace)
  propagate!(t_spike,plast.adaptive_trace)
  # increase the plasticity trace variables
  if !iszero(k_post_spike)
    update_now!(plast.post_plus_trace,k_post_spike)
    update_now!(plast.post_minus_trace,k_post_spike)
  end
  if !iszero(k_pre_spike)
    update_now!(plast.pre_plus_trace,k_post_spike)
    update_now!(plast.pre_minus_trace,k_post_spike)
    # adaptation 
  end
  # update synapses
  weights=conn.weights
  npost,npre = size(weights)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      wik = weights[i,k_pre_spike] 
      if wik > 0
        Δw = ( plast.post_minus_trace.val[i]*plast.Aminus
               + plast.αpre) # add presynaptic firing bias
        weights[i,k_pre_spike] =  plast.bounds(wik,Δw)
      end
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      wkj = weights[k_post_spike,j] 
      if wkj > 0
        Δw = ( plast.pre_plus_trace.val[j]*plast.Aplus
               + plast.αpost ) # add postsynaptic firing bias
        weights[k_post_spike,j] =  plast.bounds(wkj,Δw)
      end
    end
  end
  return nothing
end

=#