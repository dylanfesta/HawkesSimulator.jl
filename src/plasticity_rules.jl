

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

abstract type PlasticityBounds end
struct PlasticityBoundsNonnegative <: PlasticityBounds end
struct PlasticityBoundsLowHigh <: PlasticityBounds 
  low::Float64
  high::Float64
end
function (plb::PlasticityBoundsNonnegative)(w::R,Δw::R) where R
  ret = w+Δw
  return max(eps(100.0),ret)
end
function (plb::PlasticityBoundsLowHigh)(w::R,Δw::R) where R
  ret = w+Δw
  return min(plb.high,max(plb.low,ret))
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

# Pairwise plasticity 
struct PairSTDP <: PlasticityRule
  Aplus::Float64
  Aminus::Float64
  pre_trace::Trace
  post_trace::Trace
  bounds::PlasticityBounds
  function PairSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
       plasticity_bounds=PlasticityBoundsNonnegative())
    pre_trace = Trace(τplus,n_pre)
    post_trace = Trace(τminus,n_post)
    new(Aplus,Aminus,pre_trace,post_trace,plasticity_bounds)
  end
end
function reset!(pl::PairSTDP)
  reset!(pl.pre_trace)
  reset!(pl.post_trace)
  return nothing
end

function plasticity_update!(t_spike::Real,
    pspost::PopulationState,conn::Connection,pspre::PopulationState,
    plast::PairSTDP)
  k_post,k_pre = find_which_spiked(t_spike,pspost,pspre)
  return plasticity_update!(t_spike,k_post,k_pre,pspre,conn,pspost,plast)
end

# signature: plasticity_update!(tfire,label_fire,neufire,post,conn,pre,plast)

function plasticity_update!(tfire::Real,label_spike::Symbol,
    neufire::Integer,
    pspre::AbstractPopulationState,conn::Connection,
    pspost::AbstractPopulationState,
    plast::PairSTDP)
  k_post,k_pre = find_which_spiked(label_spike,neufire,pspost,pspre)
  return plasticity_update!(tfire,k_post,k_pre,pspre,conn,pspost,plast)
end

function plasticity_update!(t_spike::Real,k_post_spike::Integer,k_pre_spike::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,plast::PairSTDP)
  weights = conn.weights
  npost,npre = size(weights)
  if iszero(k_pre_spike) && iszero(k_post_spike)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_spike,plast.pre_trace)
  propagate!(t_spike,plast.post_trace)
  # update the plasticity trace variables, if needed
  update_now!(plast.pre_trace,k_pre_spike)
  update_now!(plast.post_trace,k_post_spike)
  if !iszero(k_pre_spike)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      wik = weights[i,k_pre_spike] 
      if wik > 0
        Δw = plast.post_trace.val[i]*plast.Aminus
        weights[i,k_pre_spike] =  plast.bounds(wik,Δw)
      end
    end
  end
  if !iszero(k_post_spike)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      wkj = weights[k_post_spike,j] 
      if wkj > 0
        Δw = plast.pre_trace.val[j]*plast.Aplus
        weights[k_post_spike,j] =  plast.bounds(wkj,Δw)
      end
    end
  end
  return nothing
end