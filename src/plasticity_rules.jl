

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

# traces as structs
# (after all, why not?  ... why shouldn't I use a struct?)

struct Trace
  val::Vector{Float64}
  τ::Float64
  t_last::Ref{Float64}
  function Trace(τ::R,n::Integer) where R
    val = fill(zero(R),n)
    t_last = Ref(0.0)
    return new(val,τ,t_last)
  end
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

function reset!(tra::Trace)
  fill!(tra.val,0.0) 
  tra.t_last[]=0.0
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

# utility function
function find_in_trains(t_now::R,trains_pre::Vector{Vector{R}},trains_post::Vector{Vector{R}}) where R
  is_t_now(train) = isempty(train) ? false : last(train) == t_now
  k_pre::Int64  = something(findfirst(is_t_now,trains_pre),0)
  k_post::Int64 = something(findfirst(is_t_now,trains_post),0)
  return (k_pre,k_post)
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

function plasticity_update!(t_now::Real,
    pspre::PopulationState,conn::Connection,pspost::PopulationState,
    plast::PairSTDP)
  weights = conn.weights
  npost,npre = size(weights)
  k_pre,k_post = find_in_trains(t_now,pspre.trains,pspost.trains)
  if iszero(k_pre) && iszero(k_post)
    return nothing
  end
  # update all pre and post traces to t_now
  propagate!(t_now,plast.pre_trace)
  propagate!(t_now,plast.post_trace)
  # update the plasticity trace variables, if needed
  update_now!(plast.pre_trace,k_pre)
  update_now!(plast.post_trace,k_post)
  if !iszero(k_pre)
    # k is presynaptic: go along rows of k_pre column 
    for i in 1:npost
      wik = weights[i,k_pre] 
      if wik > 0
        Δw = plast.post_trace.val[i]*plast.Aminus
        weights[i,k_pre] =  plast.bounds(wik,Δw)
      end
    end
  end
  if !iszero(k_post)
    # k is postsynaptic: go along columns of k_post row
    for j in 1:npre
      wkj = weights[k_post,j] 
      if wkj > 0
        Δw = plast.pre_trace.val[j]*plast.Aplus
        weights[k_post,j] =  plast.bounds(wkj,Δw)
      end
    end
  end
  return nothing
end