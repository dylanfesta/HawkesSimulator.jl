
# How to constrain the weights ? Upper limit only, or sum strictly equal to
# target ?
abstract type HeterosynapticConstraint end
# Whether to consider the sum over  outgoing weights, incoming weights, or both
abstract type HeterosynapticTarget end
# Additive or multiplicative  
abstract type HeterosynapticMethod end


struct HetUpperLimit <: HeterosynapticConstraint
  wsum_max::Float64
  wmin::Float64
  wmax::Float64
  tolerance::Float64
end
struct HetStrictSum <: HeterosynapticConstraint
  wsum_max::Float64
  wmin::Float64
  wmax::Float64
  tolerance::Float64
end

@inline function hardbounds(x::Float64,hc::HeterosynapticConstraint)
  hardbounds(x,hc.wmin,hc.wmax)
end

function _hetconstraint_vs_sum(sums::Array{Float64},c::HetUpperLimit)
  _lim  =  c.wsum_max + c.tolerance
  return sums .> _lim
end
function _hetconstraint_vs_sum(sums::Array{Float64},c::HetStrictSum)
  return .!isapprox.(sums,c.wsum_max;atol=c.tolerance)
end

struct HetIncoming <: HeterosynapticTarget end # rows
struct HetOutgoing <: HeterosynapticTarget end # cols
struct HetBoth <: HeterosynapticTarget end     # both

struct HetAdditive <: HeterosynapticMethod end
struct HetMultiplicative <: HeterosynapticMethod end


struct PlasticityHeterosynapticApprox{ 
    HC<:HeterosynapticConstraint,
    HM<:HeterosynapticMethod,
    HT<:HeterosynapticTarget} <:PlasticityRule 
  constraint::HC
  method::HM
  target::HT
  Δt_update::Float64
  _tcounter::Ref{Float64}
  alloc_incoming::Vector{Float64} # preallocate, for sums
  alloc_Nincoming::Vector{Int64}  # and count synapses too
  alloc_outgoing::Matrix{Float64} # row vector
  alloc_Noutgoing::Vector{Int64}
  function PlasticityHeterosynapticApprox(
      Npost::Int64,Npre::Int64,
      Δt_update::Float64,
      hc::HC,hm::HM,ht::HT) where {
        HC <: HeterosynapticConstraint,
        HM <: HeterosynapticMethod,
        HT <: HeterosynapticTarget }
    _tcounter = Ref(0.0)  
    alloc_incoming = fill(0.0,Npost)
    alloc_Nincoming = fill(0,Npost)
    alloc_outgoing = fill(0.0,1,Npre) # row!
    alloc_Noutgoing = fill(0,Npre)
    return new{HC,HM,HT}(hc,hm,ht,
      Δt_update,_tcounter,
      alloc_incoming,alloc_Nincoming,alloc_outgoing,alloc_Noutgoing)
  end
end

function reset!(plast::PlasticityHeterosynapticApprox)
  plast._tcounter[] = zero(Float64)
  # just in case...
  fill!(plast.alloc_incoming,0.0)
  fill!(plast.alloc_Nincoming,0)
  fill!(plast.alloc_outgoing,0.0)
  fill!(plast.alloc_Noutgoing,0)
  return nothing
end

function plasticity_update!(t_spike::Real,::Integer,::Integer,
    ::AbstractPopulationState,conn::Connection,::AbstractPopulationState,
    plast::PlasticityHeterosynapticApprox)
  # time since previous spike  
  if (t_spike-plast._tcounter[]) < plast.Δt_update  
    return nothing
  end
  # reset timer
  plast._tcounter[] = t_spike
  # find correction values at col/row , depending on target
  _het_plasticity_fix_incoming!(plast.alloc_incoming,plast.alloc_Nincoming,
    conn.weights,plast.constraint,plast.method,plast.target)
  _het_plasticity_fix_outgoing!(plast.alloc_outgoing,plast.alloc_Noutgoing,
    conn.weights,plast.constraint,plast.method,plast.target)
  # apply the fix  
  _het_plasticity_apply_fix!( 
      plast.alloc_incoming,plast.alloc_outgoing,
      conn.weights,
      plast.constraint,plast.method,plast.target)
  return nothing
end

# applies plasticity several time, to initialize the weight matrix connerctly
function plasticity_init_weights!(weights::Matrix,plast::PlasticityHeterosynapticApprox ; repeats::Integer=5)
  for _ in 1:repeats
    _het_plasticity_fix_incoming!(plast.alloc_incoming,plast.alloc_Nincoming,
      weights,plast.constraint,plast.method,plast.target)
    _het_plasticity_fix_outgoing!(plast.alloc_outgoing,plast.alloc_Noutgoing,
      weights,plast.constraint,plast.method,plast.target)
    # apply the fix  
    _het_plasticity_apply_fix!( 
        plast.alloc_incoming,plast.alloc_outgoing,
        weights,
        plast.constraint,plast.method,plast.target)
  end
  return nothing
end


function _het_plasticity_fix_incoming!(alloc_sum::Vector{Float64},Nel::Vector{Int64},
    weights::Matrix{Float64},
    constraint::Union{HetStrictSum,HetUpperLimit},
    ::HetAdditive,::Union{HetBoth,HetIncoming})
  sum!(alloc_sum,weights)
  tofix = _hetconstraint_vs_sum(alloc_sum,constraint)
  for (k,row) in enumerate(eachrow(weights))
    Nel[k]=count(!=(0.0),row)
  end
  sum_max = constraint.wsum_max
  @. alloc_sum = (sum_max - alloc_sum)/Nel
  alloc_sum[.! tofix] .= 0.0
  return nothing
end
function _het_plasticity_fix_incoming!(::Vector{Float64},::Vector{Int64},
    ::Matrix,::HeterosynapticConstraint,
    ::HeterosynapticMethod,::HetOutgoing)
  return nothing
end
function _het_plasticity_fix_outgoing!(alloc_sum::Matrix{Float64},Nel::Vector{Int64},
    weights::Matrix,
    constraint::Union{HetStrictSum,HetUpperLimit},
    ::HetAdditive,::Union{HetBoth,HetOutgoing})
  sum!(alloc_sum,weights)
  tofix = _hetconstraint_vs_sum(alloc_sum,constraint)
  sum_max = constraint.wsum_max
  for (j,dofix) in enumerate(tofix)
    if dofix
      _col = view(weights,:,j)
      _Nel=count(!=(0.0),_col)
      alloc_sum[j] = (sum_max - alloc_sum[j])/ _Nel
    else
      alloc_sum[j] = 0.0
    end
  end
  return nothing
end
function _het_plasticity_fix_outgoing!(::Matrix{Float64},::Vector{Int64},
    ::Matrix,::HeterosynapticConstraint,::HeterosynapticMethod,::HetIncoming)
  return nothing  
end

# returns mean if both nonzero
@inline function _mean_nonzero(a::R,b::R) where R
  if iszero(a)
    return b
  elseif iszero(b)
    return a
  else
    return 0.5*(a+b)
  end
end

function _het_plasticity_apply_fix!( 
    fixrows::Vector{Float64},fixcols::Matrix{Float64},
    weights::Matrix{Float64},constraint::HeterosynapticConstraint,
    ::HetAdditive,::HetBoth)
  @inbounds for ij in CartesianIndices(weights)
    wij = weights[ij]
    if !iszero(wij) # skip missing connections
      wij += _mean_nonzero(fixrows[ij[1]],fixcols[ij[2]])
      if !iszero(wij)
        weights[ij] = hardbounds(wij,constraint)
      end
    end
  end
  return nothing
end


# could be optimized more :-/
function _het_plasticity_apply_fix!( 
    fixrows::Vector{Float64},::Matrix{Float64},
    weights::Matrix{Float64},constraint::HeterosynapticConstraint,
    ::HetAdditive,::HetIncoming)
  @inbounds for ij in CartesianIndices(weights)
    wij = weights[ij]
    if !iszero(wij) # skip missing connections
      wij += fixrows[ij[1]]
      if !iszero(wij)
        weights[ij] = hardbounds(wij,constraint)
      end
    end
  end
  return nothing
end

# could be optimized more :-/
function _het_plasticity_apply_fix!( 
    ::Vector{Float64},fixcols::Matrix{Float64},
    weights::Matrix{Float64},constraint::HeterosynapticConstraint,
    ::HetAdditive,::HetOutgoing)
  @inbounds for ij in CartesianIndices(weights)
    wij = weights[ij]
    if !iszero(wij) # skip missing connections
      wij += fixcols[ij[2]]
      if !iszero(wij)
        weights[ij] = hardbounds(wij,constraint)
      end
    end
  end
  return nothing
end