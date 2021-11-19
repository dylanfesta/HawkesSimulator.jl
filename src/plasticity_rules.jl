
# traces as structs
# (after all, why not?  ... why shouldn't I make it a struct?)

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
  val .*= exp(-Δt/tra.τ)
  tra.t_last[]=tnow
  return nothing
end
# assumes you already popagated until now
function update_now!(tra::Trace,idx_update::Vector{I},up_val::Float64=1.0) where I
  @inbounds @simd for i in idx_update
    tra.val[i] += up_val
  end
  return nothing
end
function reset!(tra::Trace)
  fill!(tra.val,0.0) 
  tra.t_last[]=tnow
  return nothing
end

# bounds are structures too!
# because!

abstract type PlasticityBounds end
struct PlasticityBoundsNonNegative <: PlasticityBounds end

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

#=
function plasticity_update!(t_now::Real,
  pspost::PopulationState,conn::Connection,pspre::PopulationState,
  plast::PairSTDP)

  # update synapses
# presynpatic spike go along w column
for j_pre in idx_pre_spike
   Δw = -plast.post_trace[ipost]*plast.Aminus
   weightsnz[_pnz] = max(0.0,weightsnz[_pnz]-Δw)
 end
end
# postsynaptic spike: go along w row
# innefficient ... need to search i element for each column
for i_post in idx_post_spike
 for j_pre in (1:pspre.n)
   _start = _colptr[j_pre]
   _end = _colptr[j_pre+1]-1
   _pnz = searchsortedfirst(row_idxs,i_post,_start,_end,Base.Order.Forward)
   if (_pnz<=_end) && (row_idxs[_pnz] == i_post) # must check!
     Δw = plast.pre_trace[j_pre]*plast.Aplus 
     weightsnz[_pnz]+= Δw
   end
 end
end
# update the plasticity trace variables
update_traces!(plast.pre_traces,idx_pre_spike)
update_traces!(plast.post_traces,idx_post_spike)
return nothing
end
=#