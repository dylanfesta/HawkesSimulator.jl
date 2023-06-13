

abstract type Recorder end

struct RecNothing <: Recorder end
function reset!(::RecNothing)
  return nothing
end

# general signature: record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)
# example :
# @inline function record_stuff!(::RecNothing,::Real,
#     ::Integer,::Integer,::Symbol,
#     ::RecurrentNetworkExpKernel)
#   return nothing  
# end
@inline function record_stuff!(::RecNothing,::Real,whatevs...)
  return nothing  
end


# Record weights
# to simplify things, I attach the recorder the target matrix directly 
# i.e. the pointer to the matrix I want to record.
# (so it won't record all weights in all populations... too much effort)

struct RecTheseWeights <: Recorder
  Δt::Float64
  Tend::Float64
  Tstart::Float64
  weights_target::Matrix{Float64}
  weights::Vector{Matrix{Float64}}
  times::Vector{Float64}
  t_last::Ref{Float64}
  k_ref::Ref{Int64}
  function RecTheseWeights(weights::Matrix{Float64},Δt::R,Tend::R; Tstart::Real=0.0) where R<:Real
    nalloc = ceil(Integer,(Tend-Tstart)/Δt)+2
    @assert nalloc > 1
    weights_alloc = Vector{Matrix{Float64}}(undef,nalloc)
    times_alloc = Vector{Float64}(undef,nalloc)
    for i in eachindex(weights_alloc)
      weights_alloc[i] = similar(weights)
    end
    new(Δt,Tend,Tstart,weights,weights_alloc,times_alloc,Ref(-Inf),Ref(0))
  end
end
function reset!(rec::RecTheseWeights)
  rec.t_last[] = -Inf
  rec.k_ref[] = 0
  fill!(rec.times,NaN)
  for w in rec.weights
    fill!(w,NaN)
  end
  return nothing
end

function get_timesweights(rec::RecTheseWeights)
  k = rec.k_ref[]
  return rec.times[1:k],rec.weights[1:k]
end

function record_stuff!(rec::RecTheseWeights,tfire::Real,whatevs...)
  #check time
  if ((tfire - rec.t_last[]) < rec.Δt) || (tfire < rec.Tstart) || (tfire > rec.Tend) 
    return nothing
  end
  # we must record !
  # update
  k = rec.k_ref[] + 1
  rec.t_last[] = tfire
  rec.k_ref[] = k
  rec.times[k] = tfire
  copy!(rec.weights[k],rec.weights_target)
  return nothing
end


struct RecTheseWeightsContent
  times::Vector{Float64}
  weights::Vector{Matrix{Float64}}
  function RecTheseWeightsContent(r::RecTheseWeights)
    idxs_keep = 1:r.k_ref[]
    new(r.times[idxs_keep],r.weights[idxs_keep])
  end
end
function get_content(rec::RecTheseWeights)
  return RecTheseWeightsContent(rec)
end


# Generic object to do stuff every Δt

mutable struct DoEveryDt{R} <: Recorder
  Δt::R
  Tstart::R
  Tend::R
  t_last::R
  thing_to_do::Function
end

function DoEveryDt(thing_to_do::Function,Δt::R;
    Tstart::Real=0.0,Tend::Real=Inf) where R<:Real
  DoEveryDt(Δt,Tstart,Tend,-Inf,thing_to_do)
end

function reset!(rec::DoEveryDt)
  rec.t_last = -Inf
  return nothing
end

# general signature: record_stuff!(rec,tfire,popfire,neufire,label_fire,ntw)
function record_stuff!(rec::DoEveryDt{R},tfire::R,whatevs...) where R
  #check time
  if ((tfire - rec.t_last) < rec.Δt) || (tfire < rec.Tstart) || (tfire > rec.Tend) 
    return nothing
  end
  # we must record !
  # update
  rec.t_last = tfire
  rec.thing_to_do(tfire,whatevs...)
  return nothing
end