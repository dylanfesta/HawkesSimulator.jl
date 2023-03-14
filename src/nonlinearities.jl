

@inline function apply_nonlinearity(x::Float64,::NLRelu)::Float64
  return max(x,0.0)::Float64
end
struct NLRectifiedQuadratic <: AbstractNonlinearity end
@inline function apply_nonlinearity(x::Float64,::NLRectifiedQuadratic)::Float64
  if x <= 0.0
    return 0.0
  else 
    return x*x
  end
end


@inline function apply_nonlinearity(x::Float64,nl::NLRmax)::Float64
  if x < 0.0
    return 0.0
  elseif x > nl.rmax
    return nl.rmax
  else
    return x
  end
end