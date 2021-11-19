

@inline function apply_nonlinearity(x::Real,::NLRelu)
  return max(x,0.0)
end
struct NLRectifiedQuadratic <: AbstractNonlinearity end
@inline function apply_nonlinearity(x::R,::NLRectifiedQuadratic) where R<:Real
  if x <= 0.0
    return 0.0
  else 
    return x*x
  end
end