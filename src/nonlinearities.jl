

@inline function apply_nonlinearity(x::R,::NLRelu) where {R<:Real}
  return max(x,zero(R))
end
struct NLRectifiedQuadratic <: AbstractNonlinearity end
@inline function apply_nonlinearity(x::R,::NLRectifiedQuadratic) where {R<:Real}
  return ifelse(x <= zero(R),zero(R),x*x)
end

# @inline function apply_nonlinearity(x::R,nl::NLRmax{R}) where {R<:Real}
#   return min(max(x,zero(R)),nl.rmax)
# end


# @inline function apply_nonlinearity(x::R,nl::NLRmax{R}) where {R<:Real}
#   if zero(R) < x < nl.rmax
#     return x
#   elseif x < zero(R)
#     return zero(R)
#   else
#     return nl.rmax
#   end
#  end

function apply_nonlinearity(x::R,nl::NLRmax{R}) where {R<:Real}
   if zero(R) < x < nl.rmax
     return x
   elseif x < zero(R)
     return zero(R)
   else
     return nl.rmax
   end
end


