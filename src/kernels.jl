
# Interaction kernels defined here (they are unit types!)


@inline function interaction_kernel_fourier(ω::Real,popstate::PopulationState)
  return interaction_kernel_fourier(ω,popstate.unittype)
end

# a cluncky sharp step !
struct KernelStep <: UnitType
  τ::Float64
end
@inline function interaction_kernel(t::R,ker::KernelStep) where R<:Real
  return (t < zero(R)) || (t > ker.τ) ? zero(R) :  inv(ker.τ)
end
interaction_kernel_upper(t::Real,pop::KernelStep) = interaction_kernel(t,pop) + eps(100.0)

@inline function interaction_kernel_fourier(ω::R,pop::KernelStep) where R<:Real
  if iszero(ω)
    return one(R)
  end
  cc = 2π*pop.τ*ω
  return (sin(cc)-2*im*sin(0.5cc)^2)/cc
end

# Negative exponential
struct KernelExp <: UnitType
  τ::Float64
end
@inline function interaction_kernel(t::R,ker::KernelExp) where R<:Real
  return t < zero(R) ? zero(R) : exp(-t/ker.τ) / ker.τ
end
interaction_kernel_upper(t::Real,ker::KernelExp) = interaction_kernel(t,ker) + eps(100.0)
interaction_kernel_fourier(ω::Real,ker::KernelExp) =  inv( 1 + im*2*π*ω*ker.τ)

# Alpha-shape with delay

struct KernelAlphaDelay <: UnitType
  τ::Float64
  τdelay::Float64
end

@inline function interaction_kernel(t::Real,ker::KernelAlphaDelay)
  Δt = t - ker.τdelay
  if Δt <= 0.0
    return 0.0
  else
    return Δt/ker.τ^2 * exp(-Δt/ker.τ)
  end
end
@inline function interaction_kernel_upper(t::Real,ker::KernelAlphaDelay)
  Δt = t - ker.τdelay
  if Δt <= ker.τ
    return  inv(ker.τ*ℯ) + eps(100.)
  else
    return  interaction_kernel(t,ker) + eps(100.0)
  end
end
@inline function interaction_kernel_fourier(ω::Real,ker::KernelAlphaDelay)
  return exp(-im*2*π*ω*ker.τdelay) / (1+im*2*π*ω*ker.τ)^2
end
