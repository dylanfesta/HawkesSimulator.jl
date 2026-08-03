# Diagnostic only: this is not the production upper-rate implementation.
#
# Bonnet, Martinez Herrera, and Sangnier use only positive interactions
# for the piecewise-constant upper bound in Section 4.1:
# https://arxiv.org/pdf/2205.04107

baseline = 1.0
excitation_at_zero = 10.0
inhibition_at_zero = 9.0
τ_excitation = 1.0
τ_inhibition = 0.1

rate(t) = max(0.0,baseline +
  excitation_at_zero*exp(-t/τ_excitation) -
  inhibition_at_zero*exp(-t/τ_inhibition))

regressed_behavior_bound = max(baseline,rate(0.0))
positive_only_bound = baseline+excitation_at_zero
future_time = 0.2
future_rate = rate(future_time)

@assert future_rate > regressed_behavior_bound
@assert all(rate(t) <= positive_only_bound for t in range(0.0,10.0;length=10_001))

println("Regressed bound: ",regressed_behavior_bound)
println("Rate at t=",future_time,": ",future_rate)
println("Positive-interaction-only bound: ",positive_only_bound)
