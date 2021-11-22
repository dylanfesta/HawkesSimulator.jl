using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) #; plotlyjs()
using FileIO
using SparseArrays 
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using  HawkesSimulator; const global H = HawkesSimulator

##
function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end

##