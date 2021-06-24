using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using SparseArrays 
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using  HawkesSimulator; const global H = HawkesSimulator

function onedmat(x::Real)
  ret=Matrix{Float64}(undef,1,1)
  ret[1,1] = x 
  return ret
end 

##