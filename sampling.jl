using DifferentialEquations
using DataFrames
using Distributions
using Dates
using CSV

function NTot(infPath::RODESolution)

  nT = length(infPath)
  for i in 1:(nT-1)
    if infPath.u[i] == infPath.u[i+1]
      return infPath.t[i]
    end
  end

  return infPath.t[nT]
end

function SampleNTotAdd2Paths(
  nPath::Int, iniVec::Vector{<:Real}, NSimEnd::Real, NBkBnd::Vector{<:Real},
  driftFunc::Function, volFunc::Function, dN::Real, outFilePath::String;
  discScheme=EM())
  # NBkBnd is a pair of Nl and Nu.
  # The nPath values of Nbk are sampled from the uniform distribution on (Nl, Nu), and then 2 values of NTot are generated for each Nbk.

  NTotDfColName = vcat(["NBk", "NTot1", "NTot2"], ["phi_vel_vec_elem" * i for i in string.(1:length(iniVec))])
  prob = SDEProblem(driftFunc, volFunc, iniVec, (0, NSimEnd))
  NTotDf = DataFrame([Symbol(cn) => Float64[] for cn in NTotDfColName])

  for iPath in 1:nPath

    if iPath % 1000 == 1
      println("iPath=" * string(iPath) * " :" * string(now())); flush(stdout)
    end

    infPath = solve(prob, discScheme, dt = dN, adaptive=false)
    NAll = infPath.t
    NBkws = rand(Uniform(NBkBnd[1], NBkBnd[2]))

    for NBkw in NBkws
      iNReg = argmin(abs.(NAll .- NTot(infPath) .+  NBkw))
      probAdd = SDEProblem(driftFunc, volFunc, infPath.u[iNReg], (NAll[iNReg], NSimEnd))
      ensembleprobAdd = EnsembleProblem(probAdd)
      newPaths = solve(ensembleprobAdd, discScheme, EnsembleThreads(), trajectories = 2, dt = dN, adaptive=false)
      NTot1 = NTot(newPaths[1])
      NTot2 = NTot(newPaths[2])
      push!(NTotDf, vcat([NBkw, NTot1, NTot2], infPath.u[iNReg]))
    end

    if iPath == 1000
      CSV.write(outFilePath, NTotDf)
    elseif iPath % 1000 == 0
      CSV.write(outFilePath, NTotDf[(iPath - 999):iPath,:], append=true)
    end

  end

  return NTotDf

end
