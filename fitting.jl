using DataFrames
using ClassicalOrthogonalPolynomials
using LsqFit
using Statistics
using LinearAlgebra

function legendreDeriv(n, x)
  if n == 0
      return 0.0
  else
      return n / (x * x - 1.0) * (x * legendrep(n, x) - legendrep(n - 1, x))
  end
end

function legendreDeriv2(n, x)
  if n == 0 || n == 1
      return 0.0
  else
      return n / (x * x - 1.0) * (legendrep(n, x) + (n - 2) / n * x * legendreDeriv(n, x) - legendreDeriv(n - 1, x))
  end
end

function FitByLS(NTotData::DataFrame, funcType::String, basisFuncDeg::Int, NBkMin::Real, NBkMax::Real;
  basisFuncType::String="Legendre", paramIni=nothing, paramDerivDelta::Real=0.01)
  ### Input
  # NTotData: Stores sampled e-fold values.
  #           It must have the following 3 columns.
  #           NBk: On each sample path from the initial point to the end-of-inflation (EOI),
  #                2 branch paths are generated from the point at which the backward e-fold is NBk.
  #           NTot1, NTot2: The e-fold elapsed from the branch point to the EOI on the first and second branch path.
  # funcType: Specifies the overall shape of the fitting function
  #           Now, the following types can be used: with tunable real numbers a_i and basis functions p_i,
  #           "exp": f(x) = exp(a_1*p_1(x)+a_2*p_2(x)+...)
  #           "logistic": f(x) = a_1 / (1 + exp(a_2*p_1(x)+a_3*p_2(x)+...))
  #           "exp+const": f(x) = a_1+exp(a_2*p_1(x)+a_3*p_2(x)+...)
  #           "exp+poly": f(x) = exp(a_1*p_1(x)+a_2*p_2(x)+...)+a_{n+1}*p_1(x)+a_{n+2}*p_2(x)+...
  #           "logistic+linear": logistic for x>a[end] and linear for x<a[end] with continuous 0th and 1st derivatives 
  # basisFuncDeg: maximum degree of the basis function
  # NBkMin, NBkMax: In the fitting function, its variable NBk is normalized with NBkMin and NBkMax (see the code below)
  # basisFuncType: Specifies the type of the basis functions
  #                Now, the following types can be used:
  #                "Legendre": Legendre polynomial
  # paramIni: initial parameters
  # paramDerivDelta: delta used when parameter derivatives are evaluated by finite difference
  #
  ### Output: Tuple of the follwoing
  # fitResult (LsqFit.LsqFitResult): fitting result
  # delNSqFit (Function): function of NBk vs <\delta N^2> as the fitting result 
  # psFit (Function): function of NBk vs power spectrum as the fitting result
  # delNSqFit (Function): function of NBk vs <\delta N^2> as the fitting result 
  # psFit (Function): function of NBk vs power spectrum as the fitting result

  Nbks = NTotData[:,"NBk"]
  NTotDiffSqs = 0.5 * (NTotData[:,"NTot1"] .- NTotData[:,"NTot2"]).^2

  if basisFuncType == "Legendre"
    basisFuncs = [x -> legendrep(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) for n in 0:basisFuncDeg]
    basisFuncDerivs = [x -> legendreDeriv(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) * 2.0 / (NBkMax - NBkMin) for n in 0:basisFuncDeg]
    basisFuncDeriv2s = [x -> legendreDeriv2(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) * (2.0 / (NBkMax - NBkMin))^2 for n in 0:basisFuncDeg]
  elseif basisFuncType == "LegendreExcept0th"
    basisFuncs = [x -> legendrep(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) for n in 1:basisFuncDeg]
    basisFuncDerivs = [x -> legendreDeriv(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) * 2.0 / (NBkMax - NBkMin) for n in 1:basisFuncDeg]
    basisFuncDeriv2s = [x -> legendreDeriv2(n, 2.0 * (x - NBkMin) / (NBkMax - NBkMin) - 1.0) * (2.0 / (NBkMax - NBkMin))^2 for n in 1:basisFuncDeg]
  elseif basisFuncType == "monomialExcept0th"
    basisFuncs = [x -> x^n for n in 1:basisFuncDeg]
    basisFuncDerivs = [x -> n * x^(n-1) for n in 1:basisFuncDeg]
    basisFuncDeriv2s = [x -> n * (n-1) * x^(n-2) for n in 1:basisFuncDeg]
  else
    throw(ArgumentError("Unknown basisFuncType:" * basisFuncType))
  end

  if funcType == "exp" # exp(a[1]*p[0]+a[2]*p[1]+...)
    paramNum = length(basisFuncs)
    f = (xVec, a) -> [exp(dot(a, [basisFunc(x) for basisFunc in basisFuncs])) for x in xVec]

    fDeriva = (xVec, a) -> begin
      basisFuncVals = [basisFunc(x) for x in xVec, basisFunc in basisFuncs]
      expFacs = exp.(basisFuncVals * a)
      ret = expFacs .* basisFuncVals[:,1]
      for i in 2:length(basisFuncs)
        ret = hcat(ret, expFacs .* basisFuncVals[:,i])
      end
      return ret
    end

    fDerivx = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a))
      return expFac * dot(basisFuncDerivVals, a)
    end

    fDerivax = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a))
      afDeriv = dot(basisFuncDerivVals, a)
      ret = expFac * (basisFuncDerivVals[1] + afDeriv * basisFuncVals[1])
      for i in 2:length(basisFuncs)
          ret = vcat(ret, expFac * (basisFuncDerivVals[i] + afDeriv * basisFuncVals[i]))
      end
      return ret
    end

  elseif funcType == "logistic" # a[1] / (1 + exp(a[2]*p[0]+a[3]*p[1]+...))
    paramNum = length(basisFuncs) + 1
    f = (xVec, a) -> [a[1] / (1.0 + exp(dot(a[2:end], [basisFunc(x) for basisFunc in basisFuncs]))) for x in xVec]

    fDeriva = (xVec, a) -> begin
      basisFuncVals = [basisFunc(x) for x in xVec, basisFunc in basisFuncs]
      expFacs = exp.(basisFuncVals * a[2:end])
      overallFacDerivs = 1.0 ./ (1 .+ expFacs)
      ret = hcat(overallFacDerivs)
      temp = (-a[1]) .* overallFacDerivs .* overallFacDerivs  
      for i in 1:length(basisFuncs)
          ret = hcat(ret, temp .* expFacs .* basisFuncVals[:,i])
      end
      return ret
    end

    fDerivx = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a[2:end]))
      overallFacDeriv = 1.0 / (1 + expFac)
      return (-a[1]) * expFac  * overallFacDeriv * overallFacDeriv * dot(basisFuncDerivVals, a[2:end])
    end

    fDerivax = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a[2:end]))
      overallFac = (-1) * expFac / (1 + expFac) ^ 2
      afDeriv = dot(basisFuncDerivVals, a[2:end])
      ret = overallFac * afDeriv 
      for i in 1:length(basisFuncs)
          ret = vcat(ret, (-a[1]) * overallFac * (basisFuncDerivVals[i] + afDeriv * (1 - expFac) / (1 + expFac) * basisFuncVals[i]))
      end
      return ret
    end

  elseif funcType == "exp+const" # a[1]+a[2]*exp(a[3]*p[0]+...)
    paramNum = length(basisFuncs) + 2
    f = (xVec, a) -> [a[1] + a[2] * exp(dot(a[3:end], [basisFunc(x) for basisFunc in basisFuncs])) for x in xVec]

    fDeriva = (xVec, a) -> begin
      basisFuncVals = [basisFunc(x) for x in xVec, basisFunc in basisFuncs]
      expFacs = exp.(basisFuncVals * a[3:end])
      ret = fill(1, length(xVec))
      ret = hcat(ret, expFacs)
      for i in 1:length(basisFuncs)
        ret = hcat(ret, a[2] .* expFacs .* basisFuncVals[:,i])
      end
      return ret
    end

    fDerivx = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a[3:end]))
      return a[2] * expFac * dot(basisFuncDerivVals, a[3:end])
    end

    fDerivax = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a[3:end]))
      afDeriv = dot(basisFuncDerivVals, a[3:end])
      ret = 0
      ret = vcat(ret, expFac * afDeriv)
      for i in 1:length(basisFuncs)
          ret = vcat(ret, a[2] * expFac * (basisFuncDerivVals[i] + afDeriv * basisFuncVals[i]))
      end
      return ret
    end

  elseif funcType == "poly"
    paramNum = length(basisFuncs)
    f = (xVec, a) -> [dot(a, [basisFunc(x) for basisFunc in basisFuncs]) for x in xVec]
    fDeriva = (xVec, a) -> [basisFunc(x) for x in xVec, basisFunc in basisFuncs]
    fDerivx = (x, a) -> dot([basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs], a)
    fDerivax = (x, a) -> [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]

  elseif startswith(funcType, "logistic+")

    logitFunc = (x, a) -> a[1] / (1.0 + exp(dot(a[2:end], [basisFunc(x) for basisFunc in basisFuncs])))
    logitDerivFunc = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      expFac = exp(dot(basisFuncVals, a[2:end]))
      overallFacDeriv = 1.0 / (1 + expFac)
      return (-a[1]) * expFac  * overallFacDeriv * overallFacDeriv * dot(basisFuncDerivVals, a[2:end])
    end
    logitDeriv2Func = (x, a) -> begin
      basisFuncVals = [basisFunc(x) for basisFunc in basisFuncs]
      basisFuncDerivVals = [basisFuncDeriv(x) for basisFuncDeriv in basisFuncDerivs]
      basisFuncDeriv2Vals = [basisFuncDeriv2(x) for basisFuncDeriv2 in basisFuncDeriv2s]
      expFac = exp(dot(basisFuncVals, a[2:end]))
      overallFacDeriv = 1.0 / (1 + expFac)
      return (-a[1]) * expFac * overallFacDeriv^2 * ((-2 * expFac * overallFacDeriv + 1) * dot(basisFuncDerivVals, a[2:end])^2 + dot(basisFuncDeriv2Vals, a[2:end]))
    end

    connectedFuncType = funcType[10:end]
    if connectedFuncType == "linear"
      connectedFuncDeg = 1
    elseif connectedFuncType == "quadratic"
      connectedFuncDeg = 2
    else
      throw(ArgumentError("Unknown connectedFuncType:" * connectedFuncType))
    end

    logitParamNum = length(basisFuncs) + 1
    paramNum = logitParamNum + connectedFuncDeg
    idMat = Matrix{Float64}(I, paramNum, paramNum)

    f = (xVec, a) -> begin
      xCon = a[logitParamNum+1]
      aLogit = a[1:logitParamNum]
      yCon = logitFunc(xCon, aLogit)
      derivCon = logitDerivFunc(xCon, aLogit)
      conFuncCoefs = vcat([yCon, derivCon], a[logitParamNum+2:end])
      ret = [x > xCon ? logitFunc(x, aLogit) : dot(conFuncCoefs, [(x - xCon)^n for n in 0:connectedFuncDeg]) for x in xVec]
      return ret
    end

    fDeriva = (xVec, a) -> begin
      xCon = a[logitParamNum+1]
      aLogit = a[1:logitParamNum]
      ret = hcat((f(xVec, a .+ paramDerivDelta .* idMat[:,1]) .- f(xVec, a .- paramDerivDelta .* idMat[:,1])) ./ 2.0 ./ paramDerivDelta)
      for i in 2:logitParamNum
        ret = hcat(ret, (f(xVec, a .+ paramDerivDelta .* idMat[:,i]) .- f(xVec, a .- paramDerivDelta .* idMat[:,i])) ./ 2.0 ./ paramDerivDelta)
      end
      logitDeriv2xCon = logitDeriv2Func(xCon, aLogit)
      conFuncCoefs = a[logitParamNum+2:end]
      fDerivxCon = [x > xCon ? 0.0 : (logitDeriv2xCon * (x - xCon) - dot(conFuncCoefs, [n * (x - xCon)^(n-1) for n in 2:connectedFuncDeg])) for x in xVec]
      ret = hcat(ret, fDerivxCon)
      for n in 2:connectedFuncDeg
        ret = hcat(ret, [x > xCon ? 0.0 : (x - xCon)^n for x in xVec])
      end
      return ret
    end

    fDerivx = (x, a) -> begin
      xCon = a[logitParamNum+1]
      aLogit = a[1:logitParamNum]
      if x > xCon
        return logitDerivFunc(x, aLogit)
      else
        return logitDerivFunc(xCon, aLogit) + sum([n * a[logitParamNum+n] * (x - xCon)^(n-1) for n in 2:connectedFuncDeg])
      end
    end

    fDerivax = (x, a) -> begin
      ret = (fDerivx(x, a .+ paramDerivDelta .* idMat[:,1]) .- fDerivx(x, a .- paramDerivDelta .* idMat[:,1])) ./ 2.0 ./ paramDerivDelta
      for i in 2:logitParamNum
        ret = vcat(ret, (fDerivx(x, a .+ paramDerivDelta .* idMat[:,i]) .- fDerivx(x, a .- paramDerivDelta .* idMat[:,i])) ./ 2.0 ./ paramDerivDelta)
      end
      xCon = a[logitParamNum+1]
      aLogit = a[1:logitParamNum]
      conFuncCoefs = a[logitParamNum+2:end]
      ret = vcat(ret, x > xCon ? 0.0 : (logitDeriv2Func(xCon, aLogit) - dot(conFuncCoefs, [n * (n-1) * (x - xCon)^(n-2) for n in 2:connectedFuncDeg])))
      for n in 2:connectedFuncDeg
        ret = vcat(ret, x > xCon ? 0.0 : (n * (x - xCon)^(n-1)))
      end
      return ret
    end

  else  
    throw(ArgumentError("Unknown funcType:" * funcType))
  end

  aIni = paramIni === nothing ? zeros(paramNum) : paramIni
  if fDeriva === nothing
    fitResult = curve_fit(f, Nbks, NTotDiffSqs, aIni)
  else
    fitResult = curve_fit(f, fDeriva, Nbks, NTotDiffSqs, aIni)
  end
  delNSqFit = N -> only(f(N, fitResult.param))
  psFit = N -> fDerivx(N, fitResult.param)
  if fDeriva === nothing
    delNSqErr = nothing
    psErr = nothing
  else
    paramCov = estimate_covar(fitResult)
    delNSqErr = N -> begin
      jac = fDeriva([N], fitResult.param)
      return sqrt(only(jac * paramCov * jac'))
    end
    psErr = N -> begin
      jac = fDerivax(N, fitResult.param)
      return sqrt(jac' * paramCov * jac)
    end
  end

  return fitResult, delNSqFit, psFit, delNSqErr, psErr
end

function FitByBinAve(NTotData::DataFrame, NBks)
  ### Input
  # NTotData: Stores sampled e-fold values. (See the comment in the function FitByLS)
  # NBks: Boundaries of NBk bins. Must be sorted.
  ### Output: Tuple of the following
  # delNSq: Real vector with n-1 entries, where n=length(NBks)
  #         i-th entry is the average of 0.5*(NTot1-NTot2)^2 for NBk s.t. NBks[i]<=NBk<NBks[i+1]
  # ps: Real vector with n-2 entries. i-th entry corresponds to the power spectrum for NBk=NBks[i+1]
  # delNSqErr: Real vector with n-1 entries. Standard error of delNSq.
  # psErr: Real vector with n-2 entries. Standard error of ps.
  # NBksdelNSq: Real vector with n-1 entries. NBk at which delNSq is evaluated.
  # NBksPS: Real vector with n-2 entries. NBk at which ps is evaluated.

  delNSq = []
  delNSqVar = []
  delNSqErr = []
  for i in 1:length(NBks)-1
    NTotDataFilterd = filter(row -> NBks[i] <= row.NBk < NBks[i+1], NTotData)
    delNSq = vcat(delNSq, mean(0.5 * (NTotDataFilterd.NTot1 .- NTotDataFilterd.NTot2).^2))
    delNSqVar = vcat(delNSqVar, (mean(0.25 * (NTotDataFilterd.NTot1 .- NTotDataFilterd.NTot2).^4) - delNSq[end]^2) / nrow(NTotDataFilterd))
    delNSqErr = vcat(delNSqErr, sqrt(delNSqVar[end]))
  end
  NBksdelNSq = 0.5 * (NBks[1:end-1] .+ NBks[2:end])
  ps = (delNSq[2:end] - delNSq[1:end-1]) ./ (NBksdelNSq[2:end] - NBksdelNSq[1:end-1])
  psErr = sqrt.((delNSqVar[2:end] + delNSqVar[1:end-1]) ./ (NBksdelNSq[2:end] .- NBksdelNSq[1:end-1]).^2)
  NBksPS = 0.5 * (NBksdelNSq[1:end-1] .+ NBksdelNSq[2:end])
  return delNSq, ps, delNSqErr, psErr, NBksdelNSq, NBksPS

end  
