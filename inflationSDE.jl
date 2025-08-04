include("inflationBasicEqs.jl")

function makeDriftAndVolFunc(
  potFunc, potDerFunc, potDer2Func, endCond;
  epsilonEnd=1, infECFormula=nothing, infPSType="H/2pi", additionalParams::Union{Dict, Nothing}=nothing)

  if infPSType == "Starobinsky"
    phiEnd = additionalParams["phiEnd"]
    APlus = additionalParams["APlus"]
    AMinus = additionalParams["AMinus"]
    phi0 = additionalParams["phi0"]
    sigma = additionalParams["sigma"]
    return makeDriftAndVolFuncStarobinsky(potFunc, potDerFunc, phiEnd, APlus, AMinus, phi0, sigma)
  end

  if endCond == "Epsilon"
    endCondFunc = (infV, velV) -> EpsilonH(potFunc(infV), velV) > epsilonEnd
  elseif endCond == "Inflaton"
    if infECFormula === nothing
      throw(ArgumentError("endCond is Inflaton, but infECFormula is not given"))
    end
    endCondFunc = (infV, velV) -> infECFormula(infV)
  else
    throw(ArgumentError("Unknown endCond:" * endCond))
  end

  function drift!(driftVec, u, p, t)
    nInf = div(size(u)[1], 2)
    infVec = u[1:nInf]
    velVec = u[(nInf+1):(2*nInf)]

    if endCondFunc(infVec, velVec)
      driftVec .= 0
    else
      potDerVec = potDerFunc(infVec)
      h = Hubble(potFunc(infVec), velVec)
      for i in 1:nInf
        driftVec[i] = velVec[i] / h
        driftVec[i + nInf] = -3 * velVec[i] - potDerVec[i] / h
      end
    end
  end

  if infPSType == "H/2pi"
    infPSFunc = (N, infV, velV) -> (Hubble(potFunc(infV), velV) / (2 * pi))^2
  elseif infPSType == "NextSR"
    sigma = additionalParams["sigma"]
    infPSFunc = (N, infV, velV) -> begin
      pot = potFunc(infV)
      h = Hubble(pot, velV)
      return PInflatonNextSR(pot, potDer2Func(infV), h, sigma * h, velV)
    end
  else
    throw(ArgumentError("Unknown infPSType:" * infPSType))
  end

  function vol!(volVec, u, p, t)
    nInf = div(size(u)[1], 2)
    infVec = u[1:nInf]
    velVec = u[(nInf+1):(2*nInf)]

    pot = potFunc(infVec)
    h = Hubble(pot, velVec)

    if endCondFunc(infVec, velVec)
      volVec .= 0
    else
      infPS = infPSFunc(t, infVec, velVec)
      volVec[1:nInf] .= sqrt(infPS)
      volVec[(nInf+1):end] .= 0
    end
  end

  return drift!, vol!

end

function makeDriftAndVolFuncStarobinsky(potFunc, potDerFunc, phiEnd, APlus, AMinus, phi0, sigma)

  function drift!(driftVec, u, p, t)
    nInf = div(size(u)[1], 2)
    infVec = u[1:nInf]
    velVec = u[(nInf+1):(2*nInf)]

    if u[1] < phiEnd
      driftVec .= 0
    else
      potDerVec = potDerFunc(infVec)
      h = Hubble(potFunc(infVec), velVec)
      for i in 1:nInf
        driftVec[i] = velVec[i] / h
        driftVec[i + nInf] = -3 * velVec[i] - potDerVec[i] / h
      end
    end
  end

  infPSFunc = (N, u) -> begin
    infl = u[1]
    vel = u[2]
    h = Hubble(potFunc(infl), vel)
    return infl > phi0 ? (h / (2 * pi))^2 : PInflatonStarobinsky(h, APlus / AMinus, exp(N - u[3]), sigma)
  end

  function vol!(volVec, u, p, t)
    nInf = div(size(u)[1], 2)
    infVec = u[1:nInf]
    velVec = u[(nInf+1):(2*nInf)]

    pot = potFunc(infVec)
    potDerVec = potDerFunc(infVec)
    h = Hubble(pot, velVec)

    if u[1] < phiEnd
      volVec .= 0
    else
      if u[1] < phi0 && u[3] == 0
        u[3] = t
      end
      infPS = infPSFunc(t, u)
      volVec[1:nInf] .= sqrt(infPS)
      volVec[(nInf+1):end] .= 0
    end
  end

  return drift!, vol!

end