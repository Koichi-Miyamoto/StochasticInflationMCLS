function makePotFuncStarobinsky(V0, APlus, AMinus, phi0)

  function f(infVec)
    phi = infVec[1]
    return V0 + (phi > phi0 ? APlus : AMinus) * (phi - phi0)
  end

  return f
end

function makePotDerFuncStarobinsky(APlus, AMinus, phi0)

  function f(infVec)
    phi = infVec[1]
    return phi > phi0 ? APlus : AMinus
  end

  return f
end
