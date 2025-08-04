function makePotFuncChaotic(m)

  function f(infVec)
    return 0.5 * m * m * infVec[1]^2
  end

  return f
end

function makePotDerFuncChaotic(m)

  function f(infVec)
    return [m * m * infVec[1]]
  end

  return f
end

function makePotDer2FuncChaotic(m)

  function f(infVec)
    return [m * m]
  end

  return f
end
