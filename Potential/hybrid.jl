function makePotFuncHybrid(Lambda, mu1, mu2, M, phic)

  function f(infVec)
    phi = infVec[1]
    psi = infVec[2]
    return Lambda^4 * ((1 - (psi / M)^2)^2 + 2 * (phi * psi / phic / M)^2 + (phi - phic) / mu1 - (phi - phic)^2 / (mu2 * mu2))
  end

  return f
end

function makePotDerFuncHybrid(Lambda, mu1, mu2, M, phic)

  function f(infVec)
    phi = infVec[1]
    psi = infVec[2]
    potDerVec = zeros(2)
    potDerVec[1] = Lambda^4 * (1 / mu1 - 2 * (phi - phic) / (mu2 * mu2) + 4 * phi * (psi / phic / M)^2)
    potDerVec[2] = Lambda^4 * (4 * ((psi / M)^2 - 1) * psi / M / M + 4 * psi * (phi / phic / M)^2)
    return potDerVec
  end

  return f
end

function makePotDer2FuncHybrid(Lambda, mu2, M, phic)

  function f(infVec)
    phi = infVec[1]
    psi = infVec[2]
    potDer2Vec = zeros(2)
    potDer2Vec[1] = Lambda^4 * (-2 / (mu2 * mu2) + 4 * (psi / phic / M)^2)
    potDer2Vec[2] = Lambda^4 * (4 * (3 * (psi / M)^2 - 1) / M / M + 4 * (phi / phic / M)^2)
    return potDer2Vec
  end

  return f
end

