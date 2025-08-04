function Hubble(pot, velVec)
  return sqrt((pot + 0.5 * velVec' * velVec) / 3)
end

function Epsilon(pot, potDer)
  return 0.5 * potDer' * potDer / pot^2
end

function EpsilonH(pot, vel)
  velSq = vel' * vel
  return 1.5 * velSq / (0.5 * velSq + pot)
end

function Eta(pot, potDer2)
  if length(potDer2) != 1
    throw(ArgumentError("The slow-roll param Eta is defined only for single-field models."))
  end
  return potDer2[1] / pot
end

function PInflatonNextSR(pot, potDer2, h, sigmaH, vel) # c.f. 2405.10692

  epsilon = EpsilonH(pot, vel)
  eta = Eta(pot, potDer2)
  const1 = -1.7810601561285404 # 10 - 6 * eulergamma - 12 * log(2)
  const2 = 0.03648997397857667 # 2 - eulergamma - 2 * log(2)
  ret = (h / 2 / pi) ^ 2 * (sigmaH / 2 / h) ^ (-6 * epsilon + 2 * eta) * (1 + const1 * epsilon - 2 * const2 * eta)
  if ret < 0
    throw(ArgumentError("Negative PInf"))
  end
  return ret

end

function PInflatonStarobinsky(h, gamma, alpha, sigma)
  return (h / 2 / pi)^2 / (2 * (alpha * sigma)^6 * gamma^2) *
    (3 * cos(2 * (alpha - 1) * sigma) *
    (gamma^2 * (alpha^4 * (4 * alpha - 7) * sigma^6 + alpha^3 * (7 * alpha - 16) * sigma^4 + (3 - 12 * alpha) * sigma^2 - 3) +
     gamma * (2 * (5 - 2 * alpha) * alpha^4 * sigma^6 + 2 * (14 - 5 * alpha) * alpha^3 * sigma^4 + 6 * (4 * alpha - 1) * sigma^2 + 6) -
     3 * (alpha^4 * sigma^6 - (alpha - 4) * alpha^3 * sigma^4 + (4 * alpha - 1) * sigma^2 + 1)) +
    (sigma^2 + 1) * ((9 - 18 * gamma) * (alpha^2 * sigma^2 + 1)^2 + gamma^2 * (2 * alpha^6 * sigma^6 + 9 * alpha^4 * sigma^4 + 18 * alpha^2 * sigma^2 + 9)) +
    6 * sigma * sin(2 * sigma * (1 - alpha)) * (alpha^5 * (gamma - 1) * gamma * sigma^4 * (sigma^2 - 1) + alpha^4 * (7 * gamma^2 - 10 * gamma + 3) * sigma^4 - alpha^3 * (4 * gamma^2 - 7 * gamma + 3) * sigma^2 * (sigma^2 - 1) - 3 * alpha * (gamma - 1)^2 * (sigma^2 - 1) - 3 * (gamma - 1)^2))
end