module Op

using Distributions: cdf, Normal

export C


"""
:param S: stock price
:param L: strike price
:param T: time to execute
:param r: risk-free interest rate
:param σ: volatility
"""
C (S, L, T, r, σ) = S * N(D1(S, L, T, r, σ)) - L * e^(-r * T) * N(D2(S, L, T, r, σ))

D1(S, L, T, r, σ) = (log(S / L) + (r + 0.5 * σ^2) * T) / (σ * √T)
D2(S, L, T, r, σ) = (log(S / L) + (r - 0.5 * σ^2) * T) / (σ * √T)

N(x) = cdf(Normal(), x)

end Op  # module
