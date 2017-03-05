module Op

using Distributions: cdf, Normal

export C

"""
:param S: stock price (Array)
:param L: strike price
:param T: time to execute
:param σ: volatility
:param r: risk-free rate of interest
"""
C(S, L, T, σ, r = 1.04)::Float64 = S * N(D1(S, L, T, r, σ)) - L * e^(-r * T) * N(D2(S, L, T, r, σ))

D1(S, L, T, σ, r = 1.04)::Float64 = (log(S / L) + (r + 0.5 * σ^2) * T) / (σ * √T)
D2(S, L, T, σ, r = 1.04)::Float64 = (log(S / L) + (r - 0.5 * σ^2) * T) / (σ * √T)

N(x::Float64)::Float64 = cdf(Normal(), x)

end  # module
