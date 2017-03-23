module Op

using Distributions: cdf, pdf, Normal

export C

"""
Black-scholes Model

:param S: stock price (Array)
:param L: strike price
:param T: time to maturity (in years -> 252)
:param σ: volatility (Array)
:param r: risk-free rate of interest, it should be continuous rate.
"""
C(S::AbstractArray, L::AbstractArray, T::AbstractArray, σ::AbstractArray,
  r::Real = log(1.04))::Array{Float64} =
    S .* N(D1(S, L, T, σ, r)) - L .* e.^(-r .* T) .* N(D2(S, L, T, σ, r))

D1(S, L, T, σ, r = 1.04)::Array{Float64} =
    (log(S ./ L) .+ (r + 0.5 * σ.^2) .* T) ./ (σ .* √T)
D2(S, L, T, σ, r = 1.04)::Array{Float64} =
    (log(S ./ L) .+ (r - 0.5 * σ.^2) .* T) ./ (σ .* √T)

N(x::Array{Float64})::Array{Float64} = cdf(Normal(), x)

end  # module
