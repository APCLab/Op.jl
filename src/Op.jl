module Op

using DataFrames
using Distributions: cdf, pdf, Normal

export C

"""
Black-scholes Model

:param S: stock price (Array)
:param L: strike price
:param T: time to maturity (in years -> 252)
:param σ: volatility (Array)
:param r: risk-free rate of interest, it should be continuous rate.

:param round: round small value to ``0.1``, default is ``false``
"""
function C(S::AbstractArray, L::AbstractArray, T::AbstractArray,
           σ::AbstractArray, r::Real = log(1.04);
           round=false)::Array{Float64}
    c = S .* N(D1(S, L, T, σ, r)) - L .* e.^(-r .* T) .* N(D2(S, L, T, σ, r))

    if round == false
        map(c) do x
            (x <= 0.1) ? 0.1 : x
        end
    else
        c
    end
end

D1(S, L, T, σ, r = 1.04)::Array{Float64} =
    (log(S ./ L) .+ (r + 0.5 * σ.^2) .* T) ./ (σ .* √T)
D2(S, L, T, σ, r = 1.04)::Array{Float64} =
    (log(S ./ L) .+ (r - 0.5 * σ.^2) .* T) ./ (σ .* √T)

N(x::Array{Float64})::Array{Float64} = cdf(Normal(), x)

end  # module
