using Distributions: pdf, cdf, Normal


"""
Call price of Black-scholes Model

:param S: stock price (Array)
:param L: strike price
:param T: time to maturity (in years -> 252)
:param σ: volatility (Array)
:param r: risk-free rate of interest, it should be continuous rate.

:param round: round small value to ``0.1``, default is ``false``
"""
function C(S::AbstractArray{F, N}, L::AbstractArray{F, N},
           T::AbstractArray{F, N}, σ::AbstractArray{F, N},
           r::Real = log(1.04);
           round=false) where {F <: AbstractFloat, N}
    c = S .* N(D1(S, L, T, σ, r)) - L .* e.^(-r .* T) .* N(D2(S, L, T, σ, r))

    if round == false
        map(c) do x
            if isna(x)
                NA
            else
                (x <= 0.1) ? 0.1 : x
            end
        end
    else
        c
    end
end


D1(S, L, T, σ, r = 1.04) =
    (log(S ./ L) .+ (r + 0.5 * σ.^2) .* T) ./ (σ .* √T)


D2(S, L, T, σ, r = 1.04) =
    (log(S ./ L) .+ (r - 0.5 * σ.^2) .* T) ./ (σ .* √T)


function N{T<:Float64,M}(x::AbstractArray{T,M})
    ret = similar(x)
    d = Normal()

    for i ∈ 1:length(x)
        ret[i] = isna(x[i]) ? NA : cdf(d, x[i])
    end

    ret
end


vega(S, L, T, σ, r = 1.04) =
    S .* pdf(Normal(), D1(S, L, T, σ, r)) .* √T


"""
http://quant.stackexchange.com/questions/7761
https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

:param P: the option price
"""
function σ_imp(S::AbstractArray, L::AbstractArray, T::AbstractArray,
               P::AbstractArray, r::Real = log(1.04))
    σ = Array{Float64}(length(S)) .= 1   # initial guess

    for i ∈ 1:1000
        BS = C(S, L, T, σ, r)
        V = vega(S, L, T, σ)
        σ = σ .- (BS .- S) ./ V
    end

    σ
end
