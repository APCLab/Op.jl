using DataFrames
using HTTP
using JLD
using JSON
using MarketTechnicals
using Plots
using TimeSeries

out_dir = joinpath(dirname(@__FILE__), "..", "out")
data_dir = joinpath(dirname(@__FILE__), "..", "data")
data_jld = joinpath(data_dir, "data.jld")
plot_size = (4000, 800)

"""
:param key:
    [
        ["2016-11-15", "TXO", "201611", "C", "9000"]
    ]
"""
function _fetch(keys::Array)
    server = get(ENV, "COUCH_SERVER", "localhost")
    base_url = "http://$server:5984"
    view = "/market/_design/options/_view/txo?reduce=false&limit=5"
    headers = Dict("Content-Type" => "application/json")
    body = JSON.json(Dict("keys" => keys))

    res = HTTP.post(base_url * view, body = body, headers = headers)
    JSON.parse(string(res))["rows"]
end


function extract(rows::Array)
    df = DataFrame([Int64, Float64, String], [:vol, :price, :time], 0)

    map([r["value"] for r ∈ rows]) do r
        map(r) do x
            push!(df, values(x))
        end
    end

    df
end


"Generate k bar"
function k(df::DataFrame)
    ks = groupby(df, :time)
    k_df = DataFrame(
        [Int64, Float64, Float64, Float64, Float64, String],
        [:vol, :open, :high, :low, :close, :time],
        0)

    for _k ∈ ks
        vol::Int64 = sum(_k[:vol])
        open::Float64 = _k[1, :price]
        high::Float64 = maximum(_k[:price])
        low::Float64 = minimum(_k[:price])
        close::Float64 = _k[end, :price]
        time::String = _k[1, :time]

        push!(k_df, [vol, open, high, low, close, time])
    end

    k_df
end


"""
History Volatility

Calculate σ like a boss

:param k_df: a vertor of k bars
:param type: (history|implied)
:param T: the total days of ``k_df``
"""
function σ_his()
    server = get(ENV, "COUCH_SERVER", "localhost")
    url = "http://$server:5984/twse/_design/twii-daily/_view/close"
    headers = Dict("Content-Type" => "application/json")
    res = HTTP.get(url, headers = headers)
    rows = JSON.parse(string(res))["rows"]

    df = DataFrame([String, Float64], [:date, :close], 0)
    [push!(df, [r["key"], r["value"], NaN]) for r ∈ rows]

    df[:year_σ] = σ_interval(252, 21, df)
    df[:mon_σ] = σ_interval(22, 21, df)
    df[:daily_σ] = σ_interval(1, 21, df)
    df
end


doc"""
:param int: interval
:param smp: sampling number for volatility

Price change caculation::

    ln \frac{P_{t+1}}{P}
"""
function σ_interval(int::Int64, smp::Int64, df::DataFrame)
    returns = [
        (i < 1 + int) ?
        NaN : (log(df[i, :Close] / df[i - int, :Close]))
        for i ∈ 1:length(df[:Close])
    ]
    [
        (idx < 1 + smp) ?
        NaN : std(returns[idx - smp:idx])
        for idx ∈ 1:length(returns)
    ] * √(252.0 / int)
end


function plot_σ()
    pyplot()
    σ = σ_his()

    σ[:year_σ] *= 100
    σ[:mon_σ] *= 100
    σ[:daily_σ] *= 100

    plot(σ[:year_σ], ylabel="σ (%)", label=:year, size=plot_size)
    plot!(σ[:mon_σ], label=:mon)
    plot!(σ[:daily_σ], label=:daily)
    savefig(joinpath(out_dir, "vola.png"))
end

"round to nearest strike price"
function strike_price(x)::Float64
    y = div(x, 100) * 100.0
    z = y + 100
    (abs(y - x) < abs(z - x)) ? y : z
end


"""
Find the index of ``date`` in ``trade_date``

Output is the index of trading date
"""
function find_date_idx(trade_date, dates)
    map(arr) do x
        find(trade_date .== x)[1]
    end
end


function get_T(trade_date, set_date, dates)
    set_idx = find_date_idx(trade_date, set_date)
    d_idx = find_date_idx(trade_date, dates)

    map(zip(d_idx, set_idx)) do x
        x[2] - x[1]
    end
end


"""
Create empty jld
"""
function create_jld()
    jldopen(data_jld, "w") do f
        addrequire(f, DataFrames)
    end
end


function load_twii()
    global twii = readtable(joinpath(data_dir, "twii.csv"))

    d = Date(twii[:date])
    p = twii[:price]

    ta = TimeArray(d, p)
end


"""
convert setdate csv to jld
"""
function load_setdates(group=true, write=false)
    df = readtable(joinpath(data_dir, "setdates.csv"))
    df[:Date] = Date.(df[:Date])
    df[:Contract] = map(string, (df[:Contract]))
    delete!(df, :Symbol)

    if group
        df = DataFrame()
        for g in groupby(df, :Contract)
            df = vcat(df, g[end, :])
        end
    end

    if write
        rename!(df, :Date, :Settlement)
        write_jld("setdates", df)
    end

    df
end


function load_tx()
    tx = readtable(joinpath(data_dir, "tx.csv"))
    delete!(tx, :x)
    tx[:Date] = Date.(tx[:Date])

    s = load_setdates(false, false)
    join(tx, s, on=[:Date, :Contract])
end


"""
Write DataFrame to jld file
"""
function write_jld(name::AbstractString, df::DataFrame)
    jldopen(data_jld, "r+") do f
        addrequire(f, DataFrames)
        write(f, name, df)
    end
end


function jld_del(name::AbstractString)
    jldopen(data_jld, "r+") do f
        delete!(f, name)
    end
end


function get_txo()
    df = load(data_jld, "txo")

    df[df[:T] .> 3, :]
    df[:BS_err] = df[:BS] .- df[:Close]

    df
end


function get_macd_sma()
    tx = load_tx()
    ta = TimeArray(Vector(tx[:Date]), Vector(tx[:Close]))
    m = macd(ta)
    ma = sma(ta, 20)
    t = DataFrame(Date=m.timestamp, Dif=m.values[:, 2])
    t2 = DataFrame(Date=ma.timestamp, SMA=ma.values[:, 1])
    join(join(tx, t, on=:Date), t2, on=:Date)
end
