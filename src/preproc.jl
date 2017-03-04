using HTTP
using JSON
using DataFrames


"""
:param key:
    [
        ["2016-11-15", "TXO", "201611", "C", "9000"]
    ]
"""
function fetch(keys::Array)
    base_url = "http://localhost:5984"
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


function k(df::DataFrame)
    ks = groupby(df, :time)
    k_df = DataFrame(
        [Int64, Float64, Float64, Float64, Float64, String],
        [:vol, :open, :high, :low, :close, :time],
        0)

    for k ∈ ks
        vol::Int64 = sum(k[:vol])
        open::Float64 = k[1, :price]
        high::Float64 = max(k[:price])
        low::Float64 = min(k[:price])
        close::Float64 = k[end, :price]
        time::String = k[1, :time]

        push!(k_df, [vol, open, high, low, close, time])
    end

    k_df
end
