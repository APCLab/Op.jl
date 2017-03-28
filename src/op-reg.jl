using MXNet
using Distributions
using Plots
using DataFrames

import TimeSeries: TimeArray

pyplot()

out_dir = joinpath(dirname(@__FILE__), "..", "out")
plot_size = (1200, 800)

"""
Mean Normalization

:param arr: the DataArray
:return: a normalized DataArray
"""
norm_col(arr::AbstractArray) = (arr - mean(arr)) / std(arr)

# data generating process
function load(col::Array{Symbol})
    global cc = readtable("./ccc_macd2.csv")
    cc[:price] /= 100
    cc[:contract] /= 100
    cc[:mon_σ] *= 10

    global data = convert(Array{Float64}, cc[col])'

    global train_idx = Int(round(size(data)[2] * 0.80))

    global data_train = data[:, 1:train_idx]
    global data_test = data[:, train_idx+1:end]

    global target = convert(Array{Float64}, cc[:close])'
    global target_train = target[:, 1:train_idx]
    global target_test = target[:, train_idx+1:end]
end

function load_twii()
    global twii = readtable("../data/twii.csv")

    d = Date(twii[:date])
    p = twii[:price]

    ta = TimeArray(d, p)
end

function plot_bs()
    load([:BS])
    scatter(Array(cc[train_idx+1:end, :close]),
            Array(cc[train_idx+1:end, :BS]),
            xlim=(0, 600), ylim=(0, 600),
            xlabel="real price", ylabel="pred price",
            title="BS Model",
            size=plot_size)
    savefig(joinpath(out_dir, "bs.png"))
end

function input(model::Symbol)
    cols = if model == :orig
        global plot_title = "P, P_s, σ_month, T"

        [:price, :contract, :mon_σ, :T]

    elseif model == :ta  # with TA: MACD
        global plot_title = "P, P_s, σ_month, T, MACD"

        [:price, :contract, :mon_σ, :T, :macd]
    end

    load(cols)
end

function get_provider()
    batchsize = 64

    trainprovider = mx.ArrayDataProvider(
        :data => data_train,
        :label => target_train,
        batch_size = batchsize,
        shuffle = true,
        data_padding = 2.0,
        label_padding = 1.0,
        )

    evalprovider = mx.ArrayDataProvider(
        :data => data_test,
        :label => target_test,
        batch_size = batchsize,
        shuffle = false,
        data_padding = 2.0,
        label_padding = 1.0,
        )

    plotprovider = mx.ArrayDataProvider(
        :data => data_test,
        :label => target_test,
        shuffle = false,
        data_padding = 2.0,
        label_padding = 1.0,
        )

    trainprovider, evalprovider, plotprovider
end

function plot_pred(target_test, fit, net, lname::String, mname::String)
    scatter(target_test', fit',
            xlim=(0, 600), ylim=(0, 600),
            xlabel="real price", ylabel="pred price",
            title=plot_title,
            size=plot_size)

    annotate!([
        (400 + 2, 110 + 2, text("loss layer =  $lname", 10, :black, :left))
        (400 + 2,  90 - 2, text("metric = $mname", 10, :black, :left))
    ])

    name = "$lname-$mname"

    savefig(joinpath(out_dir, "out-$name.png"))

    dot_file = joinpath(out_dir, "net-$name.dot")
    net_pic = joinpath(out_dir, "net-$name.png")

    write(dot_file, mx.to_graphviz(net))
    run(`dot -Tpng -o $net_pic $dot_file`)
end

"""
Network Factory

Swapping different loss layer:
    * LinearRegressionOutput
    * MAERegressionOutput
"""
function network_factory()
    # create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
    # or add/remove a layer
    label = mx.Variable(:label)
    net =
        @mx.chain mx.Variable(:data) =>
                mx.FullyConnected(num_hidden = 1024)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 512)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 256)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 128)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 64)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 32)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 16)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 8)  =>
                mx.Activation(act_type = :relu) =>
                mx.FullyConnected(num_hidden = 4)  =>
                mx.Activation(act_type = :relu) =>
                #= mx.FullyConnected(num_hidden = 2)  => =#
                #= mx.Activation(act_type = :relu) => =#
                mx.FullyConnected(num_hidden = 1)
                #= mx.LinearRegressionOutput(label) =#
                #= mx.MAERegressionOutput(label) =#

    loss_layers = (
        mx.LinearRegressionOutput(net, label),
        mx.MAERegressionOutput(net, label),
    )
    name_postfix = (
        "linreg",
        "mae",
    )

    metrics = (
        mx.MSE,
        mx.NMSE,
    )
    metric_names = (
        "mse",
        "nmse",
    )

    for l ∈ zip(loss_layers, name_postfix)
        for m ∈ zip(metrics, metric_names)
            produce((l, m))
        end
    end
end

input(:orig)
#= input(:ta) =#
trainprovider, evalprovider, plotprovider = get_provider()

#= loss = mx.MakeLoss( =#
#=     mx.Activation(abs(net .- label) ./ label, act_type=:sigmoid), =#
#=     grad_scale = 10, =#
#=     name = :loss, =#
#=     #= normalization = "batch", =# =#
#= ) =#
#= net = mx.Group(mx.BlockGrad(net), loss) =#

# simple small net for quick test
#= net_qtest = =#
#=     @mx.chain mx.Variable(:data) => =#
#=               mx.FullyConnected(num_hidden = 10)  => =#
#=               mx.Activation(act_type = :relu) => =#
#=               mx.FullyConnected(num_hidden = 5)  => =#
#=               mx.Activation(act_type = :relu) => =#
#=               mx.FullyConnected(num_hidden = 1)  => =#
#=               mx.MAERegressionOutput(label) =#

# final model definition, don't change, except if using gpu
cpus = [mx.cpu(i) for i in 0:3]
gpu = mx.gpu()

nets = @task network_factory()
for ((net, lname), (metric, mname)) ∈ nets
    model = mx.FeedForward(net, context=gpu)

    # set up the optimizer: select one, explore parameters, if desired
    #= optimizer = mx.SGD(lr=0.001, momentum=0.9, weight_decay=0.00001) =#
    optimizer = mx.ADAM(weight_decay=0.0002)

    # train, reporting loss for training and evaluation sets
    epoch = 100
    # initial training with small batch size, to get to a good neighborhood
    mx.fit(
        model, optimizer,
        initializer = mx.UniformInitializer(0.1),
        eval_metric = metric(),
        trainprovider,
        eval_data = evalprovider,
        n_epoch = epoch)

    fit = mx.predict(model, plotprovider)
    #= plot_pred(target_test, fit; name="out-pre.png") =#

    # more training with the full sample
    mx.fit(
        model, optimizer,
        eval_metric = metric(),
        trainprovider,
        eval_data = evalprovider,
        n_epoch = epoch)

    fit = mx.predict(model, plotprovider)

    plot_pred(target_test, fit, net, lname, mname)

    result = DataFrame(
        fit = reshape(fit, length(fit)),
        real = reshape(target_test, length(target_test)))
end
