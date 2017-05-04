using MXNet
using Distributions
using Plots

import TimeSeries: TimeArray

include("./preproc.jl")

pyplot()

out_dir = joinpath(dirname(@__FILE__), "..", "out")
plot_size = (1200, 800)
plot_rng = 1000  # ploting range for x and y axis

"""
Mean Normalization

:param arr: the DataArray
:return: a normalized DataArray
"""
norm_col(arr::AbstractArray) = (arr - mean(arr)) / std(arr)


# data generating process
function load_dataset(cols::Array{Symbol})
    global cc = get_txo()

    # roughly rescaling
    cc[:S] /= 100
    cc[:Strike] /= 100
    cc[:σ] *= 100
    cc[:SMA] /= 100

    global data = convert(Array{Float64}, cc[cols])'

    global train_idx = Int(round(size(data)[2] * 0.80))

    global data_train = data[:, 1:train_idx]
    global data_test = data[:, train_idx+1:end]

    global target = convert(Array{Float64}, cc[:Close])'
    global target_train = target[:, 1:train_idx]
    global target_test = target[:, train_idx+1:end]
end


function plot_bs()
    load_dataset([:BS])
    scatter(Array(cc[train_idx+1:end, :Close]),
            Array(cc[train_idx+1:end, :BS]),
            xlim=(0, plot_rng), ylim=(0, plot_rng),
            xlabel="real price", ylabel="pred price",
            title="BS Model",
            size=plot_size)
    savefig(joinpath(out_dir, "bs.png"))
end


function input(model::Symbol)
    cols = if model == :orig
        global plot_title = "P, P_s, σ_month, T"
        global iname = "model1"

        [:S, :Strike, :σ, :T]

    elseif model == :ta  # with TA: MACD + MA(20)
        global plot_title = "P, P_s, σ_month, T, MACD, MA"
        global iname = "model2"

        [:S, :Strike, :σ, :T, :Dif, :SMA]

    elseif model == :bs  # with TA & BS
        global plot_title = "P, P_s, σ_month, T, MACD, MA, BS"
        global iname = "model3"

        [:S, :Strike, :σ, :T, :Dif, :SMA, :BS]

    elseif model == :bs_err  # with TA & BS
        global plot_title = "P, P_s, σ_month, T, MACD, MA, BS_err"
        global iname = "model4"

        [:S, :Strike, :σ, :T, :Dif, :SMA, :BS_err]
    end

    load_dataset(cols)
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


"""
:param iname: input data name
:param lname: layer name
:param mname: metric name
"""
function plot_pred(target_test, fit, net,
                   iname::String, lname::String, mname::String)
    scatter(target_test', fit',
            xlim=(0, plot_rng), ylim=(0, plot_rng),
            xlabel="real price", ylabel="pred price",
            title=plot_title,
            size=plot_size)

    annotate!([
        (400 + 2, 110 + 2, text("loss layer =  $lname", 10, :black, :left))
        (400 + 2,  90 - 2, text("metric = $mname", 10, :black, :left))
    ])

    name = "$iname-$lname-$mname"

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

#= input(:orig) =#
#= input(:ta) =#
input(:bs)
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

    plot_pred(target_test, fit, net, iname, lname, mname)

    result = DataFrame(
        fit = reshape(fit, length(fit)),
        real = reshape(target_test, length(target_test)))
end
