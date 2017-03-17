using MXNet
using Distributions
using Plots
using DataFrames

pyplot()

# data generating process
function load(col::Array{Symbol})
    global cc = readtable("./ccc_macd.csv")
    cc[:mon_σ] *= 100

    global data = convert(Array{Float64}, cc[col])'
    global data_idx = Int(round(size(data)[2] * 0.80))
    global data_train = data[:, 1:data_idx]
    global data_test = data[:, data_idx+1:end]

    global target = convert(Array{Float64}, cc[:close])'
    global target_idx = Int(round(size(target)[2] * 0.80))
    global target_train = target[:, 1:target_idx]
    global target_test = target[:, target_idx+1:end]
end

function plot_bs()
    load()
    scatter(Array(c[:close]), Array(c[:BS]),
            xlim=(0, 600), ylim=(0, 600),
            xlabel="real price", ylabel="pred price")
    savefig("bs.png")
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

function plot_pred(target_test, fit)
    scatter(target_test', fit',
            xlim = (0, 600), ylim = (0, 600),
            xlabel = "real price", ylabel = "pred price",
            title = plot_title)
    savefig("out.png")
    write("net.dot", mx.to_graphviz(net))
    run(`dot -Tpng -o net.png net.dot`)
end


input(:orig)
trainprovider, evalprovider, plotprovider = get_provider()

# create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
# or add/remove a layer
#= data_ = mx.Variable(:data) =#
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
              mx.FullyConnected(num_hidden = 2)  =>
              mx.Activation(act_type = :relu) =>
              mx.FullyConnected(num_hidden = 1)  =>
              mx.LinearRegressionOutput(label)

# net for quick test
net_qtest =
    @mx.chain mx.Variable(:data) =>
              mx.FullyConnected(num_hidden = 10)  =>
              mx.Activation(act_type = :relu) =>
              mx.FullyConnected(num_hidden = 5)  =>
              mx.Activation(act_type = :relu) =>
              mx.FullyConnected(num_hidden = 1)  =>
              mx.MAERegressionOutput(label)

# final model definition, don't change, except if using gpu
cpus = [mx.cpu(i) for i in 0:3]
gpu = mx.gpu()
model = mx.FeedForward(net, context=gpu)

# set up the optimizer: select one, explore parameters, if desired
#= optimizer = mx.SGD(lr=0.01, momentum=0.9, weight_decay=0.00001) =#
optimizer = mx.ADAM()

# train, reporting loss for training and evaluation sets
epoch = 100
# initial training with small batch size, to get to a good neighborhood
mx.fit(
    model, optimizer,
    initializer = mx.NormalInitializer(0.0,0.1),
    eval_metric = mx.NMSE(),
    trainprovider,
    eval_data = evalprovider,
    n_epoch = epoch)
# more training with the full sample
mx.fit(
    model, optimizer,
    eval_metric = mx.NMSE(),
    trainprovider,
    eval_data = evalprovider,
    n_epoch = epoch)

fit = mx.predict(model, plotprovider)

function plot_pred(target_test, fit)
    scatter(target_test', fit',
            xlim = (0, 600), ylim = (0, 600),
            xlabel = "real price", ylabel = "pred price",
            title = plot_title)
    savefig("out.png")
    write("net.dot", mx.to_graphviz(net))
    run(`dot -Tpng -o net.png net.dot`)
end

plot_pred(target_test, fit)

result = DataFrame(
    fit = reshape(fit, length(fit)),
    real = reshape(target_test, length(target_test)))
