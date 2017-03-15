using MXNet
using Distributions
using Plots
using DataFrames

pyplot()

# data generating process
function load(col::Array{Symbol})
    global cc = readtable("./ccc.csv")
    cc[:mon_σ] *= 1000

    global data = convert(Array{Float64}, cc[col])'
    global data_idx = Int(round(size(data)[2] * 0.80))
    global data_train = data[:, 1:data_idx]
    global data_test = data[:, data_idx+1:end]

    global target = convert(Array{Float64}, cc[:close])'
    global target_idx = Int(round(size(target)[2] * 0.80))
    global target_train = target[:, 1:target_idx]
    global target_test = target[:, target_idx+1:end]
end

function input_1()
    load([:price, :contract, :mon_σ, :T])

    batchsize = 64

    # WTF: if we enable ``suffle``, the optimizer will fail to converge
    trainprovider = mx.ArrayDataProvider(
        :data => data_train,
        batch_size = batchsize,
        :label => target_train)

    evalprovider = mx.ArrayDataProvider(
        :data => data_test,
        batch_size = batchsize,
        :label => target_test)

    plotprovider = mx.ArrayDataProvider(
        :data => data_test,
        :label => target_test)

    trainprovider, evalprovider, plotprovider
end

function input_2()  # with TA: MACD
    load([:price, :contract, :mon_σ, :T])

    batchsize = 64

    trainprovider = mx.ArrayDataProvider(
        :data => data_train,
        batch_size = batchsize,
        :label => target_train)

    evalprovider = mx.ArrayDataProvider(
        :data => data_test,
        batch_size = batchsize,
        :label => target_test)

    plotprovider = mx.ArrayDataProvider(
        :data => data_test,
        :label => target_test)

    trainprovider, evalprovider, plotprovider
end

trainprovider, evalprovider, plotprovider = input_1()

# create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
# or add/remove a layer
#= data_ = mx.Variable(:data) =#
label = mx.Variable(:label)
net = @mx.chain     mx.Variable(:data) =>
                    mx.FullyConnected(num_hidden = 5)  =>
                    mx.Activation(act_type = :relu) =>
                    mx.FullyConnected(num_hidden = 5)  =>
                    mx.Activation(act_type = :relu) =>
                    mx.FullyConnected(num_hidden = 1)  =>
                    mx.LinearRegressionOutput(label)

# final model definition, don't change, except if using gpu
model = mx.FeedForward(net, context=mx.cpu())

# set up the optimizer: select one, explore parameters, if desired
#= optimizer = mx.SGD(lr=0.01, momentum=0.9, weight_decay=0.00001) =#
optimizer = mx.ADAM()

# train, reporting loss for training and evaluation sets
# initial training with small batch size, to get to a good neighborhood
mx.fit(
    model, optimizer, initializer=mx.NormalInitializer(0.0,0.1),
    eval_metric=mx.MSE(), trainprovider, eval_data=evalprovider, n_epoch = 40)
# more training with the full sample
mx.fit(
    model, optimizer, eval_metric=mx.MSE(),
    trainprovider, eval_data=evalprovider, n_epoch = 40)

# obtain predictions
plotprovider = mx.ArrayDataProvider(
    :data => data_test,
    :label => target_test)
fit = mx.predict(model, plotprovider)
plot(target_test', fit', ".")
xlabel("real price")
ylabel("predicted price")
title("outputs: true versus predicted. 45º line is what we hope for")
savefig("test.png")
