#=
This script shows how a simple MLP net may be used
for regression. It shows how data in memory may be
used for training and evaluation, and how to obtain
the predictions from the trained net.
=#
using MXNet
using Distributions
using PyPlot
using DataFrames

# data generating process
cc = readtable("./ccc.csv")
cc[:mon_σ] *= 1000

data = convert(Array{Float64}, cc[[:price, :contract, :BS, :mon_σ, :T]])'
data_idx = Int(round(size(data)[2] * 0.80))
data_train = data[:, 1:data_idx]
data_test = data[:, data_idx+1:end]

target = convert(Array{Float64}, cc[:close])'
target_idx = Int(round(size(target)[2] * 0.80))
target_train = target[:, 1:target_idx]
target_test = target[:, target_idx+1:end]

# how to set up data providers using data in memory
batchsize = 100 # can adjust this later, but must be defined now for next line
trainprovider = mx.ArrayDataProvider(
    :data => data_train,
    batch_size = batchsize,
    shuffle = true,
    :label => target_train)

evalprovider = mx.ArrayDataProvider(
    :data => data_test,
    batch_size = batchsize,
    shuffle = true,
    :label => target_test)

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
