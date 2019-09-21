# Translated from PyTorch version, see cnn_minst.py

using Lilith
using MLDatasets

# include("../src/core.jl")
# __init__()

mutable struct Net
    conv1::Conv2d
    conv2::Conv2d
    fc1::Linear
    fc2::Linear
end


Net() = Net(
    Conv2d(1, 20, 5),
    Conv2d(20, 50, 5),
    Linear(4 * 4 * 50, 500),
    Linear(500, 10)
)


function (m::Net)(x::AbstractArray)
    x = maxpool2d(relu.(m.conv1(x)), (2, 2))
    x = maxpool2d(relu.(m.conv2(x)), (2, 2))
    # prod(size(x)[1:3])
    x = reshape(x, 4*4*50, :)
    x = relu.(m.fc1(x))
    x = logsoftmax(m.fc2(x))
    return x
end


function main()
    m = Net()
    X, Y = MNIST.traindata();
    X = convert(Array{Float64}, reshape(X, 28, 28, 1, :));
    Y .+= 1   # replace class label like "0" with its position like "1"
    loss_fn = NLLLoss()
    @time partial_fit!(m, X, Y, loss_fn; lr=1e-2, batch_size=64)
end
