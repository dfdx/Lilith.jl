using Yota
using Distributions
using MLDataUtils
import NNlib


include("conv.jl")
include("activations.jl")
include("losses.jl")
include("layers.jl")
include("optim.jl")
include("fit.jl")


function __init__()
    register_conv_derivs()
    register_activation_derivs()
    register_loss_derivs()
end



# TODO: datasets, data shuffling, train_test_split, etc. -- MLDataUtils?
# TODO: predict(m, X) == m(X)?
# TODO: image folder dataset (LilithVision?)
