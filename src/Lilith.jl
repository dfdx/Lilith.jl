module Lilith

export
    grad,
    # conv
    conv2d,
    maxpool2d,
    # activations
    logistic,
    sigmoid,
    softplus,
    relu,
    leakyrelu,
    softmax,
    logsoftmax,
    # losses
    nllloss,
    crossentropyloss,
    mseloss,
    NLLLoss,
    CrossEntropyLoss,
    MSELoss,
    # general layers
    Linear,
    Sequential,
    Conv2d,
    # training
    fit!


include("core.jl")

end # module
