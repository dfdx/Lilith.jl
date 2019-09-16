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
    NLLLoss,
    CrossEntropyLoss,
    # general layers
    Linear,
    Sequential,
    Conv2d    

include("core.jl")

end # module
