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
    softsign,
    logsigmoid,
    relu,
    leakyrelu,
    # elu,
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
    # optim
    update!,
    SGD,
    RMSprop,
    Adam,
    # training
    fit!,
    # device API (reexport from Yota)
    best_available_device,
    to_device,
    CPU,
    GPU
    # metrics (not exportrd by default)




include("core.jl")

end # module
