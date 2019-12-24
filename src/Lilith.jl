module Lilith

export
    grad,
    @diffrule,
    @diffrile_kw,
    @nodiff,
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
    # conv layers    
    Conv2d,
    # RNN layers
    RNN,
    LSTM,
    GRU,
    init_hidden,
    # optim
    update!,
    SGD,
    RMSprop,
    Adam,
    # training
    fit!,
    trainmode!,
    testmode!,
    # device API (reexport from Yota)
    best_available_device,
    to_device,
    CPU,
    GPU
    # metrics (not exported by default)



include("core.jl")

end # module
