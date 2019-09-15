using Yota
import NNlib
# include("../../Yota/src/core.jl")

include("conv.jl")
include("layers.jl")
include("activations.jl")


function __init__()
    register_conv_derivs()
    register_activation_derivs()
end
