using Yota
using Statistics
using LinearAlgebra
import NNlib

# include("../../Yota/src/core.jl")

include("conv.jl")
include("layers.jl")
include("activations.jl")
include("losses.jl")


function __init__()
    register_conv_derivs()
    register_activation_derivs()
    register_loss_derivs()
end
