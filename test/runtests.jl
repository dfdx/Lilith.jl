using Lilith
using Random
using Test
import CUDAapi.has_cuda


include("gradcheck.jl")
include("conv.jl")
include("activations.jl")
include("layers.jl")

if has_cuda()
    include("cuda.jl")
end
