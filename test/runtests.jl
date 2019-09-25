using Lilith
import Lilith: accuracy, recall, precision, confusion_matrix
using Random
using Test
import CUDAapi.has_cuda


include("gradcheck.jl")
include("conv.jl")
include("activations.jl")
include("layers.jl")
include("metrics.jl")

if has_cuda()
    include("cuda.jl")
end
