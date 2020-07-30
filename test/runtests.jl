using Lilith
import Lilith: accuracy, recall, precision, confusion_matrix
import Lilith: RNNCell, LSTMCell, GRUCell, rnn_forward, lstm_forward, gru_forward
import Lilith: ∇batchnorm2d
using Random
using Test
import CUDAapi.has_cuda


Random.seed!(108);

include("gradcheck.jl")
include("conv.jl")
include("rnn.jl")
include("activations.jl")
include("layers.jl")
# include("losses.jl")  -- ignored since nllloss() doesn't provide derivative w.r.t. 2nd argument
include("optim.jl")
include("metrics.jl")

if has_cuda()
    include("cuda.jl")
end
