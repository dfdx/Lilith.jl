## recurrent networks

## Vanilla RNN

mutable struct RNNCell
  W_ih::AbstractMatrix
  W_hh::AbstractMatrix
  b_ih::AbstractVector
  b_hh::AbstractVector
end


function RNNCell(inp::Integer, hid::Integer)
    k_sqrt = sqrt(1 / hid)
    d = Uniform(-k_sqrt, k_sqrt)
    return RNNCell(rand(d, hid, inp), rand(d, hid, hid), rand(d, hid), rand(hid))
end

Base.show(io::IO, m::RNNCell) = print(io, "RNNCell($(size(m.W_ih, 2)) => $(size(m.W_ih, 1)))")


# input should be of size (inp_size, batch)
function forward(m::RNNCell, x::AbstractMatrix, h::AbstractMatrix)
    inp_v = m.W_ih * x .+ m.b_ih
    hid_v = m.W_hh * h .+ m.b_hh
    h_ = tanh.(inp_v .+ hid_v)
    return h_
end

(m::RNNCell)(x::AbstractMatrix, h::AbstractMatrix) = forward(m, x, h)


mutable struct RNN
    cell::RNNCell
end


# input should be of size (inp_size, batch, seq_len)
# TODO: for multilayer, multidirectional nets h should be AbstractArray{T, 3} instead
function forward(m::RNN, x_seq::AbstractArray{T, 3}, h::AbstractArray{T, 2}) where T
    inp_size, batch, seq_len = size(x_seq)
    hid_size = length(m.cell.b_hh)
    h_all = zeros(1 * hid_size, batch, 0)
    h_all_sz = (size(h)..., 1)
    cell = m.cell
    for i=1:size(x_seq, 3)
        x = x_seq[:, :, i]
        h = forward(cell, x, h)        
        h_all = cat(h_all, reshape(h, h_all_sz); dims=3)
    end
    # TODO 2: make multilayer
    # TODO 3: make biderectional
    return h_all, h
end

(m::RNN)(x::AbstractArray{T,3}, h::AbstractArray{T,2}) where T = forward(m, x, h)
