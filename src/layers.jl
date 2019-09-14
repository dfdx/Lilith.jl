mutable struct Conv2d{T}
    w::AbstractArray{T,4}
    stride
    padding
    dilation
end

function Base.show(io::IO, c::Conv2d)
    k1, k2, i, o = size(c.w)
    print(io, "Conv2d($(k1)x$(k2), $i=>$o)")
end


function Conv2d(in_channels::Int, out_channels::Int, kernel_size::Union{Int, NTuple{2, Int}};
                stride=1, padding=0, dilation=1)
    kernel_tuple = kernel_size isa Tuple ? kernel_size : (kernel_size, kernel_size)
    w = randn(kernel_tuple..., in_channels, out_channels)
    return Conv2d(w, stride, padding, dilation)
end

(c::Conv2d)(x::AbstractArray{T,4}) where T =
    conv2d(x, c.w; stride=c.stride, padding=c.padding, dilation=c.dilation)
