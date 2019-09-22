using CuArrays

# CuArrays doesn't provide a dedicated version of conv() for CuArrays, and so call to
# NNlib.conv(CuArray(), CuArray()) leads to CPU-based algorithm on GPU arrays, which is terribly slow.
# Here we implement our wrapper conv2d() for CuArrays via in-place version CuArrays.conv!().
# The same applies to maxpool() which we implement using maxpool!().
# Note that later we will have in-place rules and rewrite them on the tape, but we still
# need to make all operations works without any reference to gradients.

## conv2d

function conv2d(x::CuArray{T,4}, w::CuArray{T,4}; stride=1, padding=0, dilation=1) where T
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    y = similar(x, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, 4))
    return CuArrays.conv!(y, x, w, cdims)
end

function ∇conv2d_w(dy::CuArray{T,4}, x::CuArray{T,4}, w::CuArray{T,4}; stride=1, padding=0, dilation=1) where T
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    dw = similar(w)
    return CuArrays.∇conv_filter!(dw, x, dy, cdims)
end

function ∇conv2d_x(dy, x, w; stride=1, padding=0, dilation=1)
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    dx = similar(x)
    return CuArrays.∇conv_data!(dx, dy, w, cdims)
end


## maxpool2d

function maxpool2d(x::CuArray, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    y = similar(x, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, 4))
    return CuArrays.maxpool!(y, x, pdims)
end

function ∇maxpool2d_x(dy::CuArray, y::CuArray, x::CuArray, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    dx = similar(x)
    return NNlib.∇maxpool!(dx, dy, y, x, pdims)
end
