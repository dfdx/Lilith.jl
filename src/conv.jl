import NNlib: DenseConvDims, PoolDims

## conv2d

function conv2d(x, w; stride=1, padding=0, dilation=1)
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    return NNlib.conv(x, w, cdims)
end

function ∇conv2d_w(dy, x, w; stride=1, padding=0, dilation=1)
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    return NNlib.∇conv_filter(x, dy, cdims)
end

function ∇conv2d_x(dy, x, w; stride=1, padding=0, dilation=1)
    cdims = DenseConvDims(x, w; stride=stride, padding=padding, dilation=dilation)
    return NNlib.∇conv_data(dy, w, cdims)
end


## maxpool2d

function maxpool2d(x, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    return NNlib.maxpool(x, pdims)
end

function ∇maxpool2d_x(dy, y, x, kernel_size; stride=kernel_size, padding=0, dilation=1)
    pdims = PoolDims(x, kernel_size; stride=stride, padding=padding, dilation=dilation)
    return NNlib.∇maxpool(dy, y, x, pdims)
end



function register_conv_derivs()
    @diffrule_kw conv2d(x, w) w ∇conv2d_w(dy, x, w)
    @diffrule_kw conv2d(x, w) x ∇conv2d_x(dy, x, w)

    @diffrule_kw maxpool2d(x, _k) x ∇maxpool2d_x(dy, y, x, _k)
    @nodiff maxpool2d(x, _k) _k
end
