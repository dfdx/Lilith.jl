import NNlib
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




# foo(x) = 42

function __init__()
    @diffrule_kw conv2d(x, w) w ∇conv2d_w(dy, x, w)
    @diffrule_kw conv2d(x, w) x ∇conv2d_x(dy, x, w)
    # println("__init__(): conv2d in PRIMITIVES? $(conv2d in Yota.PRIMITIVES)")

    @diffrule_kw maxpool2d(x, _k) x ∇maxpool2d_x(dy, y, x, _k)
    @nodiff maxpool2d(x, _k) _k
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

# println("Adding pool2d")




# function main()
#     x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
#     y = conv2d(x, w; stride=2)

#     y = maxpool2d(x, 1; padding=1)
#     dy = y
#     ∇maxpool2d_x(dy, y, x, 1; padding=1)


#     NNlib.maxpool(y, PoolDims(y, 1))


#     _, g = grad((x, w) -> sum(conv2d(x, w)), x, w)

#     r = ngradient((x, w) -> sum(conv2d(x, w)), x, w)

#     gradcheck(x -> sum(maxpool2d(x, 1; stride=2)), x)

#     trace((x, w) -> sum(conv2d(x, w; stride=2)), x, w)
#     grad((x, w) -> sum(conv2d(x, w; stride=2)), x, w)
# end



# function ngradient(f, xs::AbstractArray...)
#   grads = zero.(xs)
#   for (x, Δ) in zip(xs, grads), i in 1:length(x)
#     δ = sqrt(eps())
#     tmp = x[i]
#     x[i] = tmp - δ/2
#     y1 = f(xs...)
#     x[i] = tmp + δ/2
#     y2 = f(xs...)
#     x[i] = tmp
#     Δ[i] = (y2-y1)/δ
#   end
#   return grads
# end

# gradcheck(f, xs...) =
#   all(isapprox.(ngradient(f, xs...),
#                 collect(grad(f, xs...)[2]), rtol = 1e-5, atol = 1e-5))
