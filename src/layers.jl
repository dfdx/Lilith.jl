################################################################################
#                               Linear                                         #
################################################################################

mutable struct Linear{T}
    W::AbstractMatrix{T}
    b::AbstractVector{T}
end

Linear(in_features::Int, out_features::Int) = Linear(randn(out_features, in_features), randn(out_features))

function Base.show(io::IO, l::Linear)
    o, i = size(l.W)
    print(io, "Linear($i=>$o)")
end

(l::Linear)(x::Union{AbstractVector{T}, AbstractMatrix{T}}) where T = l.W * x .+ l.b


################################################################################
#                               Sequential                                     #
################################################################################


# TODO: stub, we can't differentiate through it right now since Yota.field_paths doesn't track
# indices of seq
mutable struct Sequential
    seq::Tuple
end

Sequential(args...) = Sequential(args)

function Base.show(io::IO, s::Sequential)
    println(io, "Sequential(")
    for m in s.seq
        println(io, "  $m,")
    end
    print(io, ")")
end


function (s::Sequential)(x)
    for m in s.seq
        x = m(x)
    end
    return x
end


################################################################################
#                           Convolutions                                       #
################################################################################

mutable struct Conv2d{T}
    W::AbstractArray{T,4}
    stride
    padding
    dilation
end

function Conv2d(in_channels::Int, out_channels::Int, kernel_size::Union{Int, NTuple{2, Int}};
                stride=1, padding=0, dilation=1)
    kernel_tuple = kernel_size isa Tuple ? kernel_size : (kernel_size, kernel_size)
    W = randn(kernel_tuple..., in_channels, out_channels)
    return Conv2d(W, stride, padding, dilation)
end

function Base.show(io::IO, c::Conv2d)
    k1, k2, i, o = size(c.W)
    print(io, "Conv2d($(k1)x$(k2), $i=>$o)")
end

(c::Conv2d)(x::AbstractArray{T,4}) where T =
    conv2d(x, c.W; stride=c.stride, padding=c.padding, dilation=c.dilation)


