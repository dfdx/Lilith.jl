using Lilith


mutable struct BasicBlock
    conv1::Conv2d       # 3x3
    bn1::BatchNorm2d
    conv2::Conv2d       # 3x3
    bn2::BatchNorm2d
    downsample::Union{Sequential, Nothing}
end


conv3x3(inplanes, outplanes; stride=1, dilation=1) =
    Conv2d(inplanes, outplanes, 3; stride=stride, padding=dilation, dilation=dilation)  # TODO: bias = false

conv1x1(inplanes, outplanes; stride=1) =
    Conv2d(inplanes, outplanes, 1; stride=stride)  # TODO: bias = false


function BasicBlock(inplanes::Int, planes::Int; downsample=nothing, stride::Int=1)
    return BasicBlock(
        conv3x3(inplanes, planes; stride=stride),
        BatchNorm2d(planes),
        conv3x3(planes, planes),
        BatchNorm2d(planes),
        downsample
    )
end


function (m::BasicBlock)(x::AbstractArray{T, 4}) where T
    identity = x
    out = m.conv1(x)
    out = m.bn1(out)
    out = relu.(out)
    out = m.conv2(out)
    out = m.bn2(out)
    if m.downsample != nothing
        identity = m.downsample(x)
    end
    out = out .+ identity
    out = relu.(out)
    return out
end



function main()
    m = BasicBlock(10, 10)
    x = rand(12, 12, 10, 4)
    m(x)

    _, g = grad((m, x) -> sum(m(x)), m, x)
end
