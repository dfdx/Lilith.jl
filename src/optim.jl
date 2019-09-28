abstract type Optimizer end


function Yota.update!(opt::Optimizer, m, gm)
    for (path, gx) in gm
        x_t0 = Yota.getfield_nested(m, path)
        x_t1 = make_update!(opt, path, x_t0, gx)
        Yota.setfield_nested!(m, path, x_t1)
    end
    opt.t += 1
end


function Yota.update!(opt::Optimizer, x::AbstractArray, gx)
    x .= make_update!(opt, (), x, gx)
    opt.t += 1
end


################################################################################
#                                    SGD                                       #
################################################################################

mutable struct SGD <: Optimizer
    t::Int
    lr::Float32
    momentum::Float32
    v_t::Dict{Any, Any}   # path => previous velocity
end

function SGD(lr; momentum=0)
    @assert momentum >= 0.0 "momentum must be >= 0"
    SGD(0, lr, momentum, Dict())
end

Base.show(io::IO, opt::SGD) = print(io, "SGD(lr=$(opt.lr), momentum=$(opt.momentum))")


function make_update!(opt::SGD, path, x, gx)
    v_t = get(opt.v_t, path, zero(gx))
    v_t1 = opt.momentum .* v_t .+ opt.lr .* gx
    opt.v_t[path] = v_t1
    x_t = x
    x_t1 = x_t .- v_t1
    return x_t1
end


################################################################################
#                                   Adam                                       #
################################################################################


mutable struct Adam <: Optimizer
    t::Int32
    eps::Float32
    lr::Float32
    beta1::Float32
    beta2::Float32
    m_t::Dict{Any, Any}
    v_t::Dict{Any, Any}
end


function Adam(;lr::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, eps::Real=10e-8)
    @assert lr > 0.0 "lr must be greater than 0"
    @assert beta1 > 0.0 "beta1 must be greater than 0"
    @assert beta2 > 0.0 "beta2 must be greater than 0"
    @assert eps > 0.0 "eps must be greater than 0"
    Adam(0, eps, lr, beta1, beta2, Dict(), Dict())
end


function make_update!(opt::Adam, path, x, gx)
    # resize biased moment estimates if first iteration
    m_t = get(opt.m_t, path, zero(gx))   # TODO: check size of the minibatch
    v_t = get(opt.v_t, path, zero(gx))
    # update biased first moment estimate
    m_t = opt.beta1 * m_t + (1f0 - opt.beta1) * gx
    opt.m_t[path] = m_t
    # update biased second raw moment estimate
    v_t = opt.beta2 * v_t + (1f0 - opt.beta2) * (gx .^ 2)
    opt.v_t[path] = v_t
    # compute bias corrected first moment estimate
    m̂_t = m_t / (1f0 - opt.beta1^opt.t)
    # compute bias corrected second raw moment estimate
    v̂_t = v_t / (1f0 - opt.beta2^opt.t)
    # apply update
    x_t1 = opt.lr * m̂_t ./ sqrt.(v̂_t .+ opt.eps)    
    return x_t1
end

# TODO: add tests on CPU
# TODO: add tests on GPU
