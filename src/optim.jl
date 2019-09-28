################################################################################
#                                    SGD                                       #
################################################################################

mutable struct SGD
    lr::Float32
    momentum::Float32
    v_t::Dict{Any, Any}   # path => previous velocity
end

function SGD(lr; momentum=0)
    @assert momentum >= 0.0 "momentum must be >= 0"
    SGD(lr, momentum, Dict())
end

Base.show(io::IO, opt::SGD) = print(io, "SGD(lr=$(opt.lr), momentum=$(opt.momentum))")


function make_update!(opt::SGD, path, x, gx)
    v_t0 = get(opt.v_t, path, zero(gx))
    v_t1 = opt.momentum .* v_t0 .+ opt.lr .* gx
    opt.v_t[path] = v_t1
    x_t0 = x
    x_t1 = x_t0 .- v_t1
    return x_t1
end


function Yota.update!(opt::SGD, m, gm)
    # for (path, gx) in gm
    #     v_t0 = get(opt.v_t, path, zero(gx))
    #     v_t1 = opt.momentum .* v_t0 .+ opt.lr .* gx
    #     opt.v_t[path] = v_t1
    #     x_t0 = Yota.getfield_nested(m, path)
    #     x_t1 = x_t0 .- v_t1
    #     Yota.setfield_nested!(m, path, x_t1)
    # end
    for (path, gx) in gm
        x_t0 = Yota.getfield_nested(m, path)
        x_t1 = make_update!(opt, path, x_t0, gx)
        Yota.setfield_nested!(m, path, x_t1)
    end
end


function Yota.update!(opt::SGD, x::AbstractArray, gx)
    # path = ()
    # v_t0 = get(opt.v_t, path, zero(gx))
    # v_t1 = opt.momentum .* v_t0 .+ opt.lr .* gx
    # opt.v_t[path] = v_t1
    # x_t0 = x
    # x_t1 = x_t0 .- v_t1
    x .= make_update!(opt, (), x, gx)
end
