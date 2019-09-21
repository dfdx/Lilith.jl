## Code of all optimizers is heavily based on https://github.com/jacobcvt12/GradDescent.jl/
## Main reasons for modifications:
## 1. Better workflow on GPU
## 2. Single code style, including variable naming
## 3. Similarity to PyTorch API

"""
**SGD constructor**
```julia
    SGD(; eta::Real=0.01, gamma::Real=0.9)
```
Algorithm :
```math
\\begin{align*}
v_t =& \\gamma v_{t-1} + \\eta g_t\\\\
\\Delta x_t =& v_t
\\end{align*}
```
"""
mutable struct SGD
    opt_type::String
    t::Int64
    eta::Float64
    gamma::Float64
    v_t::AbstractArray
end

## Note the algorithm seems flawed, \eta should be 1-\β and a supplementary global learning rate would be nice

function SGD(; eta::Real=0.01, gamma::Real=0.9)
    @assert eta > 0.0 "eta must be greater than 0"
    @assert gamma > 0.0 "gamma must be greater than 0"

    SGD("SGD", 0, eta, gamma, [])
end

params(opt::SGD) = "eta=$(opt.eta), gamma=$(opt.gamma)"

function update(opt::SGD, g_t::AbstractArray{T}) where {T<:Real}
    # resize squares of gradients
    if opt.t == 0
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    opt.v_t = opt.gamma * opt.v_t + opt.eta * g_t

    return opt.v_t
end
