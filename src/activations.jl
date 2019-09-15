logistic(x) = one(x) / (one(x) + exp(-x))
∇logistic(dy, x) = logistic(x) * (one(x) - logistic(x)) * dy
const sigmoid = logistic

softplus(x) = log(exp(x) + one(x))
∇softplus(dy, x) = logistic(x) * dy

relu(x::Number) = max(zero(x), x)
∇relu(dy::Real, y::Real) = ifelse(y > 0, dy, zero(y))

leakyrelu(x::T, alpha) where T <: Real = max(T(alpha) * x, x)
∇leakyrelu(dy::Real, y::T, alpha) where T <: Real = ifelse(y > 0, dy, T(alpha))

softmax(x::AbstractArray) = NNlib.softmax(x)
∇softmax(dy, x) = NNlib.∇softmax(dy, x)

logsoftmax(x::AbstractArray) = NNlib.logsoftmax(x)
∇logsoftmax(dy, x) = NNlib.∇logsoftmax(dy, x)


function register_activation_derivs()
    @diffrule logistic(x::Number) x ∇logistic(dy, x)
    @diffrule softplus(x::Number) x ∇softplus(dy, x)
    @diffrule relu(x::Number) x ∇relu(dy, y)
    @diffrule leakyrelu(x::Number, _alpha::Real) x ∇leakyrelu(dy, x, _alpha)
    @nodiff leakyrelu(x::Number, _alpha::Real) _alpha
    
    @diffrule softmax(x) x ∇softmax(dy, x)
    @diffrule logsoftmax(x) x ∇logsoftmax(dy, x)
end

# relu(x) = NNlib.relu(x)
# ∇relu(dy, x) = 
