function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                collect(grad(f, xs...)[2]), rtol = 1e-5, atol = 1e-5))


function check_convergence(f, args...; p=1, verbose=false, lr=1e-4, epochs=100)
    opt = Adam(lr=1e-4)
    # run once to overcome possible initialization issues
    # _, g = grad(f, args...)
    # update!(opt, args[p], g[p])
    # calculate loss at the beginning
    L0 = f(args...)
    L = L0
    for i=1:epochs
        L, g = grad(f, args...)
        if verbose
            println(L)
        end
        update!(opt, args[p], g[p])
    end
    return L < L0
end
