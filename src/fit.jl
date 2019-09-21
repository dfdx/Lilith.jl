function partial_fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn; lr=1e-3, batch_size=100)
    epoch_loss = 0
    for (i, (x, y)) in enumerate(eachbatch((X, Y), size=batch_size))
        loss, g = grad((m, x, y) -> loss_fn(m(x), y), m, x, y)
        update!(m, g[1], (x, gx) -> x .- lr * gx)
        epoch_loss += loss
        println("iter $i: loss=$loss")
    end
    return epoch_loss
end


function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, lr=1e-3, report_every=1)
    for epoch in 1:n_epochs
        time = @elapsed epoch_loss = partial_fit!(m, X, Y, loss_fn, batch_size=batch_size)
        if epoch % report_every == 0
            println("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
        end
    end
    return m
end
