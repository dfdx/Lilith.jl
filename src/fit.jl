# my_loss(m, x, y) = NLLLoss()(m(x), y)


function partial_fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
                      opt=SGD(1e-3), batch_size=100, device=CPU())
    epoch_loss = 0
    f = (m, x, y) -> loss_fn(m(x), y)
    for (i, (x, y)) in enumerate(eachbatch((X, Y), size=batch_size))
        x = to_device(device, copy(x))
        y = to_device(device, copy(y))
        loss, g = grad(f, m, x, y)
        # loss, g = grad(f, m, x)
        update!(opt, m, g[1])
        # epoch_loss += loss
        println("iter $i: loss=$loss")
    end
    return epoch_loss
end


function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    for epoch in 1:n_epochs
        time = @elapsed epoch_loss = partial_fit!(m, X, Y, loss_fn, batch_size=batch_size, device=device)
        if epoch % report_every == 0
            println("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
        end
    end
    return m
end
