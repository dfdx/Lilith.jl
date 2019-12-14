
## Supervised learning with input X and output Y

# function partial_fit!(m, X::AbstractArray, Y::AbstractArray, full_loss_fn;
#                       opt=SGD(1e-3), batch_size=100, device=CPU())
#     epoch_loss = 0
#     for (i, (x, y)) in enumerate(eachbatch((X, Y), size=batch_size))
#         x = to_device(device, copy(x))
#         y = to_device(device, copy(y))
#         loss, g = grad(full_loss_fn, m, x, y)
#         update!(opt, m, g[1])
#     end
#     return epoch_loss / size(X)[end]
# end


# function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
#               n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
#     full_loss_fn = (m, x, y) -> loss_fn(m(x), y)
#     for epoch in 1:n_epochs
#         time = @elapsed epoch_loss =
#             partial_fit!(m, X, Y, full_loss_fn, batch_size=batch_size, device=device)
#         if epoch % report_every == 0
#             println("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
#         end
#     end
#     return m
# end

function fit!(m, X::AbstractArray, Y::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    f = (m, x, y) -> loss_fn(m(x), y)
    num_batches = size(X)[end] // batch_size
    for epoch in 1:n_epochs
        epoch_loss = 0
        for (i, (x, y)) in enumerate(eachbatch((X, Y), size=batch_size))
            x = to_device(device, copy(x))
            y = to_device(device, copy(y))
            loss, g = grad(f, m, x, y)
            update!(opt, m, g[1])
            epoch_loss += loss
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_loss=$(epoch_loss / num_batches)")
        end
    end
    return m
end


## Unsupervised learning with only X

function partial_fit!(m, X::AbstractArray, loss_fn;
                      opt=SGD(1e-3), batch_size=100, device=CPU())
    epoch_loss = 0
    f = (m, x) -> loss_fn(m(x))
    for (i, x) in enumerate(eachbatch(X, size=batch_size))
        x = to_device(device, copy(x))
        loss, g = grad(f, m, x)
        update!(opt, m, g[1])
        epoch_loss += loss
        println("iter $i: loss=$loss")
    end
    return epoch_loss
end


function fit!(m, X::AbstractArray, loss_fn;
              n_epochs=10, batch_size=100, opt=SGD(1e-3), device=CPU(), report_every=1)
    for epoch in 1:n_epochs
        time = @elapsed epoch_loss = partial_fit!(m, X, loss_fn, batch_size=batch_size, device=device)
        if epoch % report_every == 0
            @info("Epoch $epoch: avg_cost=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$time")
        end
    end
    return m
end
