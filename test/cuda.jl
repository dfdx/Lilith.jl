device = best_available_device()
cpu = CPU()


@testset "cuda: conv" begin
    x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
    g = grad((x, w) -> sum(conv2d(x, w)), x, w)[2]
    d_g = grad((x, w) -> sum(conv2d(x, w)), device(x), device(w))[2]
    @test g[1] ≈ d_g[1]
    @test g[2] ≈ d_g[2]
end

@testset "cuda: pooling" begin
    x = rand(7, 7, 3, 10);
    g = grad(x -> sum(maxpool2d(x, 2)), x)[2]
    d_g = grad(x -> sum(maxpool2d(x, 2)), device(x))[2]
    @test g[1] ≈ d_g[1]
end


@testset "cuda: activations" begin
    x = rand(5, 5);
    # x = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    d_x = device(x);

    @test grad(x -> sum(logistic.(x)), x)[2][1] ≈ grad(x -> sum(logistic.(x)), d_x)[2][1]
    @test grad(x -> sum(softplus.(x)), x)[2][1] ≈ grad(x -> sum(softplus.(x)), d_x)[2][1]
    @test grad(x -> sum(softsign.(x)), x)[2][1] ≈ grad(x -> sum(softsign.(x)), d_x)[2][1]
    @test grad(x -> sum(relu.(x)), x)[2][1] ≈ grad(x -> sum(relu.(x)), d_x)[2][1]
    @test grad(x -> sum(leakyrelu.(x, 0.01)), x)[2][1] ≈ grad(x -> sum(leakyrelu.(x, 0.01)), d_x)[2][1]
    # ELU on CUDA results in scalar operations warning followed by segfault, disabling it for now
    # @test grad(x -> sum(elu.(x, 1.0)), x)[2][1] ≈ grad(x -> sum(elu.(x, 1.0)), d_x)[2][1]

    g = grad(x -> sum(softmax(x)), x)[2][1]
    d_g = grad(x -> sum(softmax(x)), d_x)[2][1]
    @test isapprox(g, cpu(d_g), rtol = 1e-5, atol = 1e-5)
        
    @test grad(x -> sum(logsoftmax(x)), x)[2][1] ≈ grad(x -> sum(logsoftmax(x)), d_x)[2][1]
end


@testset "cuda: losses" begin    
    x = rand(5, 4); x = log.(x ./ sum(x; dims=1)); c = [3, 2, 1, 4, 5]
    d_x = device(x); d_c = device(c)
    @test grad((x, c) -> nllloss(x, c), x, c)[2][1] ≈ grad((x, c) -> nllloss(x, c), d_x, d_c)[2][1]
   
    x = rand(5, 4); c = [3, 2, 1, 4, 5]
    d_x = device(x); d_c = device(c)
    @test (grad((x, c) -> crossentropyloss(x, c), x, c)[2][1] ≈
           grad((x, c) -> crossentropyloss(x, c), d_x, d_c)[2][1])

    x = rand(5, 4); x_target = rand(5, 4)
    d_x = device(x); d_x_target = device(x_target)
    @test (grad((x, x_target) -> mseloss(x, x_target), x, x_target)[2][1] ≈
           grad((x, x_target) -> mseloss(x, x_target), d_x, d_x_target)[2][1])
end


@testset "cuda: optim" begin

    # not every parameter update will lead to descreased loss
    # but at least we can check that parameters are actually changed
    m = MyModel(Linear(5, 4)) |> device; x = rand(5, 10) |> device;
    old_m = deepcopy(m); old_x = deepcopy(x)
    _, g = grad(my_model_loss, m, x)
       
    # SGD
    update!(SGD(0.1; momentum=0.5), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(SGD(0.1; momentum=0.5), x, g[2])
    @test old_x != x

    # Adam
    update!(Adam(), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(Adam(), x, g[2])
    @test old_x != x

end
