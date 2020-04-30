@testset "conv" begin
    x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
    @test gradcheck((x, w) -> sum(conv2d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv2d(x, w; padding=1, stride=2)), x, w; tol=1e-4)
end

@testset "pooling" begin
    x = rand(7, 7, 3, 10);
    @test gradcheck(x -> sum(maxpool2d(x, 2)), x)
    @test gradcheck(x -> sum(maxpool2d(x, 2; stride=2)), x)
end
