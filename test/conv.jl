@testset "conv" begin    
    
    # 1D
    @info("Hello! Can I just print here a few lines while convolutions are being tested?")
    x = rand(7, 3, 10); w = rand(3, 3, 1)
    @test gradcheck((x, w) -> sum(conv1d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv1d(x, w; padding=1, stride=2)), x, w; tol=1e-4)
    
    # 2D
    @info("Just to keep you updated...")
    x = rand(7, 7, 3, 10); w = rand(3, 3, 3, 1)
    @test gradcheck((x, w) -> sum(conv2d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv2d(x, w; padding=1, stride=2)), x, w; tol=1e-4)
    
    # 3D
    @info("Because, you know, testing may take time")
    x = rand(7, 7, 7, 3, 10); w = rand(3, 3, 3, 3, 1)
    @test gradcheck((x, w) -> sum(conv3d(x, w)), x, w; tol=1e-4)
    @test gradcheck((x, w) -> sum(conv3d(x, w; padding=1, stride=2)), x, w; tol=1e-4)

    @info("Ok, we are done with convolutions, I won't bother you anymore!")
end

@testset "pooling" begin
    x = rand(7, 7, 3, 10);
    @test gradcheck(x -> sum(maxpool2d(x, 2)), x)
    @test gradcheck(x -> sum(maxpool2d(x, 2; stride=2)), x)
end
