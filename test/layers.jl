@testset "layers" begin

    x = rand(7, 7, 3, 10); c = Conv2d(3, 5, 3)
    @test grad((x, w) -> sum(conv2d(x, w)), x, c.w)[2][2] ≈ grad((x, c) -> sum(c(x)), x, c)[2][2][(:w,)]
    
end
