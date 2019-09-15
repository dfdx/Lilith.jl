@testset "layers" begin

    # Linear
    l = Linear(5, 4); x = rand(5, 10)
    gp = grad((W, b, x) -> sum(W * x .+ b), l.W, l.b, x)[2]
    gl = grad((l, x) -> sum(l(x)), l, x)[2]
    @test gp[1] == gl[1][(:W,)]
    @test gp[2] == gl[1][(:b,)]

    # Sequential
    # TODO
    
    # Conv2d
    x = rand(7, 7, 3, 10); c = Conv2d(3, 5, 3)
    @test grad((x, w) -> sum(conv2d(x, w)), x, c.W)[2][2] == grad((x, c) -> sum(c(x)), x, c)[2][2][(:W,)]
    
end
