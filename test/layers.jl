@testset "layers" begin

    # Linear
    l = Linear(5, 4); x = rand(5, 10)
    gp = grad((W, b, x) -> sum(W * x .+ b), l.W, l.b, x)[2]
    gl = grad((l, x) -> sum(l(x)), l, x)[2]
    @test gp[1] == gl[1][(:W,)]
    @test gp[2] == gl[1][(:b,)]

    # Sequential
    l1 = Linear(5, 4); l2 = Linear(4, 3); s = Sequential(l1, l2); x = rand(5, 10)
    gp = grad((l1, l2, x) -> sum(l2(l1(x))), s.seq[1], s.seq[2], x)[2]
    gs = grad((s, x) -> sum(s(x)), s, x)[2]
    @test gp[1][(:W,)] == gs[1][(:seq, 1, :W)]
    @test gp[2][(:b,)] == gs[1][(:seq, 2, :b)]
    
    # Conv2d
    x = rand(7, 7, 3, 10); c = Conv2d(3, 5, 3)
    @test grad((x, w) -> sum(conv2d(x, w)), x, c.W)[2][2] == grad((x, c) -> sum(c(x)), x, c)[2][2][(:W,)]
    
end
