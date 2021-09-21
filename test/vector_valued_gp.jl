@testset "vector_valued_gp" begin
    f = GP(LinearMixingModelKernel([Matern52Kernel(), Matern12Kernel()], randn(2, 2)))
    x = range(0.0, 10.0; length=3)
    Σy = 0.1

    v = AbstractGPs.VectorValuedGP(f, 2)
    vx = v(x, Σy)

    M = mean(vx)

    rng = MersenneTwister(123456)
    Y = rand(rng, vx)
    logpdf(vx, Y)

    v_post = posterior(vx, Y)
end
