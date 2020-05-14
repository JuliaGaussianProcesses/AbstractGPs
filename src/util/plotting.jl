using RecipesBase

@recipe f(gp::AbstractGP, x::Array) = gp(x)
@recipe f(gp::AbstractGP, x::AbstractRange) = gp(x)
@recipe f(gp::AbstractGP, xmin::Number, xmax::Number) = (gp, range(xmin, xmax, length=1000))
@recipe function f(gp::FiniteGP)
    x = gp.x
    f = gp.f
    ms = marginals(gp)

    @series begin
        μ = mean.(ms)
        σ = std.(ms)
        ribbon := σ
        fillalpha --> 0.3
        linewidth --> 2
        x, μ
    end
end

"""
    sampleplot(GP::FiniteGP, samples)

Plot samples from the given `FiniteGP`.

# Example
```julia
f = GP(SqExponentialKernel())
sampleplot(f(rand(10), 10; markersize=5)
```
The given example plots 10 samples from the given `FiniteGP`. The `markersize` is modified from default of 0.5 to 10.
"""
@userplot SamplePlot
@recipe function f(sp::SamplePlot)
    x = sp.args[1].x
    f = sp.args[1].f
    samples = sp.args[2]
    @series begin
        samples = rand(f(x, 1e-9), samples)
        seriestype --> :line
        linealpha --> 0.2
        markershape --> :circle
        markerstrokewidth --> 0.0
        markersize --> 0.5
        markeralpha --> 0.3
        seriescolor --> "red"
        label --> ""
        x, samples
    end
end

