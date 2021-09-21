# Represents a GP whose output is vector-valued.
struct VectorValuedGP{Tf<:AbstractGP}
    f::Tf
    num_outputs::Int
end

# I gave up figuring out how to properly subtype MatrixDistribution, but I want this to
# subtype a distribution type which indicates that samples from this distribution produces
# matrix of size num_features x num_outputs, or something like that.
struct FiniteVectorValuedGP{Tv<:VectorValuedGP,Tx<:AbstractVector,TΣy<:Real}
    v::Tv
    x::Tx
    Σy::TΣy
end

(f::VectorValuedGP)(x...) = FiniteVectorValuedGP(f, x...)

function Statistics.mean(vx::FiniteVectorValuedGP)

    # Construct equivalent FiniteGP.
    x_f = KernelFunctions.MOInputIsotopicByOutputs(vx.x, vx.v.num_outputs)
    f = vx.v.f
    fx = f(x_f, vx.Σy)

    # Compute quantity under equivalent FiniteGP.
    m = mean(fx)

    # Construct the matrix-version of the quantity.
    M = reshape(m, length(vx.x), vx.v.num_outputs)
    return M
end

function Statistics.var(vx::FiniteVectorValuedGP)

    # Construct equivalent FiniteGP.
    x_f = KernelFunctions.MOInputIsotopicByOutputs(vx.x, vx.v.num_outputs)
    f = vx.v.f
    fx = f(x_f, vx.Σy)

    # Compute quantity under equivalent FiniteGP.
    v = var(fx)

    # Construct the matrix-version of the quantity.
    V = reshape(v, length(vx.x), vx.v.num_outputs)
    return V
end

function Random.rand(rng::AbstractRNG, vx::FiniteVectorValuedGP)

    # Construct equivalent FiniteGP.
    x_f = KernelFunctions.MOInputIsotopicByOutputs(vx.x, vx.v.num_outputs)
    f = vx.v.f
    fx = f(x_f, vx.Σy)

    # Compute quantity under equivalent FiniteGP.
    y = rand(rng, fx)

    # Construct the matrix-version of the quantity.
    Y = reshape(y, length(vx.x), vx.v.num_outputs)
    return Y
end

function Distributions.logpdf(vx::FiniteVectorValuedGP, Y::AbstractMatrix{<:Real})

    # Construct equivalent FiniteGP.
    x_f = KernelFunctions.MOInputIsotopicByOutputs(vx.x, vx.v.num_outputs)
    f = vx.v.f
    fx = f(x_f, vx.Σy)

    # Construct flattened-version of observations.
    y = vec(Y)

    # Compute logpdf using FiniteGP.
    return logpdf(fx, y)
end

function posterior(vx::FiniteVectorValuedGP, Y::AbstractMatrix{<:Real})

    # Construct equivalent FiniteGP.
    x_f = KernelFunctions.MOInputIsotopicByOutputs(vx.x, vx.v.num_outputs)
    f = vx.v.f
    fx = f(x_f, vx.Σy)

    # Construct flattened-version of observations.
    y = vec(Y)

    # Construct posterior AbstractGP
    f_post = posterior(fx, y)

    # Construct a new vector-valued GP.
    return VectorValuedGP(f_post, vx.v.num_outputs)
end
