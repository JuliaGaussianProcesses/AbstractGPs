module TestUtils

using AbstractGPs
using Distributions
using LinearAlgebra
using Random
using Test

using AbstractGPs: AbstractGP, FiniteGP

"""
    test_finitegp_primary_public_interface(
        rng::AbstractRNG, fx::FiniteGP; atol::Real=1e-12
    )

Basic consistency tests for the Primary Public FiniteGP API.
You should run these tests if you only implement the Primary Public API -- see API section
of docs for details about this API.

These are consistency checks, not correctness tests in the absolute sense.
For example, these tests ensure that samples generated by `rand` are of the correct size,
but does not check that they come from the intended distribution.
"""
function test_finitegp_primary_public_interface(
    rng::AbstractRNG, fx::FiniteGP; atol::Real=1e-12
)
    # Check that `rand` works, and produces something of the correct shape.
    # Doesn't verify statistical properties.
    y = rand(rng, fx)
    @test y isa AbstractVector{<:Real}
    @test length(y) == length(fx)

    y = rand(fx)
    @test y isa AbstractVector{<:Real}
    @test length(y) == length(fx)

    rand!(rng, fx, y)
    rand!(fx, y)

    N_samples = 3
    Y = rand(rng, fx, N_samples)
    @test Y isa AbstractMatrix{<:Real}
    @test size(Y) == (length(fx), N_samples)

    Y = rand(fx, N_samples)
    @test Y isa AbstractMatrix{<:Real}
    @test size(Y) == (length(fx), N_samples)

    rand!(rng, fx, Y)
    rand!(fx, Y)

    # Ensure that `marginals` produces something of the expected size and type.
    ms = marginals(fx)
    @test ms isa AbstractVector{<:Normal}
    @test length(ms) == length(fx)

    # `mean`, `var`, and `mean_and_var` should be consistent with the output of marginals.
    @test mean(fx) ≈ mean.(ms)
    @test var(fx) ≈ var.(ms)
    @test mean_and_var(fx)[1] ≈ mean(fx)
    @test mean_and_var(fx)[2] ≈ var(fx)

    # All elements of `var` should be positive.
    @test all(var(fx) .> -atol)

    # Ensure that `logpdf` produces a scalar.
    @test logpdf(fx, y) isa Real

    # Ensure that `posterior` produces a new AbstractGP.
    @test posterior(fx, y) isa AbstractGP
end

"""
    test_finitegp_primary_and_secondary_public_interface(
        rng::AbstractRNG, fx::FiniteGP; atol::Real=1e-12
    )

Basic consistency tests for both the Primary and Secondary Public FiniteGP APIs.
Runs `test_finitegp_primary_public_interface` as part of these tests.
You should run these tests if you implement both the primary and secondary public APIs --
see API section of the docs for more information about these APIs.

These are consistency checks, not correctness tests in the absolute sense.
For example, these tests ensure that samples generated by `rand` are of the correct size,
but does not check that they come from the intended distribution.
"""
function test_finitegp_primary_and_secondary_public_interface(
    rng::AbstractRNG, fx::FiniteGP; atol=1e-12
)
    # Test the primary API.
    test_finitegp_primary_public_interface(rng, fx; atol=atol)

    # Check that `cov` runs and is consistent with `var`.
    @test diag(cov(fx)) ≈ var(fx)

    # Check that `mean_and_cov` runs and is consistent with `mean` and `cov`.
    @test mean_and_cov(fx)[1] ≈ mean(fx)
    @test mean_and_cov(fx)[2] ≈ cov(fx)

    # Verify that `cov(fx)` is positive-definite.
    @test eigmin(cov(fx)) > -atol
    @test cov(fx) ≈ cov(fx)'
end

"""
    test_internal_abstractgps_interface(
        rng::AbstractRNG,
        f::AbstractGP,
        x::AbstractVector,
        z::AbstractVector;
        atol=1e-12,
        σ²::Real=1e-9,
    )

Basic consistency tests for the Internal AbstractGPs API.
Runs `test_finitegp_primary_and_secondary_public_interface` as part of these tests.
Run these tests if you implement the Internal AbstractGPs API -- see the API section of the
docs for more information about this API.

These are consistency checks, not correctness tests in the absolute sense.
For example, these tests ensure that samples generated by `rand` are of the correct size,
but does not check that they come from the intended distribution.
"""
function test_internal_abstractgps_interface(
    rng::AbstractRNG,
    f::AbstractGP,
    x::AbstractVector,
    z::AbstractVector;
    atol=1e-12,
    σ²::Real=1e-9,
)
    if length(x) == length(z)
        throw(error("x and y should be of different lengths."))
    end

    # Verify that `mean` works and is the correct length and type.
    m = mean(f, x)
    @test m isa AbstractVector{<:Real}
    @test length(m) == length(x)

    # Verify that cov(f, x, z) works, is the correct size and type.
    C_xy = cov(f, x, z)
    @test C_xy isa AbstractMatrix{<:Real}
    @test size(C_xy) == (length(x), length(z))

    # Reversing arguments transposes the return.
    @test C_xy ≈ cov(f, z, x)'

    # Verify cov(f, x) works, is the correct size and type.
    C_xx = cov(f, x)
    @test size(C_xx) == (length(x), length(x))

    # Check that C_xx is positive definite.
    @test eigmin(Symmetric(C_xx)) > -atol

    # Check that C_xx is consistent with cov(f, x, x).
    @test C_xx ≈ cov(f, x, x)

    # Check that var(f, x) works, is the correct size and type.
    C_xx_diag = var(f, x)
    @test C_xx_diag isa AbstractVector{<:Real}
    @test length(C_xx_diag) == length(x)

    # Check C_xx_diag is consistent with cov(f, x).
    @test C_xx_diag ≈ diag(C_xx)

    # Check that mean_and_cov is consistent.
    let
        m, C = mean_and_cov(f, x)
        @test m ≈ mean(f, x)
        @test C ≈ cov(f, x)
    end

    # Check that mean_and_var is consistent.
    let
        m, c = mean_and_var(f, x)
        @test m ≈ mean(f, x)
        @test c ≈ var(f, x)
    end

    # Check that the entire FiniteGP interface has been successfully implemented.
    test_finitegp_primary_and_secondary_public_interface(rng, f(x, σ²); atol=atol)

    # Construct a FiniteGP, and check that all standard methods defined on it at least run.
    fx = f(x, σ²)
    fz = f(z, σ²)

    # Generate a sample, compute logpdf, compare against VFE and DTC.
    y = rand(fx)
    @test length(y) == length(x)
    @test logpdf(fx, y) isa Real
    @test elbo(fx, y, f(x)) ≈ logpdf(fx, y) rtol = 1e-5 atol = 1e-5
    @test elbo(fx, y, fz) <= logpdf(fx, y)
    @test dtc(fx, y, f(x)) ≈ logpdf(fx, y) rtol = 1e-5 atol = 1e-5
end

end
