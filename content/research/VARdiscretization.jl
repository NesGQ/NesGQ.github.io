import QuantEcon
import LinearAlgebra
import Random


function tpm(A::AbstractMatrix, 
             Σ::AbstractMatrix,
             GridSize::Vector{Int64}=10*Int.(ones(size(A,1))),
             UppBound::AbstractVector=sqrt(10)*sqrt.(diag(solve_discrete_lyapunov(A,Σ)));
             T::Real=10^6,
             Tburn::Real=10^5,
             trim::Bool=false,
             rng = MersenneTwister(1234))

    # number of processes to discretize
    m = size(A,1)
    # Total number of possible values of the discretized state
    n = prod(GridSize)

    # Grid of discrete values of state variables
    Grid = Array{AbstractVector}(undef, m)
    Threads.@threads for g in 1:m
        Grid[g] = collect(range(-UppBound[g],UppBound[g],GridSize[g]))
    end

    # Matrix of all possible states
    S = zeros(n,m)
    Threads.@threads for s in 1:m
        if s == 1
            S[:,s] = repeat(Grid[s],prod(GridSize[s+1:end]),1)
        elseif s > 1 && s < m
            ans = repeat(Grid[s]',prod(GridSize[1:s-1]),1)
            S[:,s] = repeat(vec(ans),prod(GridSize[s+1:end]),1)
        else
            S[:,s] = vec(repeat(Grid[s]',prod(GridSize[1:s-1]),1)) 
        end
    end
    
    # Markov transition probabilty matrix
    Π = zeros(n,n)
    #Ω = Σ^(1/2)
    Ω = cholesky(Σ).L
    x₀ = zero(GridSize) # Initialize the state
    xx = zero(S)
    stdist = vec(sum((S-xx).^2, dims=2))
    i = findmin(stdist)[2]
    #rng = Xoshiro(1234)

    # Simulation of the VAR
    for t in 1:T+Tburn
        xₜ = A*x₀ + Ω*randn(rng,m)
        xx = repeat(xₜ',n,1)
        stdist = vec(sum((S-xx).^2, dims=2))
        j = findmin(stdist)[2]

        if t > Tburn
            Π[i,j] = Π[i,j] + 1
        end

        x₀ = xₜ
        i = j
    end

    if trim == true
        inds = findall(!iszero, sum(Π, dims = 1))
        z = getindex.(inds,2)
        Π = Π[z,:]
        Π = Π[:,z]
        S = S[z,:]
    end

    inds = findall(iszero, sum(Π, dims = 2))
    z = getindex.(inds,1)
    Π[z,:] .= 1.0
    n₁ = size(Π,1)
    Threads.@threads for i in 1:n₁
        Π[i,:] = Π[i,:]./sum(Π[i,:])
    end
    Π[z,:] .= 0.0

    return (Π = Π, state_values = S)
end