module LHS
export lhsnorm

using Distributions

function lhsnorm(μ::Float64, σ::Float64, n::Int64; rando=true)
  z = rand(Normal(μ,σ),n)
  x = zeros(size(z))

  idx = sortperm(z)
  x[idx] = collect(1:n)

  if rando
    x = (x - rand(size(x)))/n
  else
    x = (x - 0.5)/n
  end

  x = quantile.(Normal(μ,sqrt(σ)),x)

  return x
end
# Latin hypercube sampling for normal distribution
function lhsnorm(μ::Array{Float64,1}, Σ::Array{Float64,2}, n::Int64; rando=true)
  z = rand(MvNormal(μ,full(Symmetric(Σ))),n)'

  p = length(μ)
  x = zeros(size(z))
  for i=1:p
    idx = sortperm(z[:,i])
    x[idx,i] = collect(1:n)
  end

  if rando
    x = (x - rand(size(x)))/n
  else
    x = (x - 0.5)/n
  end

  for i=1:p
      x[:,i] = quantile.(Normal(μ[i],sqrt(Σ[i,i])),x[:,i])
  end

  return x
end

end # end LHS
