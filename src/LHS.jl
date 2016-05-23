module LHS
export lhsnorm

using Distributions

# Latin hypercube sampling for normal distribution
function lhsnorm(μ::Array{Float64,1}, Σ::Array{Float64,2}, n::Int64; rando=true)
  z = rand(MvNormal(μ,Σ),n)'

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
      x[:,i] = quantile(Normal(μ[i],sqrt(Σ[i,i])),x[:,i])
  end

  return x
end

end # end LHS
