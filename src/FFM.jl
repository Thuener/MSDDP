module FFM # Farma French model
using Distributions

export evaluate
export FFMData

type FFMData
  α::Array{Float64,1}
  β::Array{Float64,2}
  μ::Array{Float64,1}
  σ::Array{Float64,1}
end


function evaluate(ret_indexes::Array{Float64,2}, ret_assets::Array{Float64,2})
  N = size(ret_assets,1)
  Sa = size(ret_indexes,2)
  μ  = Array{Float64}(N)
  σ  = Array{Float64}(N)

  X = [ones(1,Sa); ret_indexes];
  β = X'\ret_assets'
  res = (ret_assets'-X'*β)'
  α = vec(β[1,:])
  β = β[2:end,:]

  for i = 1:N
    μ[i] = mean(res[i,:])
    σ[i] = var(res[i,:])
  end
  return FFMData(α, β, μ, σ)
end

end # end FFM module
