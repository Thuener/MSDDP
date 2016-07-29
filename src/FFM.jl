module FFM # Farma French model
using Distributions

export evaluate
export FFMData

type FFMData
  α::Array{Float64,1}
  β::Array{Float64,2}
  ϵ::Array{Distributions.MvNormal,1}
end


function evaluate(ret_indexes::Array{Float64,2}, ret_assets::Array{Float64,2})
  N = size(ret_assets,1)
  Sa = size(ret_indexes,2)
  ϵ = Array(Distributions.MvNormal,N)

  X = [ones(1,Sa); ret_indexes];
  β = X'\ret_assets'
  res = (ret_assets'-X'*β)'
  α = β[1,:]
  β = β[2:end,:]

  for i = 1:N
    μ = mean(res[i,:])
    σ = var(res[i,:])
    ϵ[i] = MvNormal([μ],σ)
  end
  return FFMData(squeeze(α,1),β,ϵ)
end

end # end FFM module
