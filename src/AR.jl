module AR # Auto regressive model

using Distributions

export series_z, series_zs, series_assets, series
export ARData

type ARData
  a_z::Array{Float64,1}
  a_r::Array{Float64,1}
  b_z::Array{Float64,1}
  b_r::Array{Float64,1}
  Σ::Array{Float64,2}
  r_f::Float64
end


# Generate the z_t series
function series_z(dF::ARData, T::Int64, S::Int64, T_max::Int64)
  N = length(dF.a_r)
  ρ = series(dF, S, T_max)
  return ρ[N+1,:,:]
end

# Generate the z_{t+1} z_t series
function series_zs(dF::ARData, S::Int64, T_max::Int64)

  z = series_z(dF, S, T_max)
  y = Array(Float64, 2, T_max-1, S)
  for s = 1:S
		y[:,:,s] = vcat(hcat(z[:,s]',0),hcat(0,z[:,s]'))[:,2:T_max]
	end
  return y
end

# Generate the risk assets series
function series_assets(dF::ARData, S::Int64, T_max::Int64)
  N = length(dF.a_r)
  ρ = series(dF, S, T_max)
  return ρ[1:N,:,:]
end

# Generate the risk assets series and z_t
function series(dF::ARData, S::Int64, T_max::Int64)
  norm = MvNormal(dF.Σ)
  N = length(dF.a_r)

  # Generate the series
  p = zeros(N+1, T_max, S)
  for s=1:S
    p[N+1,1,s] = dF.a_z[1]
    for t=1:T_max-1
      sm = rand(norm)
      p[1:N,t+1,s] = dF.a_r + dF.b_r*p[N+1,t,s] + sm[1:N]
      p[N+1,t+1,s] = dF.a_z[1] + dF.b_z[1]*p[N+1,t,s] + sm[N+1]
    end
  end
  return p
end


end # end AR module
