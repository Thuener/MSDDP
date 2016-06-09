module AR

using Distributions

export Factors

type Factors
  a_z::Array{Float64,1}
  a_r::Array{Float64,1}
  b_z::Array{Float64,1}
  b_r::Array{Float64,1}
  Σ::Array{Float64,2}
  r_f::Float64
end


# Generate the z_t series
function generateseries_z(dF::Factors, T::Int64, S::Int64, T_l::Int64)
  norm = MvNormal(dF.Σ)

  # Generate the series
  p = zeros(2,T,S)
  for s=1:S
    p[1,1,s] = dF.a_z[1]
    for t=1:T-1
      sm = rand(norm)
      p[t+1,s] = dF.a_z[1] + dF.b_z[1]*p[t,s] + sm[end]
    end
  end

  return p[T_l+1:T,:]
end

# Generate the z_{t+1} z_t series
function generateseries_zs(dF::Factors, T::Int64, S::Int64, T_l::Int64)

  z = generateseries_z(dF, T, S, T_l)
  y = Array(Float64, 2, T_l-1, S)
  for s = 1:S
		y[:,:,s] = vcat(hcat(z[:,s]',0),hcat(0,z[:,s]'))[:,2:T_l]
	end
  return y
end

# Generate the risk assets series
function generateseries_assets(dF::Factors, T::Int64, S::Int64, T_l::Int64)
  norm = MvNormal(dF.Σ)
  N = length(dF.a_r)
  # Generate the series
  p = zeros(N+1,T+1,S)
  for s=1:S
    p[N+1,1,s] = dF.a_z[1]
    for t=1:T
      sm = rand(norm)
      p[1:N,t+1,s] = dF.a_r + dF.b_r*p[N+1,t,s] + sm[1:N] .- dF.r_f
      if t < T
        p[N+1,t+1,s] = dF.a_z[1] + dF.b_z[1]*p[N+1,t,s] + sm[N+1]
      end
    end
  end
  return p[1:N,T_l+2:T+1,:]
end

# Generate the risk assets series and z_t
function generateseries(dF::Factors,T::Int64,N::Int64,S::Int64,T_l::Int64; writef=true)
  norm = MvNormal(dF.Σ)

  # Generate the series
  p = zeros(N+1,T+1,S)
  for s=1:S
    p[N+1,1,s] = dF.a_z[1]
    for t=1:T
      sm = rand(norm)
      p[1:N,t+1,s] = dF.a_r + dF.b_r*p[N+1,t,s] + sm[1:N] .- dF.r_f
      if t < T
        p[N+1,t+1,s] = dF.a_z[1] + dF.b_z[1]*p[N+1,t,s] + sm[N+1]
      end
    end
  end
  p2 = p[:,T_l+2:T+1,:]
  #p2 = reshape(p[:],N+1,120,1000)
  if writef
    writecsv("../../input/$(N)MS_120_$(S).csv",p2[:])#hcat(p',z))
  end
  return p2
end


end # end AR module
