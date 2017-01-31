# Stochastic Dynamic Programming for Autoregressive model
module SDP
using JuMP, CPLEX
using Distributions
using Logging
using LHS, AR

export SDPData
export createmodel, findslot, findslots, forward, backward, splitequaly

type SDPData
  N::Int64
  T::Int64
  L::Int64
  S::Int64
  α::Float64
  γ::Float64
end

function evaluateregression(z_l::Array{Float64,1},Q_l::Array{Float64,1})
  L = length(z_l)

  X = vcat(ones(1,L), z_l',z_l'.^2)
  β = X'\Q_l
  return vec(β)
end

# Split the z values using equal spaces
function splitequaly(L::Int64, z::Array{Float64,2})
  slot = (maximum(z)-minimum(z))/L
  z_l = Array(Float64, L)
  z_l[1] = minimum(z)+slot/2
  for l = 2:L
    z_l[l] = z_l[l-1]+slot
  end

  return z_l
end

#Interpolate Q_t using z_t
function samplestp1(dS::SDPData, dF::ARData, β::Array{Float64,1}, z_l::Array{Float64,1}, z_t::Float64)
  ϵ = lhsnorm(zeros(dS.N+1), dF.Σ, dS.S, rando=true)'
  r_s = Array(Float64, dS.N, dS.S)
  Q_s = Array(Float64, dS.S)
  z_tp1 = Array(Float64, dS.S)
  for s = 1:dS.S
    ρ =  dF.a_r +dF.b_r*z_t + ϵ[1:dS.N,s]
    # Transform ρ = ln(r) in return (r)
    r_s[:,s] = exp(ρ)-1
    # Discount risk free rate
    r_s[:,s] -= dF.r_f
    z_tp1[s] = dF.a_z[1] +dF.b_z[1]*z_t + ϵ[dS.N+1,s]
  end

  X = vcat(ones(1,dS.S), z_tp1',z_tp1'.^2)
  Q_s = X'*β
  return Q_s, r_s
end

# Create SDP model
function createmodel(dS::SDPData, p::Array{Float64,1}, r::Array{Float64,2}, Q_tp1::Array{Float64,1}; LP::Int64=2)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(Q, u0 >= 0)
  @variable(Q, u[1:dS.N] >= 0)
  @variable(Q, z)
  @variable(Q, y[1:dS.S] >= 0)

  @objective(Q, Max, sum(p[s]*(vecdot(1+r[:,s],u)+u0)*Q_tp1[s] for s = 1:dS.S))

  wealth = @constraint(Q, u0 +sum(u[i] for i = 1:dS.N) == 1).idx
  risk =  @constraint(Q,-(z - sum(p[s]*y[s] for s = 1:dS.S)/(1-dS.α)) <= dS.γ).idx

  @constraint(Q, trunc[s = 1:dS.S], y[s] >= z - vecdot(r[:,s],u))
  #sp = SubProbData( cash, assets, risk )
  return Q
end

function forward(dS::SDPData, dF::ARData, β::Array{Float64,2}, z_l::Array{Float64,1}, z_t::Array{Float64,1},
    rets_t::Array{Float64,2})
  W = 1.0
  p_s = ones(dS.S)*1/dS.S
  all = Array(Float64,dS.N+1, dS.T-1)
  for t = 1:dS.T-1
    Q_s, r_s = samplestp1(dS, dF, vec(β[t+1,:]), z_l, z_t[t])
    if t+1 == dS.T
      Q_s =  ones(Float64,dS.S)
    end
    Q = createmodel(dS, p_s, r_s, Q_s)
    status = solve(Q)
    if status ≠ :Optimal
      writeLP(Q,"prob.lp")
      error("Can't solve the problem status:",status)
    end
    u0 = getvalue(getvariable(Q,:u0))
    u = getvalue(getvariable(Q,:u))
    u = vcat(u0,u)
    all[2:dS.N+1,t] = (u[2:dS.N+1]*W).*(1+rets_t[:,t+1])
    all[1,t] = u[1]*W
    W = sum(all[:,t])
  end
  return W, all
end

# Find the slot which z belongs
function findslot(z::Float64, z_l::Array{Float64,1}, L::Int64)
  for l = 1:L-1
    if z >= z_l[l] && z < z_l[l+1]
      if abs(z- z_l[l]) <= abs(z- z_l[l+1])
        return l
      else
        return l+1
      end
    end
  end
  if z >= z_l[end]
    return L
  end

  return 1
end

# Find the slot for each z_s
function findslots(z_s::Array{Float64,1}, z_l::Array{Float64,1}, S::Int64, L::Int64)
  slots = Array(Int64,S)

  for s = 1:S
    slots[s] = findslot(z_s[s], z_l, L )
  end

  return slots
end

function backward(dF::ARData, dS::SDPData, z_l::Array{Float64,1})
  Q_l = ones(Float64, dS.L)
  β = Array(Float64, dS.T, 3)
  p_s = ones(dS.S)*1/dS.S
  for t = dS.T-1:-1:1
    info(" t = $t")
    β[t+1,:] = evaluateregression(z_l,Q_l)
    for l = 1:dS.L
      Q_s, r_s = samplestp1(dS, dF, vec(β[t+1,:]), z_l, z_l[l])
      if t+1 == dS.T
        Q_s = ones(Float64,dS.S)
      end
      Q = createmodel(dS, p_s, r_s, Q_s)
      status = solve(Q)
      if status ≠ :Optimal
        writeLP(Q,"prob.lp")
        error("Can't solve the problem status:",status)
      end
      Q_l[l] = getobjectivevalue(Q)
    end
  end
  return β
end

end # end SDP
