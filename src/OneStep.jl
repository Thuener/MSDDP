module OneStep
using JuMP, CPLEX
using Distributions
using Logging
using LHS, AR
import SDP, MSDDP

export OSData
export createmodel, findslot, findslots, forward, backward, splitequaly

type OSData
  N::Int64
  T::Int64
  L::Int64
  S::Int64
  α::Float64
  c::Float64
  γ::Float64
  Mod::Bool
  x::Array{Float64,1}
  x0::Float64
end

type SubProbData
  cash::Int64
  assets::Array{Int64,1}
  risk::Int64
end

function createmodel(dO::OSData, p::Array{Float64,1}, r::Array{Float64,2}, Q_tp1::Array{Float64,1}; LP=2)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(Q, u0 >= 0)
  @variable(Q, u[1:dO.N] >= 0)
  @variable(Q, b[1:dO.N] >= 0)
  @variable(Q, d[1:dO.N] >= 0)
  @variable(Q, z)
  @variable(Q, y[1:dO.S] >= 0)

  @objective(Q, Max, sum(p[s]*(vecdot(1+r[:,s],u)+u0)*Q_tp1[s] for s = 1:dO.S))

  cash = @constraint(Q, u0 + sum((1+dO.c)*b[i] - (1-dO.c)*d[i] for i = 1:dO.N) == dO.x0).idx
  assets = Array(Int64,dO.N)
  for i = 1:dO.N
    assets[i] = @constraint(Q, u[i] - b[i] + d[i] == dO.x[i]).idx
  end

  risk =  @constraint(Q,-(z - sum(p[s]*y[s] for s = 1:dO.S)/(1-dO.α))
                          + dO.c*sum(b[i] + d[i] for i = 1:dO.N) <= dO.γ*(sum(dO.x)+dO.x0)).idx

  @constraint(Q, trunc[s = 1:dO.S], y[s] >= z - vecdot(r[:,s],u))
  sp = SubProbData( cash, assets, risk )
  return Q, sp
end


function forward(dO::OSData, dF::ARData, dS::SDP.SDPData, β::Array{Float64,2}, z_l::Array{Float64,1}, z_t::Array{Float64,1},
    rets_t::Array{Float64,2})
  x_trial = zeros(dO.N,dO.T)
  x0_trial = zeros(dO.T)
  x_trial[1:dO.N,1] = dO.x
  x0_trial[1] = dO.x0
  p_s = ones(dS.S)*1/dS.S

  for t = 1:dO.T-1
    Q_s, r_s = SDP.samplestp1(dS, dF, vec(β[t+1,:]), z_l, z_t[t])
    if t+1 == dS.T || ~dO.Mod
      Q_s =  ones(Float64,dS.S)
    end
    H, subp = createmodel(dO, p_s, r_s, Q_s)

    # Change x
    MSDDP.chgConstrRHS(H, subp.cash, x0_trial[t])
    MSDDP.chgConstrRHS(H, subp.risk, dO.γ*(sum(x_trial[:,t])+x0_trial[t]) )
    for i = 1:dO.N
      MSDDP.chgConstrRHS(H, subp.assets[i], x_trial[i,t])
    end

    status = solve(H)
    if status ≠ :Optimal
      writeLP(H,"prob.lp")
      error("Can't solve the problem status:",status)
    end
    u = getindex(H,:u)
    for i = 1:dO.N
      x_trial[i,t+1] = (1+rets_t[i,t+1])*getvalue(u[i])
    end
    x0_trial[t+1] = getvalue(getindex(H,:u0))
  end
  return x_trial, x0_trial
end

end # end SDP
