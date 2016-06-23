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
  x_ini::Array{Float64,1}
  x0_ini::Float64
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

  @objective(Q, Max, sum{p[s]*(vecdot(1+r[:,s],u)+u0)*Q_tp1[s], s = 1:dO.S})

  cash = @constraint(Q, u0 + sum{(1+dO.c)*b[i] - (1-dO.c)*d[i], i = 1:dO.N} == dO.x0_ini).idx
  assets = Array(Int64,dO.N)
  for i = 1:dO.N
    assets[i] = @constraint(Q, u[i] - b[i] + d[i] == dO.x_ini[i]).idx
  end

  risk =  @constraint(Q,-(z - sum{p[s]*y[s] , s = 1:dO.S}/(1-dO.α))
                          + dO.c*sum{b[i] + d[i], i = 1:dO.N} <= dO.γ*(sum(dO.x_ini)+dO.x0_ini)).idx

  @constraint(Q, trunc[s = 1:dO.S], y[s] >= z - vecdot(r[:,s],u))
  sp = SubProbData( cash, assets, risk )
  return Q, sp
end


function forward(dO::OSData, H_l::Array{Model,2}, sp::Array{SubProbData,2}, z_l::Array{Float64,1}, zs::Array{Float64,1},
    rets_tp1::Array{Float64,2})
  W = 1.0
  x_trial = zeros(dO.N,dO.T)
  x0_trial = zeros(dO.T)
  x_trial[1:dO.N,1] = dO.x_ini
  x0_trial[1] = dO.x0_ini

  for t = 1:dO.T-1
    slot = SDP.findslot(zs[t], z_l, dO.L)
    H = H_l[t,slot]
    subp = sp[t,slot]

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
    u = getvariable(H,:u)
    for i = 1:dO.N
      x_trial[i,t+1] = (1+rets_tp1[i,t+1])*getvalue(u[i])
    end
    x0_trial[t+1] = getvalue(getvariable(H,:u0))
  end
  return x_trial, x0_trial
end

function backward(dF::Factors, dO::OSData, z_l::Array{Float64,1})
  Q_l = ones(Float64, dO.T, dO.L)
  return backward(dF, dO, z_l, Q_l)
end

function backward(dF::Factors, dO::OSData, z_l::Array{Float64,1}, Q_l::Array{Float64,2})
  H_l = Array(Model,dO.T-1,dO.L)
  sp = Array(SubProbData,dO.T-1,dO.L)
  # LHS
  e = lhsnorm(zeros(dO.N+1), dF.Σ, dO.S, rando=false)'
  p_s = ones(dO.S)*1/dO.S
  for t = dO.T-1:-1:1
    info(" t = $t")
    for l = 1:dO.L
      r = Array(Float64, dO.N, dO.S)
      z_tp1 = Array(Float64, dO.S)
      for s = 1:dO.S
        ρ =  dF.a_r +dF.b_r*z_l[l] + e[1:dO.N,s]
        # Transform ρ = ln(r) in return (r)
        r[:,s] = exp(ρ)-1
        # Discount risk free rate
        r[:,s] -= dF.r_f
        z_tp1[s] = dF.a_z[1] +dF.b_z[1]*z_l[l] + e[dO.N+1,s]
      end
      Q_s = Array(Float64, dO.S)
      if t+1 != dO.T && dO.Mod
        slots = SDP.findslots(z_tp1, z_l, dO.S, dO.L)
        Q_s = vec(Q_l[t+1,slots])
      else
        Q_s = ones(dO.S)
      end

      H_l[t,l], sp[t,l] = createmodel(dO, p_s, r, Q_s)
    end
  end
  return H_l, sp
end

end # end SDP
