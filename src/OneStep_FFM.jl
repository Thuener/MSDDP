# One step model with transactional costs for Farma French model
module OneStep_FFM
using JuMP, CPLEX
using Distributions
using Logging
using LHS, FFM, MSDDP

export createmodel, backward

# Create one step model
function createmodel(dH::MSDDPData, dM::MKData, ret::Array{Float64,3},
    p_state::Array{Float64,1}, V_tp1::Array{Float64,1}; LP=2)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(Q, u0 >= 0)
  @variable(Q, u[1:dH.N] >= 0)
  @variable(Q, b[1:dH.N] >= 0)
  @variable(Q, d[1:dH.N] >= 0)
  @variable(Q, z)
  @variable(Q, y[1:dH.K,1:dH.S] >= 0)
  @variable(Q, θ[1:dH.K,1:dH.S] <= dH.M)

  @objective(Q, Max, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
                  + sum{((vecdot((1+ret[:,k,s]),u)+u0)*V_tp1[k])*p_state[k]*dM.ps_k[s,k], k = 1:dH.K, s = 1:dH.S})

  cash = @constraint(Q, u0 + sum{(1+dH.c)*b[i] - (1-dH.c)*d[i], i = 1:dH.N} == dH.x0_ini).idx
  assets = Array(Int64,dH.N)
  for i = 1:dH.N
    assets[i] = @constraint(Q, u[i] - b[i] + d[i] == dH.x_ini[i]).idx
  end
  risk =  @constraint(Q,-(z - sum{p_state[k]*dM.ps_k[s,k]*y[k,s] , k = 1:dH.K, s = 1:dH.S}/(1-dH.α))
                          + dH.c*sum{b[i] + d[i], i = 1:dH.N} <= dH.γ*(sum(dH.x_ini)+dH.x0_ini)).idx

  @constraint(Q, trunc[k = 1:dH.K, s = 1:dH.S], y[k,s] >= z - sum{ret[i,k,s]*u[i], i = 1:dH.N})
  sp = MSDDP.SubProbData( cash, assets, risk )
  return Q, sp
end

function createmodels(dH::MSDDPData, dM::MKData, V_t::Array{Float64,2})
  sp = Array(MSDDP.SubProbData,dH.T-1,dH.K)
  AQ = Array(Model,dH.T-1,dH.K)

  for t = 1:dH.T-1
    for k = 1:dH.K
      ret = dM.r[t+1,:,:,:]
      p = dM.P_K[k,:]
      AQ[t,k], sp[t,k] = createmodel(dH, dM,  ret, p, V_t[t+1,:])
    end
  end
  return AQ, sp
end

end # end OneStep_FFM
