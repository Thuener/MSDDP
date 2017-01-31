# Stochastic Dynamic Programming for Farma French model
module SDP_FFM
using JuMP, CPLEX
using Distributions
using Logging
using LHS, FFM, MSDDP

export createmodel, backward

# Create SDP model
function createmodel(dH::MSDDPData, dM::MKData, ret::Array{Float64,3},
    p_state::Array{Float64,1}, V_tp1::Array{Float64,1}; LP=2)
  V = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(V, u0 >= 0)
  @variable(V, u[1:dH.N] >= 0)
  @variable(V, b[1:dH.N] >= 0)
  @variable(V, d[1:dH.N] >= 0)
  @variable(V, z)
  @variable(V, y[1:dH.K,1:dH.S] >= 0)
  @variable(V, θ[1:dH.K,1:dH.S] <= dH.M)

  @objective(V, Max, sum(((vecdot((1+ret[:,k,s]),u)+u0)*V_tp1[k])*p_state[k]*dM.ps_k[s,k]
                    for k = 1:dH.K, s = 1:dH.S))

  wealth = @constraint(V, u0 +sum(u[i] for i = 1:dH.N) == 1).idx
  risk =  @constraint(V,-(z - sum(p_state[k]*dM.ps_k[s,k]*y[k,s] for k = 1:dH.K, s = 1:dH.S)/(1-dH.α))
                          + dH.c*sum(b[i] + d[i] for i = 1:dH.N) <= dH.γ*(sum(dH.x)+dH.x0)).idx

  @constraint(V, trunc[k = 1:dH.K, s = 1:dH.S], y[k,s] >= z - vecdot(ret[:,k,s],u))
  return V
end

function backward(dH::MSDDPData, dM::MKData)
  β = Array(Float64, dH.K, dH.T)
  Vʲₜ = ones(Float64,dH.T,dH.K)
  for t = dH.T-1:-1:1
    info(" t = $t")
    for k = 1:dH.K
      ret = dM.r[t+1,:,:,:]
      p = dM.P_K[k,:]
      V = createmodel(dH, dM,  ret, p, Vʲₜ[t+1,:])
      status = solve(V)
      if status ≠ :Optimal
        writeLP(Q,"prob.lp")
        error("Can't solve the problem status:",status)
      end
      Vʲₜ[t,k] = getobjectivevalue(V)
    end
  end
  return Vʲₜ
end

end # end SDP_FFM
