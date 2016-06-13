module SDP
using JuMP, CPLEX
using Distributions
using Logging
using LHS, AR

export SDPData
export createmodel, findslot, findslots, forward, backward, splitequaly, evaluateprob

type SDPData
  N::Int64
  T::Int64
  L::Int64
  S::Int64
  α::Float64
  γ::Float64
end

# Split the z values using equal spaces
function splitequaly(dS::SDPData, z::Array{Float64,2})
  slot = (maximum(z)-minimum(z))/dS.L
  z_l = Array(Float64, dS.L)
  z_l[1] = minimum(z)+slot/2
  for l = 2:dS.L
    z_l[l] = z_l[l-1]+slot
  end

  return z_l
end

# Evaluate the probability for each interval
function evaluateprob(dS::SDPData, z::Array{Float64,2}, z_l::Array{Float64,1}, T_l::Int64, Sc::Int64)
  slot = z_l[2] - z_l[1]
  p = zeros(dS.L, dS.L)
  total = zeros(dS.L)
  for t = 1:T_l-1
    for se = 1:Sc
      # Try to find which slot z_t belongs
      for l = 1:dS.L
        # Found the slot z_t
        if z[t,se] >= z_l[l] -slot/2 && z[t,se] < z_l[l] + slot/2
          # Try to find which slot z_{t+1} belongs
          for l2 = 1:dS.L
            # Found the slot z_{t+1}
            if z[t+1,se] >= z_l[l2] -slot/2 && z[t+1,se] < z_l[l2] + slot/2
              p[l,l2] += 1
              break
            end
          end
          break
        end

      end
    end
  end

  for l = 1:dS.L
    if sum(p[l,:]) != 0
      p[l,:] = p[l,:]/sum(p[l,:])
    end
  end

  return p
end

function createmodel(dS::SDPData, p::Array{Float64,2}, r::Array{Float64,2}, Q_tp1::Array{Float64,1}, LP::Int64)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(Q, u0 >= 0)
  @variable(Q, u[1:dS.N] >= 0)
  @variable(Q, z)
  @variable(Q, y[1:dS.S] >= 0)

  @objective(Q, Max, sum{p[s]*(vecdot(r[:,s],u)+u0)*Q_tp1[s], s = 1:dS.S})

  wealth = @constraint(Q, u0 +sum{u[i], i = 1:dS.N} == 1).idx
  risk =  @constraint(Q,-(z - sum{p[s]*y[s] , s = 1:dS.S}/(1-dS.α)) <= dS.γ).idx

  @constraint(Q, trunc[s = 1:dS.S], y[s] >= z - vecdot(r[:,s],u))
  #sp = SubProbData( cash, assets, risk )
  return Q
end


function forward(dS::SDPData, u_l::Array{Float64,3}, z_l::Array{Float64,1}, zs::Array{Float64,1},
    rets::Array{Float64,2}; real_tc=0.0)
  W = 1
  for t = 1:dS.T-1
    slot = findslot(dS, zs[t], z_l)
    u = u_l[:,t,slot]
    W = vecdot(u[2:dS.N+1]*W, rets[:,t]) + u[1]*W
  end
  return W
end

# Find the slot which z belongs
function findslot(dS::SDPData, z::Float64, z_l::Array{Float64,1})
  for l = 1:dS.L-1
    if z >= z_l[l] && z < z_l[l+1]
      if abs(z- z_l[l]) <= abs(z- z_l[l+1])
        return l
      else
        return l+1
      end
    end
  end
  if z >= z_l[end]
    return dS.L
  end

  return 1
end

# Find the slot for each z_s
function findslots(dS::SDPData, z_s::Array{Float64,1}, z_l::Array{Float64,1})
  slots = Array(Int64,dS.S)

  for s = 1:dS.S
    slots[s] = findslot(dS, z_s[s], z_l )
  end

  return slots
end

function backward(dF::Factors, dS::SDPData, z_l::Array{Float64,1}, p::Array{Float64,2})
  u_l = Array(Float64, dS.N+1, dS.T, dS.L)
  Q_l = Array(Float64, dS.T, dS.L)
  # LHS
  e = lhsnorm(zeros(dS.N+1), dF.Σ, dS.S, rando=false)'
  for t = dS.T-1:-1:1
    info(" t = $t")
    for l = 1:dS.L
      info(" l = $l")
      r = Array(Float64, dS.N, dS.S)
      z_tp1 = Array(Float64, dS.S)
      for s = 1:dS.S
        r[:,s] = dF.a_r +dF.b_r*z_l[l] + e[1:dS.N,s]
        z_tp1[s] = dF.a_z[1] +dF.b_z[1]*z_l[l] + e[dS.N+1,s]
      end
      Q_s = Array(Float64, dS.S)
      if t+1 != dS.T
        slots = findslots(dS, z_tp1, z_l)
        Q_s = vec(Q_l[t+1,slots])
      else
        Q_s = ones(dS.S)
      end

      Q = createmodel(dS, p[l,:], r, Q_s, 2)
      writeLP(Q,"prob.lp")
      status = solve(Q)
      if status ≠ :Optimal
        writeLP(Q,"prob.lp")
        error("Can't solve the problem status:",status)
      end
      u = getvalue(getvariable(Q,:u))
      u0 = getvalue(getvariable(Q,:u0))
      u_l[:,t,l] = vcat(u0,u)
      Q_l[t,l] = getobjectivevalue(Q)
    end
  end
  return u_l, Q_l
end

end # end SDP
