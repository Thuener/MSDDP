module MSDDP

using JuMP, CPLEX, Util
using Distributions
using MathProgBase
using Logging
using JLD

export MKData, MSDDPData
export sddp, simulate, simulatesw, simulate_stateprob, simulatestates, readHMMPara, simulate_percport, createmodels
export chgConstrRHS

type MKData # Markov Data
  r::Array{Float64,4}
  ps_k::Array{Float64,2}
  k_ini::Int64
  P_K::Array{Float64,2}
end

type MSDDPData # MSDDP Data
  N::Int64
  T::Int64
  K::Int64
  S::Int64
  α::Float64
  x::Array{Float64,1}
  x0::Float64
  W_ini::Float64
  c::Float64
  M::Int64
  γ::Float64
  S_LB::Int64
  S_LB_inc::Int64
  S_FB::Int64
  GAPP::Float64
  Max_It::Int64
  α_lB::Float64
end

type SubProbData
  cash::Int64
  assets::Array{Int64,1}
  risk::Int64
end

function Base.copy(source::Array{Model,2})
  l,c = size(source)
  dest = Array(Model,l,c)
  for i = 1:l
    for j = 1:c
      dest[i,j] = copy(source[i,j])
    end
  end
  return dest
end
# Change the RHS using the index of the constraint
function chgConstrRHS(m::Model, idx::Int64, rhs::Number)
  constr = m.linconstr[idx]
   if constr.lb != -Inf
    if constr.ub != Inf
      if constr.ub == constr.lb
        sen = :(==)
      else
        return :range
      end
    else
      sen = :(>=)
    end
  else #if constr.lb == -Inf
    constr.ub == Inf && error("'Free' constraint sense not supported")
    sen = :(<=)
  end

  if sen == :range
    error("Modifying range constraints is currently unsupported.")
  elseif sen == :(==)
    constr.lb = float(rhs)
    constr.ub = float(rhs)
  elseif sen == :>=
    constr.lb = float(rhs)
  else
    @assert sen == :<=
    constr.ub = float(rhs)
  end
end

function getDual(m::Model, idx::Int64)
    if length(m.linconstrDuals) != MathProgBase.numlinconstr(m)
        error("Dual solution not available. Check that the model was properly solved and no integer variables are present.")
    end
    return m.linconstrDuals[idx]
end

function createmodel(dH::MSDDPData, dM::MKData, ret::Array{Float64,3}, p_state::Array{Float64,1}, LP)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @variable(Q, u0 >= 0)
  @variable(Q, u[1:dH.N] >= 0)
  @variable(Q, b[1:dH.N] >= 0)
  @variable(Q, d[1:dH.N] >= 0)
  @variable(Q, z)
  @variable(Q, y[1:dH.K,1:dH.S] >= 0)
  @variable(Q, θ[1:dH.K,1:dH.S] <= dH.M)

  @objective(Q, Max, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
                  + sum{(sum{ret[i,k,s]*u[i], i = 1:dH.N} + θ[k,s])*p_state[k]*dM.ps_k[s,k], k = 1:dH.K, s = 1:dH.S})

  cash = @constraint(Q, u0 + sum{(1+dH.c)*b[i] - (1-dH.c)*d[i], i = 1:dH.N} == dH.x0).idx
  assets = Array(Int64,dH.N)
  for i = 1:dH.N
    assets[i] = @constraint(Q, u[i] - b[i] + d[i] == dH.x[i]).idx
  end
  risk =  @constraint(Q,-(z - sum{p_state[k]*dM.ps_k[s,k]*y[k,s] , k = 1:dH.K, s = 1:dH.S}/(1-dH.α))
                          + dH.c*sum{b[i] + d[i], i = 1:dH.N} <= dH.γ*(sum(dH.x)+dH.x0)).idx

  @constraint(Q, trunc[k = 1:dH.K, s = 1:dH.S], y[k,s] >= z - sum{ret[i,k,s]*u[i], i = 1:dH.N})
  sp = SubProbData( cash, assets, risk )
  return Q, sp
end

function createmodels(dH::MSDDPData, dM::MKData, LP=2)
  sp = Array(SubProbData,dH.T-1,dH.K)
  AQ = Array(Model,dH.T-1,dH.K)

  for t = 1:dH.T-1
    for j = 1:dH.K
      ret = dM.r[t+1,:,:,:]
      p = dM.P_K[j,:]
      AQ[t,j], sp[t,j] = createmodel(dH, dM,  ret, p, LP)
      solve(AQ[t,j])
    end
  end

  # Add cuts to T-1
  for j = 1:dH.K
    θ = getvariable(AQ[dH.T-1,j],:θ)
    @constraint(AQ[dH.T-1,j],corte_ks[k = 1:dH.K, s = 1:dH.S], θ[k,s] ==  0)
    solve(AQ[dH.T-1,j])
  end
  return AQ, sp
end

# Simulating forward states
function simulatestates(dH::MSDDPData, dM::MKData, K_forward, r_forward)
  K_forward[1] = dM.k_ini

  for t = 2:dH.T
    # Simulating states
    prob_trans = vec(dM.P_K[K_forward[t-1],1:dH.K])
    K_forward[t] = rand(Categorical(prob_trans))

    # Simulationg forward scenarios
    r_idx = rand(Categorical(dM.ps_k[1:dH.S,K_forward[t]]))
    r_forward[1:dH.N,t] = dM.r[t,1:dH.N,K_forward[t],r_idx];
  end
end

function forward(dH::MSDDPData, dM::MKData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
    K_forward, ret; real_tc=0.0)

  # Initialize
  x_trial = zeros(dH.N,dH.T)
  x0_trial = zeros(dH.T)
  x_trial[1:dH.N,1] = dH.x
  x0_trial[1] = dH.x0
  u_trial = zeros(dH.N+1,dH.T)

  FO_forward = dH.W_ini
  for t = 1:dH.T-1
    k = K_forward[t]
    subp = sp[t,k]
    Q = AQ[t,k]

    chgConstrRHS(Q, subp.cash, x0_trial[t])
    chgConstrRHS(Q, subp.risk, dH.γ*(sum(x_trial[:,t])+x0_trial[t]) )
    for i = 1:dH.N
      chgConstrRHS(Q, subp.assets[i], x_trial[i,t])
    end
    # Resolve subprob in time t
    status = solve(Q)
    if status ≠ :Optimal
     writeLP(Q,"prob.lp")
     error("Can't solve the problem status:",status)
    end

    # Evalute immediate benefit
    b = getvariable(Q,:b)
    d = getvariable(Q,:d)
    u = getvariable(Q,:u)
    p_state = dM.P_K[K_forward[t],:]'
    @expression(Q, B_imed, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
             + sum{(sum{dM.r[t+1,i,j,s]*u[i], i = 1:dH.N} )*p_state[j]*dM.ps_k[s,j], j = 1:dH.K, s = 1:dH.S} )

    FO_forward += getvalue(B_imed)

    # Update trials
    u = getvariable(Q,:u)
    for i = 1:dH.N
      u_trial[i+1,t] = getvalue(u[i])
      x_trial[i,t+1] = (1+ret[i,t+1])*getvalue(u[i])
    end
    u_trial[1,t] = getvalue(getvariable(Q,:u0))

    # If transactional cost is different from the optimization model
    if real_tc != 0.0
      b = getvariable(Q,:b)
      d = getvariable(Q,:d)
      b_v = getvalue(b)
      d_v = getvalue(d)
      x0_trial[t+1] = - sum((1.0+real_tc)*b_v) + sum((1.0-real_tc)*d_v) + x0_trial[t]
    else
      x0_trial[t+1] = getvalue(getvariable(Q,:u0))
    end
  end

  return x_trial, x0_trial, FO_forward, u_trial
end

function backward(dH::MSDDPData, dM::MKData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
     x_trial, x0_trial)
  # Initialize
  α = ones(dH.T,dH.K)
  β = ones(dH.N+1,dH.T,dH.K)

  # Add cuts to t < T-1
  for t = dH.T-1:-1:2
    for j = 1:dH.K
      subp = sp[t,j]
      Q = AQ[t,j]

      chgConstrRHS(Q, subp.cash, x0_trial[t])
      chgConstrRHS(Q, subp.risk, dH.γ*(sum(x_trial[:,t])+x0_trial[t]) )
      for i = 1:dH.N
        chgConstrRHS(Q, subp.assets[i], x_trial[i,t])
      end

      status = solve(Q)
      if status ≠ :Optimal
       writeLP(Q,"prob.lp")
       error("Can't solve the problem status:",status)
      end

      # Evalute custs
      λ0 = getDual(Q, subp.cash)
      λ = zeros(dH.N)
      for i = 1:dH.N
        λ[i] = getDual(Q, subp.assets[i])
      end
      π = getDual(Q, subp.risk)
      FO = getobjectivevalue(Q)
      α[t,j] =  FO - (λ0 + dH.γ*π)*x0_trial[t] - sum([(λ[i] + dH.γ*π)*x_trial[i,t] for i = 1:dH.N])
      β[:,t,j] = vcat(λ0, λ) + dH.γ*π
    end

    for k = 1:dH.K
      addcut(dH, dM, AQ[t-1,k], α, β, t)
    end
  end
end

function addcut(dH::MSDDPData, dM::MKData, Q, α::Array{Float64,2}, β::Array{Float64,3}, t::Int64)
  θ = getvariable(Q,:θ)
  u = getvariable(Q,:u)
  u0 = getvariable(Q,:u0)

  @constraint(Q, corte_js[j = 1:dH.K, s = 1:dH.S],
      θ[j,s] <= α[t,j] + β[1,t,j]*u0 + sum{β[i+1,t,j]*(1+dM.r[t+1,i,j,s])*u[i], i = 1:dH.N})
end

function sddp( dH::MSDDPData, dM::MKData ;LP=2, parallel=false, simuLB=false, stabUB=0.5, fastLBcal=true)

  x_trial = []
  x0_trial = []
  u_trial = []
  r_forward = zeros(dH.N,dH.T)
  K_forward = Array(Int64,dH.T)

  AQ, sp = createmodels( dH, dM, LP )
  quatil = quantile(Normal(),dH.α_lB)

  GAP = 100.0
  It = 0
  S_LB_Ini = dH.S_LB

  list_firstu = vcat(dH.x0,dH.x)
  list_UB = [-1000]
  list_LB = [-1000 -1000]
  if parallel
    LB = SharedArray(Float64,dH.S_LB)
  else
    LB = Array(Float64,dH.S_LB)
  end
  UB = 9999999.0
  it_stable = 0
  UB_last = 9999999.0
  LB_conserv = 0.0
  eps_UB = 1e-6
  s_f = 0
  while abs(GAP) > dH.GAPP && UB > eps_UB && It < dH.Max_It
    It += 1
    tic()
    for s_f = 1:dH.S_FB
      # Forward
      debug("Forward Step memuse $(memuse())")
      simulatestates(dH, dM, K_forward, r_forward)
      x_trial, x0_trial, FO_forward, u_trial = forward(dH, dM, AQ, sp, K_forward, r_forward)

      # Backward
      debug("Backward Step memuse $(memuse())")
      backward(dH, dM, AQ, sp, x_trial, x0_trial)


      # Evaluate upper bound
      k = dM.k_ini
      t = 1
      Q = AQ[t,k]
      status = solve(Q)
      if status ≠ :Optimal
       writeLP(Q,"prob.lp")
       error("Can't solve the problem status:",status)
      end
      UB = getobjectivevalue(Q) + dH.W_ini
      u = getvalue(getvariable(Q,:u))
      u0 = getvalue(getvariable(Q,:u0))

      list_firstu = hcat(list_firstu,vcat(u0,u))

      debug("FO_forw = $FO_forward, UB = $UB, stabUB $(abs(UB/UB_last -1)*100)")
      if abs(UB/UB_last -1)*100 < stabUB || UB < eps_UB || isnan(abs(UB/UB_last -1)*100)
        it_stable += 1
        if it_stable >= 5
          if dH.S_LB < 30*S_LB_Ini && fastLBcal
            dH.S_LB = round(Int64,dH.S_LB + dH.S_LB_inc)
            info("Stable UB Increasing S_LB for $(dH.S_LB)")
            if parallel
              LB = SharedArray(Float64,dH.S_LB)
            else
              LB = Array(Float64,dH.S_LB)
            end
          end
          break
        end
      else
        it_stable = 0
        dH.S_LB = S_LB_Ini
        if parallel
          LB = SharedArray(Float64,dH.S_LB)
        else
          LB = Array(Float64,dH.S_LB)
        end
      end
      UB_last = UB
    end

    # Lower bound
    debug("Evaluating the Lower Bound memuse $(memuse())")
    LB_conserv = 0
    GAP = 100
    sumLB =0
    s_f = 0
    for s_f = 1:dH.S_LB
      simulatestates(dH, dM, K_forward, r_forward)
      x_trial, x0_trial, FO_forward, u_trial = forward(dH, dM, AQ, sp, K_forward, r_forward)
      LB[s_f] = FO_forward
      sumLB += FO_forward
      # Start testing after S_LB_Ini simulations
      if fastLBcal
        if s_f >= S_LB_Ini/3
          meanLB = sumLB/s_f
          GAP_mean = 100*(UB - meanLB)/UB
          if GAP_mean > dH.GAPP*3 && it_stable < 10
            info("GAP_mean $GAP_mean is higher than $(dH.GAPP*3) using $s_f Forwards. Aborting LB evaluation.")
            break
          end
        end

        if s_f >= S_LB_Ini
          meanLB = sumLB/s_f
          LB_conserv = (meanLB - quatil * std(LB[1:s_f])/sqrt(s_f))
          GAP = 100*(UB - LB_conserv)/UB
          if abs(GAP) < dH.GAPP
            info("SDDP ended: GAP LB $GAP is lower than $(dH.GAPP) using $s_f Forwards")
            break
          end

          GAP_mean = 100*(UB - meanLB)/UB
          if GAP_mean > dH.GAPP && it_stable < 10
            info("GAP_mean $GAP_mean is higher than $(dH.GAPP) UB using $s_f Forwards. Aborting LB evaluation.")
            break
          end
        end
      end
    end

    if !fastLBcal # Evaluate the lower bound always using S_LB forwards
      meanLB = sumLB/s_f
      LB_conserv = (meanLB - quatil * std(LB[1:s_f])/sqrt(s_f))
      GAP = 100*(UB - LB_conserv)/UB
    end
    list_UB = vcat(list_UB,UB)
    list_LB = vcat(list_LB,[mean(LB[1:s_f]) std(LB[1:s_f])/sqrt(s_f)])

    time_it = toq()
    info("LB = $LB_conserv, UB = $UB, GAP(%) = $GAP, Time_it: $time_it")
    if It >= dH.Max_It
      info("SDDP ended: maximum number of iterations exceeded")
      break
    end
    if UB <= eps_UB
      info("SDDP ended: UB is zero")
      break
    end
  end



  if simuLB
    file = jldopen("./output/allo_data_G$(string(dH.γ)[3:end])_C$(string(dH.c)[3:end]).jld", "w")
    debug("Evaluating the Lower Bound")
    K_forward_o = Array(Int64,dH.T)
    r_forward_o = zeros(dH.N,dH.T)
    addrequire(file, MSDDP)
    write(file,"dH",dH)
    write(file,"dM",dM)
    for s_f = 1:3
      simulatestates(dH, dM, K_forward_o, r_forward_o)
      x_trial_o, x0_trial_o, FO_forward_o, u_trial_o = forward(dH, dM, AQ, sp, K_forward_o, r_forward_o; parallel=parallel)
      write(file, "x$s_f", vcat(x0_trial_o',x_trial_o))
      write(file, "u$s_f", u_trial_o)
      write(file, "K$s_f", K_forward_o)
    end
  end
  dH.S_LB = S_LB_Ini
  return LB[1:s_f], UB, LB_conserv, AQ, sp, vcat(x0_trial',x_trial), u_trial, list_LB, list_UB, list_firstu
end

function simulate_stateprob(dH::MSDDPData, dM::MKData, AQ::Array{Model,2},sp::Array{SubProbData,2},
    ret_test::Array{Float64,2}, pk_r::Array{Float64,2}; real_tc=0.0)

   k_test = Array(Int64,dH.T-1)
   for t = 1:dH.T-1
     k_test[t] = findmax(pk_r[:,t])[2]
   end
   return simulate(dH, dM, AQ, sp, ret_test, k_test, real_tc=real_tc)
end

function simulate(dH::MSDDPData, dM::MKData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
    ret_test::Array{Float64,2}, k_test::Array{Int64,1}; real_tc=0.0)
  samps = size(ret_test,2)
  if samps != dH.T
    error("Return series has to have $(dH.T) and has $(samps) samples, use simulatesw if you want to really do that.")
  end

  x, x0, exp_ret, u = forward(dH, dM, AQ, sp, k_test, ret_test, real_tc=real_tc)

  return x, x0, exp_ret
end

# Simulate using sliding windows
function simulatesw(dH::MSDDPData, dM::MKData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
  ret_test::Array{Float64,2}, k_test::Array{Int64,1}; real_tc=0.0)
   dH_ = deepcopy(dH)
   init_T = dH_.T
   T_test = size(ret_test,2)
   its=floor(Int,(T_test-1)/(dH_.T-1))

   all_x = dH_.x
   all_x0 = dH_.x0
   for i = 1:its
     r_forward   = ret_test[:,(i-1)*(dH_.T-1)+1:(i)*(dH_.T-1)+1]
     K_forward_a =     k_test[(i-1)*(dH_.T-1)+1:(i)*(dH_.T-1)+1]

     x, x0, expret, u = forward(dH_, dM, AQ, sp, K_forward_a, r_forward, real_tc=real_tc)
     all_x = hcat(all_x,x[:,2:end])
     all_x0 = vcat(all_x0,x0[2:end])
     dH_.x = x[:,end]
     dH_.x0 = x0[end]
   end

   # Simulate last periods
   diff_t = round(Int, T_test-1 - (its*(dH_.T-1)))
   if diff_t > 0
     r_forward_a = zeros(dH_.N,diff_t)
     K_forward_a = Array(Int64,diff_t)
     r_forward_a = ret_test[:,its*(dH_.T-1)+1:end]
     K_forward_a =     k_test[its*(dH_.T-1)+1:end]
     dH_.T = diff_t +1
     x, x0, expret, u = forward(dH_, dM, AQ, sp, K_forward_a, r_forward_a, real_tc=real_tc)
     dH_.T = init_T
     all_x = hcat(all_x,x[:,2:end])
     all_x0 = vcat(all_x0,x0[2:end])
   end

   return all_x, all_x0
 end

function simulate_percport(dH::MSDDPData, ret_test::Array{Float64,2}, x_p::Array{Float64,1})
   T_test = size(ret_test,2)
   x = Array(Float64,dH.N+1, T_test)
   x[:,1] = vcat(dH.x0,dH.x)
   cost = 0.0
   for t = 2:T_test
     # Evaluate transaction costs
     total = sum(x[:,t-1])
     cost = 0.0
     for i = 2:dH.N+1
       cost += abs(x[i,t-1]-total*x_p[i])*dH.c
       x[i,t] = total*x_p[i]
     end
     x[1,t] = total*x_p[1] - cost

     # Evalute the return
     for i = 2:dH.N+1
       x[i,t] = (1+ret_test[i-1,t])*x[i,t]
     end

   end
   return x
 end

end # SDDP Module
