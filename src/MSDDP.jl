module MSDDP

using JuMP, CPLEX
using Distributions
using MathProgBase
using Logging
using JLD

export HMMData, MSDDPData
export sddp, simulate, simulatesw, simulate_stateprob, simulatestates, readHMMPara, simulate_percport

#=
function debug(msg)
  println("DEBUG: ",msg)
end

function Base.info(msg)
  println("INFO: ",msg)
end
=#
type Cut
  λ0::Float64
  λ::Array{Float64,1}
  π::Float64
  FO::Float64
end

type HMMData
  r::Array{Float64,3}
  ps_j::Array{Float64,2}
  k_ini::Int64
  P_K::Array{Float64,2}
end

type MSDDPData
  N::Int64
  T::Int64
  K::Int64
  S::Int64
  α::Float64
  x_ini::Array{Float64,1}
  x0_ini::Float64
  c::Float64
  M::Int64
  γ::Float64
  S_LB::Int64
  S_FB::Int64
  GAPP::Float64
  Max_It::Int64
  α_lB::Float64
end

type SubProbData
  caixa::Int64
  ativos::Array{Int64,1}
  risco::Int64
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

function createmodel(dH::MSDDPData, dM::HMMData, p_state, LP)
  Q = Model(solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=LP))
  @defVar(Q, u0 >= 0)
  @defVar(Q, u[1:dH.N] >= 0)
  @defVar(Q, b[1:dH.N] >= 0)
  @defVar(Q, d[1:dH.N] >= 0)
  @defVar(Q, z)
  @defVar(Q, y[1:dH.K,1:dH.S] >= 0)
  @defVar(Q, θ[1:dH.K,1:dH.S] <= dH.M)

  @setObjective(Q, Max, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
                  + sum{(sum{dM.r[i,j,s]*u[i], i = 1:dH.N} + θ[j,s])*p_state[j]*dM.ps_j[s,j], j = 1:dH.K, s = 1:dH.S})

  caixa = @addConstraint(Q, u0 + sum{(1+dH.c)*b[i] - (1-dH.c)*d[i], i = 1:dH.N} == dH.x0_ini).idx
  ativos = Array(Int64,dH.N)
  for i = 1:dH.N
    ativos[i] = @addConstraint(Q, u[i] - b[i] + d[i] == dH.x_ini[i]).idx
  end
  risco =  @addConstraint(Q,-(z - sum{p_state[j]*dM.ps_j[s,j]*y[j,s] , j = 1:dH.K, s = 1:dH.S}/(1-dH.α))
                          + dH.c*sum{b[i] + d[i], i = 1:dH.N} <= dH.γ*(sum(dH.x_ini)+dH.x0_ini)).idx

  @addConstraint(Q, trunc[j = 1:dH.K, s = 1:dH.S], y[j,s] >= z - sum{dM.r[i,j,s]*u[i], i = 1:dH.N})
  sp = SubProbData( caixa, ativos, risco )
  return Q, sp
end

function createmodels(dH::MSDDPData, dM::HMMData, LP)
  sp = Array(SubProbData,dH.T-1,dH.K)
  AQ = Array(Model,dH.T-1,dH.K)

  for t = 1:dH.T-1
    for k = 1:dH.K
      AQ[t,k], sp[t,k] = createmodel(dH, dM, dM.P_K[k,:]', LP)
    end
  end
  return AQ, sp
end

# Simulando estados forward
function simulatestates(dH::MSDDPData, dM::HMMData, K_forward, r_forward)
  K_forward[1] = dM.k_ini

  for t = 2:dH.T
    # Simulando estados forward
    prob_trans = squeeze(dM.P_K[K_forward[t-1],1:dH.K],1);
    K_forward[t] = rand(Categorical(prob_trans));

    # Simulando cenário forward
    r_idx = rand(Categorical(dM.ps_j[1:dH.S,K_forward[t]]))
    r_forward[1:dH.N,t] = dM.r[1:dH.N,K_forward[t],r_idx];
  end
end

function forward(dH::MSDDPData, dM::HMMData, AQ::Array{Model,2}, sp::Array{SubProbData,2}, K_forward, ret; real_tc=0.0)

  # Inicializando
  x_trial = zeros(dH.N,dH.T)
  x0_trial = zeros(dH.T)
  x_trial[1:dH.N,1] = dH.x_ini
  x0_trial[1] = dH.x0_ini
  u_trial = zeros(dH.N+1,dH.T)

  FO_forward = 0
  for t = 1:dH.T-1
    k = K_forward[t]
    subp = sp[t,k]
    Q = AQ[t,k]

    chgConstrRHS(Q, subp.caixa, x0_trial[t])
    chgConstrRHS(Q, subp.risco, dH.γ*(sum(x_trial[:,t])+x0_trial[t]) )
    for i = 1:dH.N
      chgConstrRHS(Q, subp.ativos[i], x_trial[i,t])
    end
    # Resolvendo o subprob do tempo t
    status = solve(Q)
    if status ≠ :Optimal
      writeLP(Q,"prob.lp")
      error("Can't solve the problem status:",status)
    end

    # Calculando benefício imediato
    b = getVar(Q,:b)
    d = getVar(Q,:d)
    u = getVar(Q,:u)
    p_state = dM.P_K[K_forward[t],:]'
    @defExpr(B_imed, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
             + sum{(sum{dM.r[i,j,s]*u[i], i = 1:dH.N} )*p_state[j]*dM.ps_j[s,j], j = 1:dH.K, s = 1:dH.S} )

    FO_forward += getValue(B_imed)

    # Atualizando o estado
    u = getVar(Q,:u)
    for i = 1:dH.N
      u_trial[i+1,t] = getValue(u[i])
      x_trial[i,t+1] = (1+ret[i,t+1])*getValue(u[i])
    end
    u_trial[1,t] = getValue(getVar(Q,:u0))

    # If transactional cost is different from the optimization model
    if real_tc != 0.0
      b = getVar(Q,:b)
      d = getVar(Q,:d)
      x0_trial[t+1] = - sum((1+real_tc)*b) + sum((1-real_tc)*d) + x0_trial[t]
    else
      x0_trial[t+1] = getValue(getVar(Q,:u0))
    end
  end
  #debug("NEW FO_forw = $FO_forward, x_T = $(x0_trial[end]+sum(x_trial[:,end]))")
  return x_trial, x0_trial, FO_forward, u_trial
end

function backward(dH::MSDDPData, dM::HMMData, AQ::Array{Model,2}, sp::Array{SubProbData,2}, x_trial, x0_trial)
  # Inicializando
  cuts = Array(Cut,dH.T,dH.K)
  α = ones(dH.T,dH.K)
  β = ones(dH.N+1,dH.T,dH.K)

  # Adicionando cortes para T-1
  for k = 1:dH.K
    θ = getVar(AQ[dH.T-1,k],:θ)
    @addConstraint(AQ[dH.T-1,k],corte_js[j = 1:dH.K, s = 1:dH.S], θ[j,s] ==  0)
  end

  # Adicionando cortes para t < T-1
  for t = dH.T-2:-1:1
    for j = 1:dH.K
      subp = sp[t+1,j]
      Q = AQ[t+1,j]

      chgConstrRHS(Q, subp.caixa, x0_trial[t+1])
      chgConstrRHS(Q, subp.risco, dH.γ*(sum(x_trial[:,t+1])+x0_trial[t+1]) )
      for i = 1:dH.N
        chgConstrRHS(Q, subp.ativos[i], x_trial[i,t+1])
      end

      # Resolvendo
      status = solve(Q)
      if status ≠ :Optimal
        info(Q)
        error("Can't solve the problem status:",status)
      end
      b = getVar(Q,:b)
      d = getVar(Q,:d)
      u = getVar(Q,:u)
      θ = getVar(Q,:θ)


      # Computando corte
      λ0 = getDual(Q, subp.caixa)
      λ = zeros(dH.N)
      for i = 1:dH.N
        λ[i] = getDual(Q, subp.ativos[i])
      end
      π = getDual(Q, subp.risco)
      FO = getObjectiveValue(Q)
      cuts[t+1,j] = Cut(λ0, λ, π, FO)
      α[t+1,j] =  cuts[t+1,j].FO - (cuts[t+1,j].λ0 + dH.γ*cuts[t+1,j].π)*x0_trial[t+1] -
      sum([(cuts[t+1,j].λ[i] + dH.γ*cuts[t+1,j].π)*x_trial[i,t+1] for i = 1:dH.N])
      β[:,t+1,j] = vcat(λ0, λ) + dH.γ*π
    end

    for k = 1:dH.K
      addcut(dH, dM, AQ[t,k], cuts, t, x0_trial, x_trial)
    end
  end
  return α, β
end

function addcut(dH::MSDDPData, dM::HMMData, Q, cuts, t, x0_trial, x_trial)
  θ = getVar(Q,:θ)
  u = getVar(Q,:u)
  u0 = getVar(Q,:u0)
  @addConstraint(Q,corte_js[j = 1:dH.K, s = 1:dH.S],
      θ[j,s] <= cuts[t+1,j].FO + (cuts[t+1,j].λ0 + dH.γ*cuts[t+1,j].π)*(u0 - x0_trial[t+1]) +
      + sum{(cuts[t+1,j].λ[i] + dH.γ*cuts[t+1,j].π)*((1+dM.r[i,j,s])*u[i] - x_trial[i,t+1]), i = 1:dH.N})
end

function sddp( dH::MSDDPData, dM::HMMData ;LP=2, parallel=false, simuLB=false )

  x_trial = []
  x0_trial = []
  u_trial = []
  r_forward = zeros(dH.N,dH.T);
  K_forward = Array(Int64,dH.T);

  AQ, sp = createmodels( dH, dM, LP )

  GAP = 100.0
  It = 0
  S_LB_Ini = dH.S_LB

  if parallel
    LB = SharedArray(Float64,dH.S_LB)
  else
    LB = Array(Float64,dH.S_LB)
  end
  UB = 9999999.0
  it_stable = 0
  list_α = Array{Float64,2}[]
  list_β = Array{Float64,3}[]
  UB_last = 9999999.0
  LB_conserv = 0.0
  while abs(GAP) > dH.GAPP && UB > 1e-5
    It += 1
    tic()
    for s_f = 1:dH.S_FB
      # Forward
      debug("Forward Step")
      simulatestates(dH, dM, K_forward, r_forward)
      x_trial, x0_trial, FO_forward, u_trial = forward(dH, dM, AQ, sp, K_forward, r_forward)

      # Backward
      debug("Backward Step")
      α, β = backward(dH, dM, AQ, sp, x_trial, x0_trial)
      push!(list_α,α)
      push!(list_β,β)


      #Evaluate upper bound
      status = solve(AQ[1,dM.k_ini])
      if status ≠ :Optimal
        writeLP(AQ[1,dM.k_ini],"prob.lp")
        error("Can't solve the problem status:",status)
      end
      UB = getObjectiveValue(AQ[1,dM.k_ini])
      debug("FO_forw = $FO_forward, UB = $UB, stabUB $(abs(UB/UB_last -1)*100)")
      if abs(UB/UB_last -1)*100 < 1
        it_stable += 1
        if it_stable >= 5
          if dH.S_LB < 100*S_LB_Ini
            dH.S_LB = round(Int64,dH.S_LB*1.2)
            info("Increasing S_LB for $(dH.S_LB)")
            if parallel
              LB = SharedArray(Float64,dH.S_LB)
            else
              LB = Array(Float64,dH.S_LB)
            end
            break
          end
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
    debug("Evaluating the Lower Bound")
    LB_conserv = 0
    GAP = 100
    for s_f = 1:dH.S_LB #@sync @parallel TODO parallel not working
      simulatestates(dH, dM, K_forward, r_forward)
      x_trial, x0_trial, FO_forward, u_trial = forward(dH, dM, AQ, sp, K_forward, r_forward)
      LB[s_f] = FO_forward
      if s_f >= S_LB_Ini
        LB_conserv = (mean(LB[1:s_f]) - quantile(Normal(),dH.α_lB) * std(LB[1:s_f])/sqrt(s_f))
        GAP = 100*(UB - LB_conserv)/UB
        if GAP < dH.GAPP
          LB = LB[1:s_f] # only return the LB that were used
          info("GAP LB $GAP is lower then $(dH.GAPP)")
          break
        end
      end
    end

    time_it = toq()
    info("LB = $LB_conserv, UB = $UB, GAP(%) = $GAP, Time_it: $time_it")

    if It > dH.Max_It
      info("Maximum number of iterations exceeded")
      break
    end
  end

  if simuLB
    file = jldopen("./output/allo_data_G$(string(dH.γ)[3:end])_C$(string(dH.c)[3:end]).jld", "w")
    debug("Evaluating the Lower Bound")
    K_forward_o = Array(Int64,dH.T);
    r_forward_o = zeros(dH.N,dH.T);
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
  return LB, UB, LB_conserv, AQ, sp, list_α, list_β, vcat(x0_trial',x_trial), u_trial
end

function changeP_j(dH::MSDDPData, dM::HMMData, Q::Model, subp::SubProbData, p_state)
  chgConstrRHS(Q, subp.risco, dH.M ) # Disable the constraint
  chgConstrRHS(Q, subp.caixa, dH.x0_ini)
  for i = 1:dH.N
    chgConstrRHS(Q, subp.ativos[i], dH.x_ini[i])
  end
  y = getVar(Q,:y)
  b = getVar(Q,:b)
  d = getVar(Q,:d)
  z = getVar(Q,:z)
  u = getVar(Q,:u)
  θ = getVar(Q,:θ)
  @addConstraint(Q,-(z - sum{p_state[j]*dM.ps_j[s,j]*y[j,s] , j = 1:dH.K, s = 1:dH.S}/(1-dH.α))
                          + dH.c*sum{b[i] + d[i], i = 1:dH.N} <= dH.γ*(sum(dH.x_ini)+dH.x0_ini)).idx

  @setObjective(Q, Max, - dH.c*sum{b[i] + d[i], i = 1:dH.N} +
                  + sum{(sum{dM.r[i,j,s]*u[i], i = 1:dH.N} + θ[j,s])*p_state[j]*dM.ps_j[s,j], j = 1:dH.K, s = 1:dH.S})
end

function simulate_stateprob(dH::MSDDPData, dM::HMMData, AQ::Array{Model,2},sp::Array{SubProbData,2},
    ret_test::Array{Float64,2}, pk_r::Array{Float64,2}; real_tc=0.0)

   K_forward = Array(Int64,dH.T);
   K_forward[1] = dM.k_ini
   for t = 1:dH.T
     K_forward[t] = findmax(pk_r[:,t-1])[2]
   end
   return simulate(dH, dM, AQ, sp, ret_test, K_forward , x_ini, x0_ini, real_tc=real_tc)
end

function simulate(dH::MSDDPData, dM::HMMData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
    ret_test::Array{Float64,2}, k_test::Array{Int64,1}; real_tc=0.0)
  if size(ret_test,2) != dH.T-1
    error("Return series has to have $(dH.T-1) samples use simulatesw.")
  end

  K_forward = Array(Int64,dH.T);
  r_forward = zeros(dH.N,dH.T);

  r_forward[:,2:dH.T] = ret_test

  K_forward[1] = dM.k_ini
  K_forward[2:dH.T] = k_test[1:end]
  x, x0, exp_ret, u = forward(dH, dM, AQ, sp, K_forward, r_forward, real_tc=real_tc)

  return x, x0, exp_ret
end
# Simulate using sliging windows
function simulatesw(dH::MSDDPData, dM::HMMData, AQ::Array{Model,2}, sp::Array{SubProbData,2},
  ret_test::Array{Float64,2}, k_test::Array{Int64,1}; real_tc=0.0)
   dH_ = deepcopy(dH)
   init_T = dH_.T
   T_test = size(ret_test,2)
   its=floor(Int,T_test/(dH_.T-1))

   all_x = dH_.x_ini
   all_x0 = dH_.x0_ini

   r_forward = zeros(dH_.N,dH_.T)
   K_forward_a = Array(Int64,dH_.T)
   K_forward_a[1] = dM.k_ini
   for i = 1:its
     r_forward[:,2:dH_.T] = ret_test[:,(i-1)*(dH_.T-1)+1:(i)*(dH_.T-1)]
     K_forward_a[2:dH_.T] = k_test[(i-1)*(dH_.T-1)+1:(i)*(dH_.T-1)]
     x, x0, expret, u = forward(dH_, dM, AQ, sp, K_forward_a, r_forward, real_tc=real_tc)
     all_x = hcat(all_x,x[:,2:end])
     all_x0 = vcat(all_x0,x0[2:end])
     dH_.x_ini = x[:,end]
     dH_.x0_ini = x0[end]
     K_forward_a[1] = K_forward_a[end]
   end

   # Run for the last periods
   diff_t = round(Int, T_test - its*(dH_.T-1)+1)
   if diff_t > 0
     r_forward_a = zeros(dH_.N,diff_t)
     K_forward_a = Array(Int64,diff_t)
     r_forward_a[:,2:diff_t] = ret_test[:,its*(dH_.T-1)+1:end]
     K_forward_a[1] = dM.k_ini
     K_forward_a[2:diff_t] = k_test[its*(dH_.T-1)+1:end]
     dH_.T = diff_t
     x, x0, expret, u = forward(dH_, dM, AQ, sp, K_forward_a, r_forward_a, real_tc=real_tc)
     dH_.T = init_T
     all_x = hcat(all_x,x[:,2:end])
     all_x0 = vcat(all_x0,x0[2:end])
   end

   return all_x, all_x0
 end

function simulate_percport(dH::MSDDPData, ret_test::Array{Float64,2}, x_ini::Array{Float64,1},
    x_p::Array{Float64,1})
   T_test = size(ret_test,2)
   x = Array(Float64,dH.N+1, T_test+1)
   x[:,1] = x_ini
   for t = 2:T_test+1
     for i = 2:dH.N+1
       x[i,t] = (1+ret_test[i-1,t-1])*x[i,t-1]
     end
     x[1,t] = x[1,t-1]
     total = sum(x[:,t])

     # Ajust the portfolio and discont transactional costs
     cost = 0.0
     for i = 2:dH.N+1
       cost += abs(x[i,t]-total*x_p[i])*dH.c
       x[i,t] = total*x_p[i]
     end
     x[1,t] = total*x_p[1] - cost
   end
   return x
 end

end # SDDP Module
