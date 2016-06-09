using MSDDP, HMM_MSDDP, AR
using Distributions, Logging, JuMP
Logging.configure(level=Logging.DEBUG)

srand(123)
Sc = 1000
T_l = 120

# Factors
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 0.00042
dF = Factors(a_z, a_r, b_z, b_r, Σ, r_f)

# Parmeters
N = 3
T = 13
K = 4
S = 500
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.002
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
file_name = "$(N)MS_120_$(Sc)"

#########################################################################################################################
tic()
γs = [0.05,0.1, 0.2,0.3] #0.6, 0.4,0.3,0.2, 0.1, 0.08, 0.05, 0.02, 0.01]
cs = [0.005,0.01,0.02] #0.005, 0.01, 0.012, 0.013, 0.014, 0.015, 0.017, 0.02]
Ts = [12,24,48]

# Return series data
output_dir = "../../output3/"
input_dir = "../../input/"
input_file = string(input_dir, file_name, ".csv")
ln_ret = readcsv(input_file, Float64)
ln_ret = reshape(ln_ret, N+1, T_l, Sc)
ln_ret = ln_ret[1:N,:,:]
ret = exp(ln_ret[1:N,:,:])-1

# HMM data
info("Train HMM")
dM, model, y = inithmm_onefactor(ln_ret, dF, dH, T_l, Sc)
states = Array(Int64,Ts[end],Sc)
for s=1:Sc
  states[:,s] = predict(model,y[:,1:Ts[end],s]')
end
ret_p = Array(Float64,3,Sc)
for dH.T in Ts
  rets = zeros(3,length(γs),length(cs))
  file = string(output_dir,file_name,"_$(dH.T)_table.csv")
  for i_γ = 1:length(γs)
    dH.γ = γs[i_γ]
    for i_c = 1:length(cs)
      info("Start testes with γ = $(dH.γ), c = $(dH.c) and T = $(dH.T)")
      dH.c = cs[i_c]
      dH_o = deepcopy(dH) # Store the original

      ############ MSDDP ###########
      info("#### MSDDP ####")
      info("Training MSDDP...")
      @time LB, UB, LB_c, AQ, sp = sddp(dH, dM)

      info("Simulating MSDDP...")
      for s=1:Sc
        x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret[1:dH.N,1:dH.T-1,s], states[1:dH.T-1,s])
        ret_p[1,s] = x0[end]+sum(x[:,end])-1
      end
      ############ One Step ############
      dH.T = 2
      info("#### One Step ####")
      info("Training One Step...")
      @time LB, UB, LB_c, AQ, sp = sddp(dH, dM)

      info("Simulating One Step...")
      for s=1:Sc
        x, x0 = simulatesw(dH, dM, AQ, sp, ret[1:dH.N,1:dH_o.T-1,s], states[1:dH_o.T-1,s])
        ret_p[2,s] = x0[end]+sum(x[:,end])-1
      end
      dH.T = dH_o.T

      ############ One Step with future cost ############
      # Run MSDDP with no costs
      info("#### One Step FC ####")
      info("Training One Step FC...")
      dH.c = 0
      @time LB, UB, LB_c, AQ, sp, list_α, list_β = sddp(dH, dM)
      # using the future value function without cost to one-step
      dH.c = dH_o.c
      AQ, sp = createmodels( dH, dM, 2 )
      for cuts = 1:length(list_α)
        α = list_α[cuts]
        β = list_β[cuts]
        for t = 1:dH.T-1
          for k = 1:dH.K
            Q = AQ[t,k]
            θ = getvariable(Q,:θ)
            u = getvariable(Q,:u)
            u0 = getvariable(Q,:u0)
            @constraint(Q,corte_js[j = 1:dH.K, s = 1:dH.S],
                θ[j,s] <= α[t+1,j] + β[1,t+1,j]*u0 + sum{β[i+1,t+1,j]*(1+dM.r[i,j,s])*u[i], i = 1:dH.N})
          end
        end
      end
      info("Simulating One Step FC...")
      for s = 1:Sc
        x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret[1:dH.N,1:dH.T-1,s], states[1:dH.T-1,s])
        ret_p[3,s] = x0[end]+sum(x[:,end])-1
      end

      for i = 1:3
        rets[i,i_γ,i_c] = mean(ret_p[i,:])
      end
      open(file,"a") do x
        writecsv(x,hcat(dH.c, dH.γ,rets[1,i_γ,i_c],rets[2,i_γ,i_c],rets[3,i_γ,i_c]))
      end
    end
  end
end
toc()

run(`/home/tas/woofy.sh 62491240 "onefactor"`)
