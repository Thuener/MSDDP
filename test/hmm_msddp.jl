using Base.Test
using HMM_MSDDP

srand(12345)
T_s = 240
N = 3
Sc = 10
T_l = 120

N = 3
T = 13
K = 3
S = 500
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.012
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

γ = 0.003
c = 0.01

dH  = MSDDPData( N+1, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

file = "./test_hmm_msddp.csv"
ret = readcsv(file, Float64)
ret = reshape(ret, dH.N, T_l, Sc)
ret = reshape(ret, dH.N, T_l*Sc)

dM = inithmm(ret', dH, T_l, Sc)

order = Array(Int64,dH.K)
found = true
for k = 1:dH.K
  if abs(mean(reshape(dM.r[1,k,:],dH.S)) - 0.00839532360645153) < 1e-6
    order[1] = k
  elseif abs(mean(reshape(dM.r[1,k,:],dH.S)) - 0.002996800086701159) < 1e-6
      order[2] = k
    elseif abs(mean(reshape(dM.r[1,k,:],dH.S)) - 0.015196632580590699) < 1e-6
      order[3] = k
    else
      found = false
  end
end
# Has to enter in one if every time
@test found == true
@test_approx_eq_eps mean(reshape(dM.r[:,order[1],:],dH.N,dH.S),2) [0.00839532360645153 0.010430380640150642  0.012686062611771737 -0.13354913446138864] 1e-6
@test_approx_eq_eps mean(reshape(dM.r[:,order[2],:],dH.N,dH.S),2) [0.002996800086701159  0.005042811147145921 0.007042256239115454 -0.6852052890669273] 1e-6
@test_approx_eq_eps mean(reshape(dM.r[:,order[3],:],dH.N,dH.S),2) [0.015196632580590699 0.022208399618642474 0.024857176445799788   1.4464545530498576] 1e-6

@test_approx_eq_eps sum(dM.ps_j,1) [1 1 1] 1e-6

@test dM.k_ini == order[3]

@test_approx_eq_eps sum(dM.P_K,2) [1 1 1]'  1e-6
@test_approx_eq_eps dM.P_K[order[1],order[1]] 0.925952  1e-3
@test_approx_eq_eps dM.P_K[order[2],order[2]] 0.960826  1e-3
@test_approx_eq_eps dM.P_K[order[3],order[3]] 0.967949  1e-3
