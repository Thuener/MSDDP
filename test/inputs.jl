using HMM_MSDDP
using Distributions
using Base.Test

info("Test with specific base")
srand(123)
N = 5
T = 10
K = 1
S = 100
α = 0.95
W_ini = 1.0
x_ini = zeros(N)
x0_ini = W_ini
c = 0.00
M = 9999999
γ = 0.1
S_LB = 1000
S_LB_inc = 100
S_FB = 1
GAPP = 1
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, W_ini, c, M, γ,
                S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )

#Run the code to output HMM/LHS data
file = "./test_inputs.csv"
ret = readcsv(file, Float64)

dM, m = inithmm(ret, dH)


LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.0943571 1e-4
@test_approx_eq_eps UB       1.0940507 1e-4

# Changing some parameters
info("Test with same base changing some parameters")
dH.γ = 0.01
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.012397 1e-4
@test_approx_eq_eps UB       1.012393 1e-4

dH.α = 0.90
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.014320 1e-4
@test_approx_eq_eps UB       1.014329 1e-4

dH.T = 2
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.0015821 1e-4
@test_approx_eq_eps UB       1.0015821 1e-4

# Changing the number of states
dH.K = 3
dM, m = inithmm(ret, dH)

order = Array(Int64,dH.K)
found = true
mm = mean(m[:means_],2)
for k = 1:dH.K
  if abs(mm[k] - 0.007) < 1e-3
    order[1] = k
  elseif abs(mm[k] + 0.012) < 1e-3
      order[2] = k
    elseif abs(mm[k] - 0.021) < 1e-3
      order[3] = k
    else
      found = false
  end
end
# Has to enter in one if every time
@test found == true

dM.k_ini = order[1]
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.002671 1e-4
@test_approx_eq_eps UB       1.002671 1e-4

dM.k_ini = order[2]
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1 1e-4
@test_approx_eq_eps UB       1 1e-4

dM.k_ini = order[3]
LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.0027506 1e-4
@test_approx_eq_eps UB       1.0027506 1e-4

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
dM.k_ini = order[2]
r = zeros(dH.T, dH.N, dH.K, dH.S)
for t = 1:dH.T
  r[t,:,:,:] = dM.r[1,:,:,:]
end
dM.r = r

LB, UB = sddp(dH, dM)
@test_approx_eq_eps mean(LB) 1.04428 1e-2
@test_approx_eq_eps UB       1.04243 1e-2
