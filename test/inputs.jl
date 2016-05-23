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
x_ini = zeros(N)
x0_ini = 1
c = 0.00
M = 9999999
γ = 0.1
S_LB = 1000
S_FB = 1
GAPP = 1
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

#Run the code to output HMM/LHS data
file = "./test_inputs.csv"
ret = readcsv(file, Float64)

dM = inithmm(ret, dH)


LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.0937307 1e-4
@test_approx_eq_eps UB       0.0940507 1e-4

# Changing some parameters
info("Test with same base changing some parameters")
dH.γ = 0.01
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.015342 1e-4
@test_approx_eq_eps UB       0.015342 1e-4

dH.α = 0.90
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.0165660 1e-4
@test_approx_eq_eps UB       0.0165642 1e-4

dH.T = 2
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.0018270 1e-4
@test_approx_eq_eps UB       0.0018270 1e-4

# Changing the number of states
dH.K = 3
dM = inithmm(ret, dH)

LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.002502 1e-4
@test_approx_eq_eps UB       0.002502 1e-4

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.122333 1e-2
@test_approx_eq_eps UB       0.122204 1e-2
