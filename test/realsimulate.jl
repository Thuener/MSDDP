using HMM_MSDDP
using Base.Test

info("Test with specific base")
srand(123)
N = 5
T = 11
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

start_train = 1
n_rows_train = 100
n_rows_test = 5
file = "./test_inputs.csv"
ret = readcsv(file, Float64)
ret_train = ret[start_train:start_train+n_rows_train-1,:]
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]

dM, model = inithmm(ret_train, dH)

LB, UB, LB_c, AQ, sp, = sddp(dH, dM)

k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
# With one state has to invest everything on the asset with the best profit (second asset)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 0.964855 5e-3

# With bigger test ret
n_rows_test = 10
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 0.964855 5e-3
@test_approx_eq_eps x[2,11] 1.12171 5e-3

# the results should be equal to simulate
x_sw, x0_sw = simulate(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x_sw x 1e-6
@test_approx_eq_eps x0_sw x0 1e-6

n_rows_test = 15
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 0.964855 5e-3
@test_approx_eq_eps x[2,11] 1.12171 5e-3
@test_approx_eq_eps x[2,16] 1.27935 5e-3

n_rows_test = 40
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 0.964855 5e-3
@test_approx_eq_eps x[2,11] 1.12171 5e-3
@test_approx_eq_eps x[2,16] 1.27935 5e-3
@test_approx_eq_eps x[2,41] 1.5701 5e-3

# Back to test with 5 samples
n_rows_test = 5
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
dH.T = 5

# Changing the CVaR limit
dH.γ = 0.01
LB, UB, LB_c, AQ, sp, = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0,0.8860859495014164,0.8842791657544444,0.8897186841157093,0.8933673366398218,0.8937438748760086] 1e-4
@test_approx_eq_eps sum(x,1) [0.0 0.11187498899671838 0.11982058846955264 0.11849878862776717 0.11527508160290185 0.10299505426932623] 1e-4

dH.α = 0.90
LB, UB, LB_c, AQ, sp, = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0,0.8662482911536769,0.864174357968608,0.8704159370107242,0.8746070432895877,0.8750398699756242] 1e-4
@test_approx_eq_eps sum(x,1) [0.0 0.13135755326039203 0.14063678671081967 0.1392334351395846 0.13554198547240695 0.12111184993069664] 1e-4

# Changing the number of states
dH.K = 3
dM, model = inithmm(ret_train, dH)

LB, UB, LB_c, AQ, sp, = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0 0.82014 0.845609 0.837851 0.857162 0.830954] 1e-3
@test_approx_eq_eps sum(x,1) [0.0 0.180077 0.167223 0.179511 0.156024 0.161469] 1e-3

dM.k_ini = 2
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0  0.82014  0.845609  0.837851  0.857162  0.830954] 1e-3
@test_approx_eq_eps sum(x,1) [0.0  0.180077  0.167223  0.179511  0.156024  0.161469] 1e-3

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
r = zeros(dH.T, dH.N, dH.K, dH.S)
for t = 1:dH.T
  r[t,:,:,:] = dM.r[1,:,:,:]
end
dM.r = r
LB, UB, LB_c, AQ, sp, = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0 0.0 0.0 0.0 0.0 0.0] 1e-3
@test_approx_eq_eps sum(x,1) [0.0 1.011 1.10068 1.11168 1.08111 0.942297] 1e-3

dH.T = 2
LB, UB, LB_c, AQ, sp, = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0 0.0 0.0 0.0 0.0 0.0] 1e-3
@test_approx_eq_eps sum(x,1) [0.0  1.011  1.10068  1.11168  1.08111  0.942297] 1e-3
