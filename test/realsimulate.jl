using HMM_MSDDP
using Base.Test

info("Test with specific base")
srand(123)
N = 5
T = 11
K = 1
S = 100
α = 0.95
x_ini = zeros(N)
x0_ini = 1.0
c = 0.00
M = 9999999
γ = 0.1
S_LB = 1000
S_FB = 1
GAPP = 1
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

start_train = 1
n_rows_train = 100
n_rows_test = 5
file = "./test_inputs.csv"
ret = readcsv(file, Float64)
ret_train = ret[start_train:start_train+n_rows_train-1,:]
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test-1,:]

dM, model = inithmm(ret_train, dH)

LB, UB, LB_c, AQ, sp = sddp(dH, dM)

k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
# With one state has to invest everything on the asset with the best profit (second asset)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 1.10306 5e-3

# With bigger test ret
n_rows_test = 10
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test-1,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 1.10306 5e-3
@test_approx_eq_eps x[2,11] 1.07393 5e-3

# the results should be equal to simulate
x_sw, x0_sw = simulate(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x_sw x 1e-6
@test_approx_eq_eps x0_sw x0 1e-6

n_rows_test = 15
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test-1,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 1.10306 5e-3
@test_approx_eq_eps x[2,11] 1.07393 5e-3
@test_approx_eq_eps x[2,16] 1.22899 5e-3

n_rows_test = 40
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test-1,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps sum(x0[2:end]) 0 1e-6
@test_approx_eq_eps sum(x[collect(1:5) .!= 2,:]) 0 1e-6
@test_approx_eq_eps x[2,6] 1.10306 5e-3
@test_approx_eq_eps x[2,11] 1.07393 5e-3
@test_approx_eq_eps x[2,16] 1.22899 5e-3
@test_approx_eq_eps x[2,41] 1.771 5e-3

# Back to test with 5 samples
n_rows_test = 5
ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test-1,:]
k_test = predict(model,ret_test)
ret_test = exp(ret_test)-1
dH.T = 5

# Changing the CVaR limit
dH.γ = 0.01
LB, UB, LB_c, AQ, sp = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0,0.8860859495014164,0.8885892029633965,0.8867773149286607,0.8922322003016439,0.8958911605188956] 1e-4
@test_approx_eq_eps sum(x,1) [0.0 0.11673911895090322 0.11219104349873663 0.12015908983397636 0.11883355581715754 0.11560074159927679] 1e-4

dH.α = 0.90
LB, UB, LB_c, AQ, sp = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test', k_test)
@test_approx_eq_eps x0 [1.0,0.8662482911536769,0.8691216734464607,0.8670408609371251,0.873303143561438,0.8775081519174449] 1e-4
@test_approx_eq_eps sum(x,1) [0.0 0.1370687512256588 0.13179327183140258 0.1411032848924331 0.13969527834456313 0.13599158398239827] 1e-4

# Changing the number of states
dH.K = 3
dM, model = inithmm(ret_train, dH)

LB, UB, LB_c, AQ, sp = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test, k_test)
@test_approx_eq_eps x0 [1.0,1.0,0.8374101083489728,0.8422626275774806,0.8487339769296859,1.018401902080441] 1e-3
@test_approx_eq_eps sum(x,1) [0.0 0.0 0.1683845664228364 0.17125986094733442 0.16966792515075504 0.0] 1e-3

dM.k_ini = 2
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test, k_test)
@test_approx_eq_eps x0 [1.0,0.8827963133481778,0.8362359989782762,0.8410817146248343,0.8475439906784971,0.8977809237139723] 1e-3
@test_approx_eq_eps sum(x,1) [0.0 0.11580161452612095 0.16814847911585665 0.17101974227004868 0.1694300384822756 0.11921830049210108] 1e-3

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
LB, UB, LB_c, AQ, sp = sddp(dH, dM)
x, x0 = simulatesw(dH, dM, AQ, sp, ret_test, k_test)
@test_approx_eq_eps x0 [1.0,0.04548421573744038,0.0,0.0,0.0,0.0] 1e-3
@test_approx_eq_eps sum(x,1) [0.0 0.9442376918146463 1.0253518961857253 1.0775423077129858 1.0883177308410836 1.1574259067979251] 1e-3
