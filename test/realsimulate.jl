using HMM_MSDDP, MSDDP
using Base.Test, CPLEX

@testset "Real" begin
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
    maxvl = 9999999
    γ = 0.1
    samplower = 1000
    samplower_inc = 100
    nit_before_lower = 1
    gap = 1.
    max_it = 100
    α_lower = 0.9

    ms = ModelSizes(T, N, K, S)

    start_train = 1
    n_rows_train = 100
    n_rows_test = 5
    file = "./test_inputs.csv"
    ret = readcsv(file, Float64)
    ret_train = ret[start_train:start_train+n_rows_train-1,:]
    ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]

    mk, hmm = inithmm(ms, ret_train)

    m = MSDDPModel(MAAParameters(α, γ, c, x_ini, x0_ini, maxvl),
        SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower),
        mk;
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))

    LB, UB, LB_c, AQ, sp, = solve(m, m.param)

    k_test = predict(hmm, ret_test)
    ret_test = exp(ret_test)-1
    x, x0 = simulatesw(m, ret_test', k_test)
    # With one state has to invest everything on the asset with the best profit (second asset)
    @test isapprox( sum(x0[2:end]), 0; atol= 1e-6)
    @test isapprox( sum(x[collect(1:5) .!= 2,:]), 0; atol= 1e-6)
    @test isapprox( x[2,6], 0.964855; atol= 5e-3)

    # With bigger test ret
    n_rows_test = 10
    ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
    k_test = predict(hmm, ret_test)
    ret_test = exp(ret_test)-1
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( sum(x0[2:end]), 0; atol= 1e-6)
    @test isapprox( sum(x[collect(1:5) .!= 2,:]), 0; atol= 1e-6)
    @test isapprox( x[2,6], 0.964855; atol= 5e-3)
    @test isapprox( x[2,11], 1.12171; atol= 5e-3)

    # the results should be equal to simulate
    x_sw, x0_sw = simulate(m, ret_test', k_test)
    @test isapprox( x_sw, x; atol= 1e-6)
    @test isapprox( x0_sw, x0; atol= 1e-6)

    n_rows_test = 15
    ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
    k_test = predict(hmm, ret_test)
    ret_test = exp(ret_test)-1
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( sum(x0[2:end]), 0; atol= 1e-6)
    @test isapprox( sum(x[collect(1:5) .!= 2,:]), 0; atol= 1e-6)
    @test isapprox( x[2,6], 0.964855; atol= 5e-3)
    @test isapprox( x[2,11], 1.12171; atol= 5e-3)
    @test isapprox( x[2,16], 1.27935; atol= 5e-3)

    n_rows_test = 40
    ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
    k_test = predict(hmm, ret_test)
    ret_test = exp(ret_test)-1
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( sum(x0[2:end]), 0; atol= 1e-6)
    @test isapprox( sum(x[collect(1:5) .!= 2,:]), 0; atol= 1e-6)
    @test isapprox( x[2,6], 0.964855; atol= 5e-3)
    @test isapprox( x[2,11], 1.12171; atol= 5e-3)
    @test isapprox( x[2,16], 1.27935; atol= 5e-3)
    @test isapprox( x[2,41], 1.5701; atol= 5e-3)

    # Back to test with 5 samples
    n_rows_test = 5
    ret_test  = ret[start_train+n_rows_train:start_train+n_rows_train+n_rows_test,:]
    k_test = predict(hmm, ret_test)
    ret_test = exp(ret_test)-1
    setnstages!(m, 5)

    # Changing the CVaR limit
    setγ!(m, 0.0000001)
    reset!(m)
    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, ones(6); atol= 1e-4)
    @test isapprox( sum(x,1), zeros(6)'; atol= 1e-4)

    setγ!(m, 0.01)
    reset!(m)
    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0,0.8860859495014164,0.8842791657544444,0.8897186841157093,0.8933673366398218,0.8937438748760086]; atol= 1e-4)
    @test isapprox( sum(x,1), [0.0 0.11187498899671838 0.11982058846955264 0.11849878862776717 0.11527508160290185 0.10299505426932623]; atol= 1e-4)

    setα!(m, 0.90)
    reset!(m)
    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0,0.8662482911536769,0.864174357968608,0.8704159370107242,0.8746070432895877,0.8750398699756242]; atol= 1e-4)
    @test isapprox( sum(x,1), [0.0 0.13135755326039203 0.14063678671081967 0.1392334351395846 0.13554198547240695 0.12111184993069664]; atol= 1e-4)

    # Changing the number of states
    srand(1234)
    setnstates!(m, 3)
    mk, hmm = inithmm(m.sizes, ret_train)
    setmarkov!(m, mk)
    reset!(m)

    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0,0.842685,0.854163,0.83551,0.836874,0.855227]; atol= 1e-3)
    @test isapprox( sum(x,1), [0.0 0.158198 0.159443 0.181837 0.178009 0.141161]; atol= 1e-3)

    setinistate!(markov(m), 2)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0,0.842685,0.854163,0.83551,0.836874,0.855227]; atol= 1e-3)
    @test isapprox( sum(x,1), [0.0 0.158198 0.159443 0.181837 0.178009 0.141161]; atol= 1e-3)

    setγ!(m, 0.1)
    setα!(m, 0.95)
    setnstages!(m, 10)
    r = zeros(nstages(m), nassets(m), nstates(m), nscen(m))
    for t = 1:nstages(m)
      r[t,:,:,:] = mk.ret[1,:,:,:]
    end
    mk.ret = r
    reset!(m)
    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0 0.0 0.0 0.0 0.0 0.0]'; atol= 1e-3)
    @test isapprox( sum(x,1), [0.0 1.011 1.10068 1.11168 1.08111 0.942297]; atol= 1e-3)

    setnstages!(m, 2)
    reset!(m)
    LB, UB, LB_c, AQ, sp, = solve(m, m.param)
    x, x0 = simulatesw(m, ret_test', k_test)
    @test isapprox( x0, [1.0 0.0 0.0 0.0 0.0 0.0]'; atol= 1e-3)
    @test isapprox( sum(x,1), [0.0  1.011  1.10068  1.11168  1.08111  0.942297]; atol= 1e-3)
end
