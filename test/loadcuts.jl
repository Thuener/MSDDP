using HMM_MSDDP, MSDDP
using Base.Test, CPLEX

@testset "LoadCuts" begin
    info("Test with specific base")
    srand(123)
    N = 5
    T = 5
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

    morig = MSDDPModel(MAAParameters(α, γ, c, x_ini, x0_ini, maxvl),
        SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower),
        mk;
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))

    LB, UB, LB_c, = solve(morig, param(morig); cutsfile="cuts.csv")

    m = MSDDPModel(MAAParameters(α, γ, c, x_ini, x0_ini, maxvl),
        SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower),
        mk;
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))
    loadcuts!(m, "cuts.csv")
    rm("cuts.csv")

    k_test = predict(hmm, ret_test)
    ret_test = exp.(ret_test)-1
    x, x0 = simulatesw(m, ret_test', k_test)
    # With one state has to invest everything on the asset with the best profit (second asset)
    @test isapprox( sum(x0[2:end]), 0; atol= 1e-6)
    @test isapprox( sum(x[collect(1:5) .!= 2,:]), 0; atol= 1e-6)
    @test isapprox( x[2,6], 0.964855; atol= 5e-3)

    # Changing the number of states
    setnstates!(morig, 3)
    mk, hmm = inithmm(morig.sizes, ret_train)
    setmarkov!(morig, mk)
    reset!(morig)

    LB, UB, LB_c, = solve(morig, param(morig); cutsfile="cuts.csv")
    x, x0 = simulatesw(morig, ret_test', k_test)
    m = MSDDPModel(MAAParameters(α, γ, c, x_ini, x0_ini, maxvl),
        SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower),
        mk;
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))
    loadcuts!(m, "cuts.csv")
    rm("cuts.csv")
    x_, x0_ = simulatesw(m, ret_test', k_test)
    @test isapprox( x0_, x0; atol= 1e-6)
    @test isapprox( x_, x; atol= 1e-6)

end
