using HMM_MSDDP, MSDDP
using Distributions, Base.Test, CPLEX

@testset "Inputs" begin
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
    maxvl = 9999999
    γ = 0.1
    samplower = 1000
    samplower_inc = 100
    nit_before_lower = 1
    gap = 2.
    max_it = 100
    α_lower = 0.9

    ms = ModelSizes(T, N, K, S)

    #Run the code to output HMM/LHS data
    file = "./test_inputs.csv"
    ret = readcsv(file, Float64)

    mk, hmm = inithmm(ms, ret)

    m = MSDDPModel(MAAParameters(α, γ, c, x_ini, x0_ini, maxvl),
        SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower),
        mk;
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))


    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.0943571; atol= 1e-4)
    @test isapprox( UB      , 0.0940507; atol= 1e-4)

    # Changing some parameters
    info("Test with same base changing some parameters")
    setγ!(m, 0.01)
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.012397; atol= 1e-4)
    @test isapprox( UB      , 0.012393; atol= 1e-4)

    setα!(m, 0.90)
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.014320; atol= 1e-4)
    @test isapprox( UB      , 0.014329; atol= 1e-4)

    setnstages!(m, 2)
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.0015821; atol= 1e-4)
    @test isapprox( UB      , 0.0015821; atol= 1e-4)

    # Changing the number of states
    setnstates!(m, 3)
    mk, hmm = inithmm(m.sizes, ret)

    order = Array(Int64, nstates(m))
    found = true
    mm = mean(hmm[:means_],2)
    for k = 1:nstates(m)
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

    setmarkov!(m, mk)
    setinistate!(markov(m), order[1])
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.002671; atol= 1e-4)
    @test isapprox( UB      , 0.002671; atol= 1e-4)

    setinistate!(markov(m), order[2])
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.; atol= 1e-4)
    @test isapprox( UB      , 0.; atol= 1e-4)

    setinistate!(markov(m), order[3])
    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.0027506; atol= 1e-4)
    @test isapprox( UB      , 0.0027506; atol= 1e-4)

    setγ!(m, 0.1)
    setα!(m, 0.95)
    setnstages!(m, 10)
    setinistate!(markov(m), order[2])
    r = zeros(nstages(m), nassets(m), nstates(m), nscen(m))
    for t = 1:nstages(m)
      r[t,:,:,:] = mk.ret[1,:,:,:]
    end
    mk.ret = r

    reset!(m)
    LB, UB = solve(m, m.param)
    @test isapprox( mean(LB), 0.04428 ; atol= 1e-2)
    @test isapprox( UB      , 0.04243 ; atol= 1e-2)
end
