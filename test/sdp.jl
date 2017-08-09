using Base.Test
using SDP, AR

@testset "SDP" begin
    srand(123)
    T_s = 240
    N = 3
    Sc = 1000
    T_l = 120

    # AR
    Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
    b_r = [ 0.0028; 0.0049; 0.0061]
    b_z = [0.9700]
    a_r  = [0.0053; 0.0067; 0.0072]
    a_z  = [0.0000]
    r_f = 0.00042

    dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)

    # Generate the series ρ and z
    se_ = series(dF, Sc, T_s)
    ρ = se_[1:N,:,:]
    rets = exp(ρ)-1 -dF.r_f
    z = reshape(se_[N+1,:,:], T_s, Sc)


    # Parameters SDP
    T = 12
    L = 1000
    S = 500
    α = 0.9
    γ = 0.012

    dS = SDPData(N, T, L, S, α, γ)

    # Split z
    z_l = splitequaly(dS.L, z)

    # Run SDP
    Q_l = backward(dF, dS, z_l)
    ws = zeros(Float64, Sc)
    all = Array(Float64, N+1,T-1, Sc)
    for se = 1:Sc
      ws[se], all[:,:,se] = forward(dS, dF, Q_l, z_l, vec(z[:,se]), rets[:,:,se])
    end


    @test isapprox( mean(ws), 1.01161; atol= 1e-4)
    @test isapprox( std(ws), 0.02505; atol= 1e-4)
    @test isapprox( mean(all[1,:,:]), 0.893047 ; atol= 1e-4)
    @test isapprox( mean(all[4,:,:]), 0.0378636; atol= 1e-4)
end
