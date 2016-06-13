using Base.Test
using SDP, AR

srand(123)
T_s = 240
N = 3
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

# Generate the series ρ and z
se_ = series(dF, Sc, T_l, T_s)
ρ = se_[1:N,:,:]
rets = exp(ρ)-1
z = reshape(se_[N+1,:,:], T_l, Sc)


# Parameters SDP
T = 12
L = 1000
S = 500
α = 0.9
γ = 0.012

dS = SDPData(N, T, L, S, α, γ)

# Split z
z_l = splitequaly(dS, z)

# Evalute the probability
p = evaluateprob(dS, z, z_l, T_l, Sc)

# Run SDP
u_l, Q_l = backward(dF, dS, z_l, p)
ws = Array(Float64, Sc)
for se = 1:Sc
  ws[se] = forward(dS, u_l, z_l, z[:,se], rets[:,:,se])
end


@test_approx_eq_eps mean(ws) 0.8598176591615809 1e-6
@test_approx_eq_eps std(ws) 0.3467966766360369 1e-6
