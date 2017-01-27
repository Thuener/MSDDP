using MSDDP

Logging.configure(level=Logging.DEBUG)

#Parameters
N = 100
T = 12
K = 3
S = 1000
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.01
M = 9999999
γ = 0.05
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
