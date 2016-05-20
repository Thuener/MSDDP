push!(LOAD_PATH, "../")
using H2SDDP
using Distributions
using Base.Test

## Teste para uma distribuição aleatória
Logging.info("Test with specific distribution")
srand(123)
N = 5
μ = [0.11 0.12 0.13 0.14 0.15]'
Σ = [0.01 0 0 0 0;
     0 0.015 0 0 0;
     0 0 0.02 0 0;
     0 0 0 0.025 0;
     0 0 0 0 0.03];
T = 10
K = 1
S = 100
r = zeros(N,K,S)
srand(123)
r[:,1,:] = μ[1:N]*ones(1,S) + rand(MvNormal(Σ[1:N,1:N]),S)
#r = μ[1:N]*ones(1,S) + rand(MvNormal(Σ[1:N,1:N]),S)
α = 0.95
x_ini = zeros(N)
x0_ini = 1.0
c = 0
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parâmetros
S_LB = 1000
S_FB = 5
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

# Gerando aleatoriamente probabilidades (condicionais a cada estado) para cada cenario
p = rand(Uniform(),S,K)
p = p./(ones(S)*sum(p,1))
# prob iniciais geradas aleatoriamente
prob_ini = rand(Uniform(),K)
prob_ini = prob_ini/sum(prob_ini)
# Matriz de transicao aleatoria (K_t x K_(t+1))
P_K = rand(Uniform(),K,K)
P_K = P_K./(sum(P_K,2)*ones(1,K))

dM = HMMData( r, p, prob_ini, P_K )

LB, UB = SDDP(dH, dM)
@test std(LB) ≈ 0.5958493114217336
@test mean(LB) ≈ 2.584326371598783
@test UB ≈ 2.5807450358011

## Teste para uma distribuição aleatória com parâmetros diferentes
Logging.info("Test same distribution but with different parameters")
K = 3
c = 0.01
γ = 0.1
r = zeros(N,K,S)
srand(123)
r[:,1,:] = μ[1:N]*ones(1,S) + rand(MvNormal(Σ[1:N,1:N]),S)
r[:,2,:] = r[:,1,:]
r[:,3,:] = r[:,1,:]


dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

# Gerando aleatoriamente probabilidades (condicionais a cada estado) para cada cenario
p = rand(Uniform(),S,K)
p = p./(ones(S)*sum(p,1))
# prob iniciais geradas aleatoriamente
prob_ini = rand(Uniform(),K)
prob_ini = prob_ini/sum(prob_ini)
# Matriz de transicao aleatoria (K_t x K_(t+1))
P_K = rand(Uniform(),K,K)
P_K = P_K./(sum(P_K,2)*ones(1,K))

dM = HMMData( r, p, prob_ini, P_K )

LB, UB, cuts = SDDP(dH, dM)
@test std(LB) ≈ 0.5563595900647895
@test mean(LB) ≈ 2.646568907544095
@test UB ≈ 2.6346509717728837
