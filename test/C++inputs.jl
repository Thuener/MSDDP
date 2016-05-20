push!(LOAD_PATH, "../")
using H2SDDP
using Distributions
using Base.Test

## Teste para uma distribuição específica
Logging.info("Test with ken_5MInd base")
srand(123)
N = 5
T = 10
K = 1
S = 100
α = 0.95
x_ini = zeros(N)
x0_ini = 1
c = 0.00
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 0.1
#Parâmetros
S_LB = 1000
S_FB = 1
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

#Run the C++ code to output HMM/LHS data
file_name = "ken_5MInd"
readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S`)

# HMM data
file = string("../C++/output/",file_name)
dM = readHMMPara(file, dH)


LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.095 1e-3
@test UB ≈ 0.09395446917685459

# Mudando alguns parâmetros
Logging.info("Test with ken_5MInd base changing some parameters")
dH.γ = 0.01
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.012 1e-3
@test UB ≈ 0.012520202010382406

dH.α = 0.90
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.014 1e-3
@test UB ≈ 0.014398067051775272

dH.T = 2
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.0015 1e-3
@test UB ≈ 0.0015896398277930022

# Mudando a quantidade de estados
dH.K = 3
readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name 3 $N $S`)
dM = readHMMPara(file, dH)

LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.002 1e-3
@test UB ≈ 0.0024893066023200477

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
LB, UB = SDDP(dH, dM)
@test_approx_eq_eps mean(LB) 0.121 1e-3
@test UB ≈ 0.12172195885947175
