using MSDDP, HMM_MSDDP, Simulate, Util
using Distributions
using Logging

import FFM

Logging.configure(level=Logging.DEBUG)

srand(123)

#Parameters
N = 100
T = 6
K = 3
S = 1000
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.03
M = 9999999
γ = 0.005
S_LB = 300
S_FB = 10
GAPP = 1
Max_It = 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

output_dir = "../../output/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
dFF = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
dM, model = inithmm_ffm(lnret[1:F,:]', dFF, dH)
