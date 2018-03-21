using MSDDP

#Parameters
N = 100
T = 12
K = 3
S = 1000
α = 0.9
x0_ini = 1000000.0 #TODO 1.0
x_ini = zeros(N)
c = 0.01
maxvl = 9999999
γ = 0.05
samplower = 300
samplower_inc = 100
nit_before_lower = 100
gap = 1. #TODO 0.5
max_it = 1000
α_lower = 0.9
