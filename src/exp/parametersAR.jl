T_max = 240
Sc = 1000
T_hmm = 120

# AR
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 0.00042

# Parmeters
N = 3
T = 12
K = 3
S = 750
α = 0.9
x0_ini = 1.0
x_ini = zeros(N)
c = 0.005
maxvl = 9999999
γ = 0.02
samplower = 300
samplower_inc = 100
nit_before_lower = 100
gap = 1.
max_it = 15
α_lower = 0.9

L = 1000

# Read series
file_name = string("$(N)MS_$(T_max)_$(Sc)")
file_dir = "../../input/"
file = string(file_dir,file_name,".csv")
serie = readcsv(file, Float64)
serie = reshape(serie,N+1,T_max,Sc)

# Divide the series
ln_ret = serie[1:N,:,:]
rets = exp(ln_ret)-1 -r_f

z = reshape(serie[N+1,:,:], T_max, Sc)

output_dir = "../../output/outputAR/"
