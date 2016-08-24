#addprocs(3)
using Simulate, MSDDP

#srand(123)

#Parameters
N = 30 # TODO 25
T = 22
K = 3
S = 1000 #TODO 1000
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.02 #TODO 0.08
S_LB = 300
S_FB = 10
GAPP = 1
Max_It = 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

output_dir = "../../output3/"

# Read series
F =3 #TODO # number of factors
file_dir = "../../input/"
file_name = "3FF_Ind30_Daily_small" #TODO
file = string(file_dir,file_name,".csv")
series = readcsv(file, Float64)'

R = 21 #TODO 6 # Regression for FFM and number o test samples(avoid border effect, has to be lower than dH.T-1)
nrows_train = 1759 #6564 #966 #522 # TODO 438 # 882
its=floor(Int,(size(series,2)-nrows_train)/(R))
series = series[:,1:nrows_train+(R)*its]
ret = zeros(Float64,5,(R)*its+1)

info("K = $(dH.K), F = $(F), R = $(R) , nrows_train = $(nrows_train)")
info("dH = $dH")

ret[1,:] =       runEqualy(deepcopy(dH), series, nrows_train, F, output_dir, file_name)
ret[2,:] =       runMyopic(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
ret[3,:] =  runMSDDP_TD_TC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
ret[4,:] = runMSDDP_TD_NTC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
ret[5,:] = runMSDDP_NTD_TC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)

#=
remote1  = @spawn runEqualy(deepcopy(dH), series, nrows_train, F, output_dir, file_name)
remote2  = @spawn runMyopic(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
remote3  = @spawn runMSDDP_TD_TC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
remote4  = @spawn runMSDDP_TD_NTC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)
remote5  = @spawn runMSDDP_NTD_TC(deepcopy(dH), series, nrows_train, F, R, output_dir, file_name)

ret[1,:] = fetch(remote1)
ret[2,:] = fetch(remote2)
ret[3,:] = fetch(remote3)
ret[4,:] = fetch(remote4)
ret[5,:] = fetch(remote5)
=#
writecsv(string(output_dir,file_name,"_ret_k$(string(dH.K))g$(string(dH.γ)[3:end]).csv"),ret)

run(`/home/tas/woofy.sh 62491240 "simulate"`)
