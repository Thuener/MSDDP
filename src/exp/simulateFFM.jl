# Simulate the Farma French Model
#addprocs(3)
using Simulate, MSDDP

srand(123)

Logging.configure(level=Logging.DEBUG)
include("parametersBM100.jl")
#dH.T = T = 13

output_dir = "../../output/SimulateFFM/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
series = readcsv(file, Float64)'

R = 11 # Regression for FFM and number o test samples(avoid border effect, has to be lower than dH.T-1)
nrows_train = 618 #TODO 318
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
