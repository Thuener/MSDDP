using MSDDP
using Distributions
using Logging
Logging.configure(level=Logging.DEBUG)

srand(123)
Se = 1000
T_l = 120

#Parâmetros
N = 3
T = 13
K = 6
S = 500
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.002
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N+1, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
file_name = "$(N)MS_120_$(Se)"
file_prefix = "../C++/MS/output/"

# HMM data
dH.N = N+1
readall(`../C++/MS/HMM_MS /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $(dH.K) $(dH.N) $(dH.S) 1`)
file = string(file_prefix,file_name,"_1")
dM = readHMMPara(file, dH)
dM.r = dM.r[1:N,:,:] # removing the state z
dH.N -= 1
pk_r = readcsv(string(file_prefix,file_name,"_PK_r.csv"),Float64)'
pk_r = reshape(pk_r,dH.K,T_l,Se)
pk_r = pk_r[:,1:dH.T-1,:]


#Series data
input_file = string("../C++/input/",file_name,".csv")
r = readcsv(input_file,Float64)
r = reshape(r,N+1,T_l,Se)
r = r[1:N,1:dH.T-1,:]
r = exp(r)-1 -dF.r_f


ret = zeros(Float64,5,dH.T-1)
all = zeros(Float64,N+1,dH.T-1,Se)
##########################################################################################################################
info("#### SDDP with temporal dependecy and transactional costs ####")
info("Train")
@time LB, UB, LB_c, AQ, list_α, list_β, x_trial, u_trial = sddp(dH, dM)

info("Simulating")
tic()
γs = [0.001,0.003,0.005,0.007]
for i_γ = 1:length(γs)
  dH.γ = γs[i_γ]
  for i=1:Se
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, r[:,:,i], pk_r[:,:,i] , x_ini_s[2:N+1], x_ini_s[1])
    if dH.γ == γ
      ret[1,:] = sum(vcat(x0',x),1)
    end
    all[:,:,i] = vcat(x0',x)
  end
  writecsv(string("./output/",file_name,"_SDDP_TD_TC_g$(string(dH.γ)[3:end])_all.csv"),reshape(all,N+1,(dH.T-1)*Se)')
end
toc()

##########################################################################################################################
info("#### SDDP with temporal dependecy and no transactional costs ####")
dH.c = 0.0

info("Train")
@time LB, UB, LB_c, AQ, list_α, list_β, x_trial, u_trial = sddp(dH, dM)


info("Simulating")
tic()
for i=1:Se
  x, x0, exp_ret = simulate(dH, dM, AQ, sp, r[:,:,i], pk_r[:,:,i] , x_ini_s[2:N+1], x_ini_s[1]; real_trans_cost=c)
  ret[2,:] = sum(vcat(x0',x),1)
  all[:,:,i] = vcat(x0',x)
end
writecsv(string("./output/",file_name,"_SDDP_TD_NTC_all.csv"),reshape(all,N+1,(dH.T-1)*Se)')
toc()
dH.c = c

##########################################################################################################################
info("#### Myopic ####")
dH.T = 2

info("Train")
@time LB, UB, LB_c, AQ, list_α, list_β, x_trial, u_trial = sddp(dH, dM)

info("Simulating")
tic()
for i=1:Se
  all_m = simulatePercPort(dH, r[:,:,i], x_ini_s, u_trial[:,1]/sum(u_trial[:,1]))], x_ini_s[1])
  ret[3,:] = sum(all_m,1)
  all[:,:,i] = vcat(x0',x)
end
writecsv(string("./output/",file_name,"_SDDP_My_all.csv"),reshape(all,N+1,(dH.T-1)*Se)')
dH.T = T
toc()

##########################################################################################################################
info("#### Equaly weight ####")

info("Simulating")
tic()
for i=1:Se
  all_e = simulatePercPort(dH, test_r, x_ini_s, ones(dH.N+1)*1/(dH.N+1))
  ret[4,:] = sum(all_e,1)
  all[:,:,i] = vcat(x0',x)
end
writecsv(string("./output/",file_name,"_SDDP_Eq_all.csv"),reshape(all,N+1,(dH.T-1)*Se)')
toc()

##########################################################################################################################
info("#### SDDP with no temporal dependecy and transactional costs ####")
dH.K = 1

# HMM data
dH.N = N+1
readall(`../C++/MS/HMM_MS /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $(dH.K) $(dH.N) $(dH.S)`)
file = string(file_prefix,file_name)
dM = readHMMPara(file, dH)
dM.r = dM.r[1:N,:,:] # removing the state z
pk_r = readcsv(string(file_prefix,file_name,"_PK_r.csv"),Float64)'
pk_r = reshape(pk_r,dH.K,T_l,Se)
pk_r = pk_r[:,1:dH.T-1,:]
dH.N -= 1

info("Train")
@time LB, UB, LB_c, AQ, list_α, list_β, x_trial, u_trial = sddp(dH, dM)

info("Simulating")
tic()
for i=1:Se
  x, x0, exp_ret = simulate(dH, dM, AQ, sp, r[:,:,i], pk_r[:,:,i] , x_ini_s[2:N+1], x_ini_s[1])
  ret[5,:] = sum(vcat(x0',x),1)
end
writecsv(string("./output/",file_name,"_SDDP_NTD_TC_all.csv"),reshape(all,N+1,(dH.T-1)*Se)')
toc()

dH.K = K


writecsv(string("./output/",file_name,"_ret.csv"),ret)
