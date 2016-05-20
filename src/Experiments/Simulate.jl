push!(LOAD_PATH, "../")
using H2SDDP
using Distributions
using Base.Test
@Logging.configure(level=Logging.DEBUG)

srand(123)
N = 4#4#5
T = 11
K = 4#4#5
S = 100
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005#0.005#0.003
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 0.012#0.027
#Parâmetros
S_LB = 300
S_FB = 5
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

slot_dates = 10
train_rows = 256*2
start_line = 1242-1#1242-1#3273-1 #2477-1 # first line is 0
n_stots_test = Int(floor(256*3/slot_dates))

x_ini_m = [1.0;zeros(N)]
x_ini_e = [1.0;zeros(N)]

run_SDDP = true
run_myopic = true
run_equal = true

info(H2SDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB ))

for i = 1:n_stots_test
  dH  = H2SDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

  #Run the C++ code to output HMM/LHS data
  #file_name = "5Ind_97a15"
  file_name = "db_br_2000a2015"
  id = 14
  start_train  = start_line + (i-1)*slot_dates+1
  n_rows_train = train_rows
  n_rows_test  = slot_dates
  info("start_train ($start_train, $n_rows_train, $n_rows_test)")
  readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S
    $id $start_train $n_rows_train $n_rows_test`)

  # HMM data
  file_prefix = string("../C++/output/",file_name,"_",id)
  dM = readHMMPara(file_prefix, dH)

#  info("Train SDDP portfolio")
  if run_SDDP
    LB, UB, AQ, sp = SDDP(dH, dM)
  end

  input_file = string("../C++/input/",file_name,".csv")
  r = readcsv(input_file,Float64)
  r = exp(r)-1
  pk_r = readcsv(string(file_prefix,"_PK_r.csv"),Float64)'
  start_test = start_train+n_rows_train
  test_r = r[start_test:start_test+n_rows_test-1,:]'

  if run_SDDP
    info("Simulating SDDP portfolio")
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini_s[2:N+1], x_ini_s[1])
    open(string("./output/",file_name,"_",id,"_sim_SDDP_G$(string(dH.γ)[3:end]).csv"),"a") do file
      writecsv(file,vcat(x0',x)')
    end
    open(string("./output/",file_name,"_",id,"sim_SDDP_G$(string(dH.γ)[3:end])_Er.csv"),"a") do file
      writecsv(file,exp_ret')
    end
    x_ini_s[2:N+1] = x[:,end]
    x_ini_s[1] = x0[end]
  end

  if run_myopic
    # Simulate myopic portfolio
    info("Training myopic portfolio")
    dH.T = 2
    LB, UB, AQ, sp, list_α, list_β, x_trial, u_trial = SDDP(dH, dM)
    info("Simulating myopic portfolio")
    all_m = simulatePercPort(dH, test_r, x_ini_m, u_trial[:,1]/sum(u_trial[:,1]))
    open(string("./output/",file_name,"_",id,"sim_myopic.csv"),"a") do file
      writecsv(file,all_m[:,2:end]')
    end
    x_ini_m = all_m[:,end]
  end

  if run_equal
    # Simulate weitght portfolio
    info("Simulating weitght portfolio")
    all_e = simulatePercPort(dH, test_r, x_ini_e, ones(dH.N+1)*1/(dH.N+1))
    open(string("./output/",file_name,"_",id,"sim_equal.csv"),"a") do file
      writecsv(file,all_e[:,2:end]')
    end
    x_ini_e = all_e[:,end]
  end

end

run(`/home/tas/woofy.sh 62491240 "Finish SDDP"`)
