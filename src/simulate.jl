push!(LOAD_PATH, "./")
using H2SDDP
using Distributions
using Base.Test
using Logging

function readHMMPara(file)
  r = readcsv(string(file,"_samples.csv"),Float64)'
  r = exp(r)-1

  # Probabilidades (condicionais a cada estado) para cada cenario p(S|K)
  p = readcsv(string(file,"_PS.csv"),Float64)

  # prob iniciais
  prob_ini = readcsv(string(file,"_Pini.csv"),Float64)
  prob_ini = reshape(prob_ini,size(prob_ini,1))

  # Matriz de transicao  (K_t x K_(t+1))
  P_K = readcsv(string(file,"_PK.csv"),Float64)'

  dM = HMMData( r, p, prob_ini, P_K )
  return dM
end

#=
                              <input_csv_file> <test_case> <test_id> <dir_files>        <train_rows> <test> <test_rows>
Comandline parameters: ./SDDP ken_10DInd_90a14      6          1     /home/tas/qt/SDDP1    4290        91       22
                  stages,samples,trial_scen,St F,St B,Assets,Alpha,Gamma,TransCost
File parameters:   22    2500    1          3    3    11     0.9   0.03  0.01
=#


@Logging.configure(level=Logging.DEBUG, moreinfo=true)

## Teste para uma distribuição específica
srand(123)
N = 10
T = 22
K = 3
S = 2500
α = 0.9
x_ini = zeros(N)
x0_ini = 1.0
c = 0.001
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1#0.03#0.01
#Parâmetros
S_LB = 1000
S_FB = 1
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100

dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It )

# Fronteira efficiente



file_name = "ken_10DInd_90a14"
id = 1
start_train = 0
n_rows_train = 4290
n_rows_test = 5#22
n_tests = 1

all_x = Array{Float64,2}[]
for i = 1:n_tests
  Logging.info("End train: ",start_train+n_rows_train,
    " end test: ",start_train+n_rows_train+n_rows_test-1)
  #Run the C++ code to output HMM/LHS data
  readall(`./C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S
    $id $start_train $n_rows_train $n_rows_test`)

  # HMM data
  file_prefix = string("./C++/output/",file_name,"_",id)
  dM = readHMMPara(file_prefix)

  Logging.info("Running SDDP to evalute the cuts")
  LB, UB, AQ, sp = SDDP(dH, dM)

  input_file = string("./C++/input/",file_name,".csv")
  r = readcsv(input_file,Float64)
  r = exp(r)-1
  pk_r = readcsv(string(file_prefix,"_PK_r.csv"),Float64)'
  start_test = start_train+n_rows_train+1
  test_r = r[start_test:start_test+n_rows_test-1,:]'
  Logging.info("Simulating on test data")
  x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)

  push!(all_x,[x0'; x])
  start_train += n_rows_test-1
  dH.x_ini  = x[:,n_rows_test-1]
  dH.x0_ini =x0[n_rows_test-1]
end
println(all_x)
