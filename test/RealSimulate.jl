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
x0_ini = 1.0
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
id = 1
start_train = 0
n_rows_train = 100
n_rows_test = 5
readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S
  $id $start_train $n_rows_train $n_rows_test`)

# HMM data
file_prefix = string("../C++/output/",file_name,"_",id)
dM = readHMMPara(file_prefix, dH)


LB, UB, AQ, sp = SDDP(dH, dM)

input_file = string("../C++/input/",file_name,".csv")
r = readcsv(input_file,Float64)
r = exp(r)-1
pk_r = readcsv(string(file_prefix,"_PK_r.csv"),Float64)'
start_test = start_train+n_rows_train+1
test_r = r[start_test:start_test+n_rows_test-1,:]'
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.0,0.0,0.0,0.0] 1e-6
@test_approx_eq_eps x[1,:] [1.0205000000328985 1.0044781500795639 1.0479720539866164 1.0916724886288345] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.0205000000328985 1.0044781500795639 1.0479720539866164 1.0916724886288345] 1e-5

# Mudando alguns parâmetros
dH.γ = 0.01
LB, UB, AQ, sp = SDDP(dH, dM)
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.858401692576853,0.8620998682740822,0.861057282583879,0.8688678993344314] 1e-6
@test_approx_eq_eps x[1,:] [0.0013866475012752548 0.0013432212781440424 0.0014220136022725061 0.0014327120564163085] 1e-6
@test_approx_eq_eps x[2,:] [0.08466804493326922 0.08148977888117075 0.08734991861527637 0.08663699039563105] 1e-6
@test_approx_eq_eps x[3,:] [0.018105369182835385 0.01788761839882134 0.01904756536670252 0.019100388320421678] 1e-6
@test_approx_eq_eps x[4,:] [0.041746456902285185 0.04027315805454114 0.0433158877262674 0.0405491802031727] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.004308211096518 1.0030936448867596 1.012192667894398 1.016587170310073] 1e-5

dH.α = 0.90
LB, UB, AQ, sp = SDDP(dH, dM)
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.8399509733122524,0.8442330176427562,0.8430291909630733,0.8513321958447] 1e-6
@test_approx_eq_eps x[1,:] [0.03987867932903003 0.03866015927018128 0.04091905927592926 0.04125870025816071] 1e-6
@test_approx_eq_eps x[2,:] [0.07599508721747775 0.07319990335191821 0.07844688210533292 0.07786661161522966] 1e-6
@test_approx_eq_eps x[4,:] [0.049273229506782625 0.047571678451575526 0.05115473293835722 0.047924260219477495] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.0050979693655429 1.0036647587164311 1.0135498652826926 1.018381767937568] 1e-5

dH.T = 5
LB, UB, AQ, sp = SDDP(dH, dM)
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.8399509733122524,0.8442330176427562,0.8430291909630733,0.8513321958447] 1e-6
@test_approx_eq_eps x[1,:] [0.03987867932903003 0.03866015927018128 0.04091905927592926 0.04125870025816071] 1e-6
@test_approx_eq_eps x[2,:] [0.07599508721747775 0.07319990335191821 0.07844688210533292 0.07786661161522966] 1e-6
@test_approx_eq_eps x[4,:] [0.049273229506782625 0.047571678451575526 0.05115473293835722 0.047924260219477495] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.0050979693655429 1.0036647587164311 1.0135498652826926 1.018381767937568] 1e-5


# Mudando a quantidade de estados
dH.K = 3
readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name 3 $N $S
  $id $start_train $n_rows_train $n_rows_test`)
dM = readHMMPara(file_prefix, dH)
input_file = string("../C++/input/",file_name,".csv")
r = readcsv(input_file,Float64)
r = exp(r)-1
pk_r = readcsv(string(file_prefix,"_PK_r.csv"),Float64)'
start_test = start_train+n_rows_train+1
test_r = r[start_test:start_test+n_rows_test-1,:]'

LB, UB, AQ, sp = SDDP(dH, dM)
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.8751433563272023,0.8808798980958331,0.8820897188746781,0.8918586722383713] 1e-6
@test_approx_eq_eps x[4,:] [0.13141161746898375 0.12705750152818618 0.13701037003112307 0.1285138308401227] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.006554973796186 1.0079373996240193 1.0191000889058013 1.020372503078494] 1e-5

dH.γ = 0.1
dH.α = 0.95
dH.T = 10
LB, UB, AQ, sp = SDDP(dH, dM)
x, x0 = simulate(dH, dM, AQ, sp, test_r, pk_r , x_ini, x0_ini)
@test_approx_eq_eps x0 [0.024255369908554036,0.025497889873494217,0.02577156359289401,0.028002055019142325] 1e-6
@test_approx_eq_eps x[4,:] [1.026971223197538 1.0370117189284034 1.1286967098134018 1.137730880624307] 1e-6
@test_approx_eq_eps sum([x0'; x],1) [1.051226593106092 1.0625096088018977 1.1544682734062959 1.1657329356434492] 1e-5
