
push!(LOAD_PATH, "../")
using H2SDDP
using Distributions
using Base.Test
using Logging
using Gadfly

N = 10
T = 5
K = 3
S = 100
α = 0.9
x_ini = zeros(N)
x0_ini = 1.0
c = 0.001
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parâmetros
S_LB = 500
S_FB = 5
GAPP = -10 # GAP mínimo em porcentagem
Max_It = 5
α_lB = 0.9

dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

file_name = "ken_10DInd_90a14"
file = string("../C++/output/",file_name)

@Logging.configure(level=Logging.INFO)

function test_algo(dH::H2SDDPData, LP)
  Logging.configure(level=INFO, filename="CPLEX_$LP.log")

  γs = [0.001; 0.01; 0.02]
  samp = 3
  TimesCPLEXS = zeros(samp,size(γs,1))
  Logging.info("Samples, γ, time")
  tic()
  for j = 1:samp
      readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $(S) 1`)
      for k = 1:size(γs,1)
          tic()
          dM = readHMMPara(file, dH)
          dH.γ = γs[k]
          dH.S_LB = S_LB
          SDDP(dH, dM, LP=LP)
          TimesCPLEXS[j,k] = toq()
          println("[$j, $k] = $(γs[k]), $(TimesCPLEXS[j,k])")
      end
  end
  writecsv("./CPLEX_$LP.csv", TimesCPLEXS)
  toc()
  println(TimesCPLEXS)
end

LPs = collect(1:6)
for LP in LPs
  test_algo(dH, LP)
end
