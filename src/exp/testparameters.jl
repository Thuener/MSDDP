using MSDDP
using Distributions
using Base.Test
using Gadfly
using Logging
Logging.configure(level=Logging.INFO)

N = 10
T = 5
K = 3
S = 100
α = 0.9
W_ini = 1.0
x_ini = zeros(N)
x0_ini = W_ini
c = 0.001
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parameters
S_LB = 500
S_LB_inc = 100
S_FB = 5
GAPP = -10 # GAP mínimo em porcentagem
Max_It = 5
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, W_ini, c, M, γ,
                S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )

file_name = "ken_10DInd_90a14"
file = string("../C++/output/",file_name)

function test_algo(dH::MSDDPData, LP)
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
          sddp(dH, dM, LP=LP)
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
