using MSDDP
using Distributions
using Base.Test
using Logging
using Gadfly
using DataFrames
using Logging

N = 10
T = 5
K = 3
S = 200
α = 0.9
x_ini = zeros(N)
x0_ini = 1.0
c = 0.001
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parâmetros
S_LB = 100
S_FB = 20
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

file_name = "ken_10DInd_90a14"
file = string("../C++/output/",file_name)
#readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S`)
#dM = readHMMPara(file, dH)
Logging.configure(level=Logging.INFO,filename="FrontEff3D.log")

readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $(dH.K) $(dH.N) $(dH.S)`)
dM = readHMMPara(file, dH)

γs = collect(0.001:0.001:0.02)
Ts = [3,5,10,20]
UBγs = Array(Float64,size(γs,1),size(Ts,1))

println("γ, UB, time")
for j = 1:length(Ts)
    dH.T = Ts[j]
    for i = 1:length(γs)
        tic()
        dH.γ = γs[i]
        LB, UB, AQ, sp = SDDP(dH, dM)
        UBγs[i] = UB
        println("$(γs[i]), $(UBγs[i]), $(toq())")
    end
    p = plot(layer(x=γs,y=UBγs,Geom.line,Geom.point))
    draw(PDF("./output/figuras/fronteff_T$(Ts[j])_ga$(string(γs[i])[3:end]).pdf", 4inch, 3inch),p )
end

#TODO
# Put the data in a DataFrame
d = DataFrame(
  x = vcat(x,x,x),
  y = vcat(y1,y2,y3),
  group = vcat( rep("1",n), rep("2",n), rep("3",n) )
)

# Plot
plot(
  d,
  x=:x, y=:y, color=:group,
  Geom.point,
  Scale.discrete_color_manual("green","red","blue")
)
