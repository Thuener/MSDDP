
using MSDDP
using Distributions
using Base.Test
using Logging
using Gadfly
using HDF5
using JLD

function piecewise(B::Array{Float64,2},A::Array{Float64,1})
    @assert size(B,2) == 2
    @assert size(A,1) == size(B,1)
    f(x_1,x_2) = minimum(x_1*B[:,1] + x_2*B[:,2] + A)
    return f
end

function evafunc(dH::MSDDPData, α, β, t, k )
    n_cuts = size(α,1)
    a = Array(Float64,n_cuts)
    b = Array(Float64,dH.N+1,n_cuts)
    pf = Array{Function,1}(dH.K)
    for j = 1:dH.K
    for i = 1:n_cuts
        a[i] = α[i][t,j]
        b[:,i] = β[i][:,t,j]
    end
    pf[j]=piecewise(b', a)
    #    p = plot(pf[j],0.0,1)
    #    draw(PNG("./output/figuras/cuts_T$(T)_k$(k)_j$(j).png", 4inch, 3inch),p )
    end
    return pf
end;

N = 1
T = 5
K = 3
S = 100
α = 0.9
W_ini = 1.0
x_ini = zeros(N)
x0_ini = W_ini
c = 0.00
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parameters
S_LB = 500
S_LB_inc = 100
S_FB = 5
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, W_ini, c, M, γ,
                S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )

γs = [0.001; 0.01; 0.02]
s_γs = size(γs,1)
pf = Array(Function,dH.K,s_γs)
u_trial = Array(Float64,N+1,T,s_γs)
x_trial = Array(Float64,N+1,T,s_γs)
for i = 1:s_γs
    file = jldopen("./output/cuts_$i.jld", "r")
    list_β = read(file, "beta")
    list_α = read(file, "alpha")
    u_trial[:,:,i] = read(file, "u")
    x_trial[:,:,i] = read(file, "x")
    close(file)
    pf[:,i] = evafunc(dH, list_α, list_β, 2, i)
end

using PyPlot




for i = 1:s_γs
    for j = 1:dH.K
        println(j,i)
        pf1 = pf[j,i]
        fig = figure()
        X_1 = linspace(0, 1, 100)'
        X_2 = linspace(0, 1, 100)
        Z = Array(Float64,length(X_1),length(X_2))
        for k = 1:length(X_1)
            for l = 1:length(X_2)
                Z[k,l] = pf1(X_1[k],X_2[l])
            end
        end
        display(plot_surface(X_1, X_2, Z, rstride=1, cstride=1, linewidth=0, antialiased=false, cmap="coolwarm"))
    end
end
