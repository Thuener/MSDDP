#addprocs(3)
include("parametersBM100.jl")

γs = [0.05]#[0.005, 0.01,0.02]
cs = [0.02]#[0.005, 0.01,0.02]

file = string(output_dir,file_name,"_ret.csv")

# For each risk level (γ)
@time for i_γ = 1:length(γs)
  dH.γ = γs[i_γ]
  # For each transactional cost (c)
  for i_c = 1:length(cs)
    dH.c = cs[i_c]
    debug(dH)
    LB, UB, LB_c, AQ, sp, x_trial, u_trial, list_LB, List_UB, list_firstu = sddp(dH, dM; stabUB=0.1)#TODO 0.1 for BM100 and 0.5 for BM25

    γ_srt = string(dH.γ)[3:end]
    c_srt = string(dH.c)[3:end]

    writecsv(string(output_dir,file_name,"_LB_g$(γ_srt)_c$(c_srt).csv"),list_LB)
    writecsv(string(output_dir,file_name,"_UB_g$(γ_srt)_c$(c_srt).csv"),List_UB)
    writecsv(string(output_dir,file_name,"_u_g$(γ_srt)_c$(c_srt).csv"),list_firstu)
  end
end


run(`/home/tas/woofy.sh 62491240 "GAP"`)
