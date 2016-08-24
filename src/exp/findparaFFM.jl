using HMM_MSDDP, FFM
using Distributions, HypothesisTests
using Logging, ArgParse


# Choose the number os samples for the LHS
# using standard deviation stabilization of the UB
function sampleslhs_stabUB(dH::MSDDPData, dFF::FFMData, ln_index::Array{Float64,2}, output_dir)
  last_std = 10000.0
  last_mean = 10000.0
  best_samp = 0
  samps = collect(250:250:1500)
  len_samps = length(samps)
  MUBs = zeros(Float64, len_samps)
  STDUBs = zeros(Float64, len_samps)

  for i = 1:len_samps
    dH.S = samps[i]
    max_it = 10
    UBs = SharedArray(Float64,max_it)
    @sync @parallel for it=1:max_it
      dM, model = inithmm_ffm(ln_index', dFF, dH)

      info("Train SDDP with $(dH.S) LHS samples")
      @time LB, UB, LB_c = sddp(dH, dM)
      UBs[it] = UB
    end
    STDUBs[i] = sqrt(var(UBs))
    MUBs[i] = mean(UBs)
    γ_srt = string(dH.γ)[3:end]
    c_srt = string(dH.c)[3:end]
    writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)_table_samp.csv"),hcat(STDUBs,MUBs))
    if i > 1 && abs(STDUBs[i] - STDUBs[i-1])/STDUBs[i] < 1e-1
      info("Stabilization with $(dH.S) samples")
      dH.S = samps[i-1]
      best_samp = dH.S
    end
  end
  return best_samp
end

function slidingwindow(dH::MSDDPData, dFF::FFMData, ln_index::Array{Float64,2})
  sum = 0
  perc = 0.1
  lot_train = 7
  rows_lot = floor(Int64, perc*size(ln_index,2))
  total = floor(Int64, 1/perc)
  for i = lot_train:total-1
    train = ln_index[:,1:rows_lot*i]
    test  = ln_index[:,rows_lot*i+1:rows_lot*(i+1)]
    try
      dM, model = inithmm_ffm(train', dFF, dH)
      sum += score(model, test')
    catch
      i = i -1
    end
  end
  return sum/(total-lot_train+1)
end

# Choose the number of sates for the HMM
function beststate_slidingwindow(dH::MSDDPData, dFF::FFMData, ln_index::Array{Float64,2})
  max_state = 7
  best_k = 0
  max_logll = -Inf
  for k=1:max_state
    dH.K = k
    logll = 0
    for i= 1:10
      logll += slidingwindow(dH, dFF, ln_index)
    end
    info("With $k states, logll $logll")
    if logll > max_logll
      max_logll = logll
      best_k = k
      info("New best with $k states, logll $max_logll ")
    end
  end
  return best_k
end

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--debug", "-d"
        help = "Show debug messanges"
        action = :store_true
    "--samp", "-S"
        help = "To selecting samples of the LHS"
        action = :store_true
    "--stat", "-K"
        help = "To selecting the number of states of the Markov model"
        action = :store_true
  end

  return parse_args(s)
end



# Start of the scipt
args = parse_commandline()

if args["debug"]
  Logging.configure(level=Logging.DEBUG)
  DEBUG = true
else
  Logging.configure(level=Logging.INFO)
end

srand(123)
#Parameters
N = 30
T = 12
K = 1
S = 1000
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.02
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

# Read series
F = 3
file_dir = "../../input/"
file_name = "3FF_Ind30_Daily"
file = string(file_dir,file_name,".csv")
series = readcsv(file, Float64)'
#nrows_train = 84 #(2000 to 2006)
#series = series[:,1:84]
ln_series = log(series+1)
ln_index = ln_series[1:F,:]
dFF = FFM.evaluate(ln_index,ln_series[F+1:end,:])


if args["stat"]
  its = 10
  best_ks = Array(Float64,its)
  for i=1:its
    dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
    output_dir = "../../output5/"

    best_ks[i] = beststate_slidingwindow(dH, dFF, ln_index)
    writecsv(string(output_dir, file_name, "_best_K.csv"), best_ks)
  end
end

if args["samp"]
  γs = [0.02,0.05,0.08,0.1,0.20]
  cs = [0.005]
  info(" γs = $(γs), cs = $(cs)")
  bests_sam = zeros(Int64, length(γs), length(cs))
  # For each risk level (γ)
  for i_γ = 1:length(γs)
    γ = γs[i_γ]

    # For each transactional cost (c)
    for i_c = 1:length(cs)
      c = cs[i_c]
      info("Start testes with γ = $(γ) and c = $(c)")

      dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
      output_dir = "../../output5/"

      bests_sam[i_γ,i_c] = sampleslhs_stabUB(dH, dFF, ln_index, output_dir)
      writecsv(string(output_dir, file_name, "_best_S.csv"),bests_sam)
    end
  end
end

run(`/home/tas/woofy.sh 62491240 "Finish findparaFFM"`)
