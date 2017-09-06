module MSDDP

using JuMP, CPLEX
using Distributions, MathProgBase
using Logging, JLD

# Types
export MKData, MAAParameters, SDDPParameters, ModelSizes, MSDDPModel

export solve, simulate, simulatesw, simulate_stateprob, simulatestates, simulate_percport,  createmodels!, reset!,
loadcuts!, param
export nstages, nassets, nstates, nscen, transcost, probscen
export setnstages!, setnstates!, setnassets!, setnscen!, setα!, setmarkov!, setγ!, setinistate!, settranscost!
export chgrrhs_low!
export memuse

export Subproblem


## Types ##

type Subproblem
    jmodel::JuMP.Model
    # Ids for the constraints
    cash::Int64
    assets::Vector{Int64}
    risk::Int64
end

type Stage
    subproblems::Vector{Subproblem}
end

" Markov Chain data "
type MKData
    inistate::Int64                         # Initial state
    transprob::Array{Float64,2}             # Transtition probability
    prob_scenario_state::Array{Float64,2}   # Probability of each scenario given a state
    ret::Array{Float64,4}                   # Return with stages, assets, states and samples
end

" Multistage asset allocation model parameters "
type MAAParameters
    α::Float64                   # Confidence level CVaR
    γ::Float64                   # Limit CVaR constraint
    transcost::Float64           # Transactional costs
    iniassets::Vector{Float64}  # Initial wealth in assets
    inirf::Float64               # Initial wealth in risk free asset
    maximum::Int64               # Maximum value future value function
end

" Parameters for the SDDP execution "
type SDDPParameters
    max_iterations::Int64           # Max interations of SDDP
    samplower::Int64                # Number of samples used in lower bound evaluation
    samplower_inc::Int64            # Number of samples to increment lower bound when stable
    nit_before_lower::Int64         # Number of fowards and backwards before evaluate lower bound
    gap::Float64                    # Minimum percentage gap between the lower and upper bounds (%)
    α_lower::Float64                # Confidence level for the lower bound
    diff_upper::Float64             # Difference in the upper bound to define stabilization
    print_level::Int                # Log print level
    lowlevel_api::Bool              # Use or not low level CPLEX api
    parallel::Bool                  # Solve the SDDP in parallel
    simu_lower::Bool                # Simulate the lower bound
    fast_lower::Bool                # Use the fast evaluation of the lower bound
    file::String                    # Output JLD file
end

type ModelSizes
    nstages::Int64
    nassets::Int64
    nstates::Int64
    nscen::Int64
end

type MSDDPModel
    sizes::ModelSizes
    lpsolver::JuMP.MathProgBase.AbstractMathProgSolver
    asset_parameters::MAAParameters
    param::SDDPParameters
    markov_data::MKData
    stages::Vector{Stage}
end


## Util functions ##

SDDPParameters(max_it::Int64, samplower::Int64, samplower_inc::Int64, nit_before_lower::Int64, gap::Float64, α_lower::Float64;
    diff_upper= 0.5, print_level=0, lowlevel_api=true, parallel=false, simu_lower=false, fast_lower=true, file= "" ) =
    SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower, diff_upper,
        print_level, lowlevel_api, parallel, simu_lower, fast_lower, file)

" Construct the MSDDPModel without ModelSizes "
function MSDDPModel(asset_parameters::MAAParameters,
        param::SDDPParameters,
        markov_data::MKData;
        nassets  = length(asset_parameters.iniassets),
        nstages  = size(markov_data.ret, 1),
        nstates  = size(markov_data.transprob, 1),
        nscen    = size(markov_data.prob_scenario_state, 1),
        lpsolver = ClpSolver())
    MSDDPModel(ModelSizes(nstages, nassets, nstates, nscen), lpsolver,
        asset_parameters, param, markov_data)
end

" Construct the MSDDPModel using ModelSizes "
function MSDDPModel(msize::ModelSizes,
        lpsolver::JuMP.MathProgBase.AbstractMathProgSolver,
        asset_parameters::MAAParameters,
        param::SDDPParameters,
        markov_data::MKData)
    stages = Vector{Stage}(nstages(msize))
    for t = 1:nstages(msize)
        stages[t] = Stage(Vector{Subproblem}(nstates(msize)))
    end
    m = MSDDPModel(msize, lpsolver, asset_parameters, param, markov_data, stages)
    createmodels!(m)
    m
end

" Reset the stages vector inside MSDDPModel and create models"
function reset!(m::MSDDPModel)
    stages = Vector{Stage}(nstages(m))
    for t = 1:nstages(m)
        stages[t] = Stage(Vector{Subproblem}(nstates(m)))
    end
    m.stages = stages
    createmodels!(m)
    nothing
end

#  Utils for ModelSizes
nstages(ms::ModelSizes) = ms.nstages
nassets(ms::ModelSizes) = ms.nassets
nstates(ms::ModelSizes) = ms.nstates
nscen(ms::ModelSizes)   = ms.nscen

setnstages!(ms::ModelSizes, n::Int)   = ms.nstages = n
setnassets!(ms::ModelSizes, n::Int)   = ms.nassets = n
setnstates!(ms::ModelSizes, n::Int)   = ms.nstates = n
setnscen!(ms::ModelSizes, n::Int)     = ms.nscen = n

#  Utils for MSDDPModel
msddp(m::MSDDPModel)   = m
nstages(m::MSDDPModel) = nstages(m.sizes)
nassets(m::MSDDPModel) = nassets(m.sizes)
nstates(m::MSDDPModel) = nstates(m.sizes)
nscen(m::MSDDPModel)   = nscen(m.sizes)

param(m::MSDDPModel)   = m.param
assetspar(m::MSDDPModel)   = m.asset_parameters

function inialloc!(m::MSDDPModel, x::AbstractVector{Float64}, rf::Float64)
    m.asset_parameters.iniassets = x
    m.asset_parameters.inirf = rf
    nothing
end

inirf(m::MSDDPModel)                 = m.asset_parameters.inirf
iniassets(m::MSDDPModel)             = m.asset_parameters.iniassets
iniassets(m::MSDDPModel, index::Int) = iniassets(m)[index]
inialloc(m::MSDDPModel)              = vcat(inirf(m), iniassets(m))
inialloct(m::MSDDPModel)             = inirf(m), iniassets(m)
initwealth(m::MSDDPModel)            = sum(inialloc(m))
inistate(m::MSDDPModel)              = m.markov_data.inistate

stage(m::MSDDPModel, stage::Int)     = m.stages[stage]
subproblem(m::MSDDPModel, stage::Int, state::Int) = m.stages[stage].subproblems[state]

transcost(m::MSDDPModel)             = m.asset_parameters.transcost
transprob(m::MSDDPModel)             = m.markov_data.transprob
transprob(m::MSDDPModel, state::Int) = transprob(m)[state,:]
probscen(m::MSDDPModel)              = m.markov_data.prob_scenario_state
probscen(m::MSDDPModel, scen::Int, state::Int) = probscen(m)[scen, state]

returns(m::MSDDPModel)               = m.markov_data.ret
returns(m::MSDDPModel, stage::Int)   = returns(m)[stage,:,:,:]
returns(m::MSDDPModel, stage::Int, asset::Int, state::Int, sample::Int)  = returns(m)[stage, asset, state, sample]

maximum(ap::MAAParameters)           = ap.maximum
jumpmodel(sp::Subproblem)            = sp.jmodel

setnstages!(m::MSDDPModel, n::Int)   = m.sizes.nstages = n
setnassets!(m::MSDDPModel, n::Int)   = m.sizes.nassets = n
setnstates!(m::MSDDPModel, n::Int)   = m.sizes.nstates = n
setnscen!(m::MSDDPModel, n::Int)     = m.sizes.nscen = n

setmarkov!(m::MSDDPModel, mk::MKData)= m.markov_data = mk
setγ!(m::MSDDPModel, γ::Float64)     = m.asset_parameters.γ = γ
setα!(m::MSDDPModel, α::Float64)     = m.asset_parameters.α = α
setinistate!(m::MSDDPModel, state::Int) = m.markov_data.inistate = state
settranscost!(m::MSDDPModel, tc::Float64) = m.asset_parameters.transcost = tc

function memuse()
  pid = getpid()
  return string(round(Int,parse(Int,readstring(`ps -p $pid -o rss=`))/1024),"M")
end


" Change the RHS using the index of the constraint "
function chgrrhs(sp::JuMP.Model, idx::Int64, rhs::Number)
    constr = sp.linconstr[idx]
    if constr.lb != -Inf
        if constr.ub != Inf
            if constr.ub == constr.lb
                sen = :(==)
            else
                sen = :range
            end
        else
            sen = :(>=)
        end
    else #if constr.lb == -Inf
        constr.ub == Inf && error("'Free' constraint sense not supported")
        sen = :(<=)
    end

    if sen == :range
        error("Modifying range constraints is currently unsupported.")
    elseif sen == :(==)
        constr.lb = float(rhs)
        constr.ub = float(rhs)
    elseif sen == :>=
        constr.lb = float(rhs)
    else
        @assert sen == :<=
        constr.ub = float(rhs)
    end
end

" Set constraint RHS using the Cplex low level function "
function setrhs_low!(cmodel::CPLEX.Model, idx::Int, rhs::Number)
    ncons = 1
    stat = ccall(("CPXchgrhs",CPLEX.libcplex), Cint, (
                    Ptr{Void},
                    Ptr{Void},
                    Cint,
                    Ptr{Cint},
                    Ptr{Cdouble}
                    ),
                    cmodel.env.ptr, cmodel.lp, ncons, Cint[idx-1;], float([rhs]))
    if stat != 0
        throw(CplexError(cmodel.env, stat))
    end
end

" Change constraint RHS using the Cplex low level function "
function chgrrhs_low!(sp::JuMP.Model, idx::Int64, rhs::Number)
    setrhs_low!(sp.internalModel.inner, idx, rhs)
end

function getDual(sp::JuMP.Model, idx::Int64)
    if length(sp.linconstrDuals) != MathProgBase.numlinconstr(sp)
        error("Dual solution not available. Check that the model was properly solved and no integer variables are present.")
    end
    return sp.linconstrDuals[idx]
end

function getdual_low(sp::JuMP.Model, idx::Int64)
    duals = CPLEX.get_constr_duals(sp.internalModel.inner)
    return duals[idx]
end

function getduals_low(sp::JuMP.Model, ids::Vector{Int})
    duals = CPLEX.get_constr_duals(sp.internalModel.inner)
    return duals[ids]
end

function solve_low(sp::JuMP.Model)
    CPLEX.optimize!(sp.internalModel.inner)
    return CPLEX.status(sp.internalModel)
end

" Evalute immediate benefit using low level Cplex api "
function immediatebenefit_low(model, sp::JuMP.Model, state::Int, rets::Array{Float64,3})

    u = getvalue_low(sp, 2:nassets(model)+1)
    b = getvalue_low(sp, 2+nassets(model):2*nassets(model)+1)
    d = getvalue_low(sp, 2*(1+nassets(model)):3*nassets(model)+1)

    stateprob = transprob(model, state)
    B_imed = - transcost(model)*sum(b[i] + d[i] for i = 1:nassets(model)) +
             sum((sum(rets[i,k,s]*u[i] for i = 1:nassets(model)) )*stateprob[k]*probscen(model, s, k)
             for k = 1:nstates(model), s = 1:nscen(model))
    return B_imed
end

function getobjectivevalue_low(sp::JuMP.Model)
    CPLEX.get_objval(sp.internalModel.inner)
end

function getvalue_low(sp::JuMP.Model, idx::Int64)
    return CPLEX.get_solution(sp.internalModel.inner)[idx]
end

function getvalue_low(sp::JuMP.Model, ridx::UnitRange{Int64})
    vidx = collect(ridx)
    ret = Array(Float64,length(vidx))
    values = CPLEX.get_solution(sp.internalModel.inner)
    for id = 1: length(vidx)
        ret[id] = values[vidx[id]]
    end
    return ret
end

" Add subproblem to model "
function createmodel!(m::MSDDPModel, stage::Int, state::Int)
    rets = returns(m, stage+1)
    probstates = transprob(m, state)
    ap = assetspar(m)


    jmodel = Model(solver = m.lpsolver)
    @variable(jmodel, u0 >= 0)
    @variable(jmodel, u[1:nassets(m)] >= 0)
    @variable(jmodel, b[1:nassets(m)] >= 0)
    @variable(jmodel, d[1:nassets(m)] >= 0)
    @variable(jmodel, z)
    @variable(jmodel, y[1:nstates(m),1:nscen(m)] >= 0)
    @variable(jmodel, θ[1:nstates(m),1:nscen(m)] <= maximum(ap))
    @objective(jmodel, Max, - transcost(m)*sum(b[i] + d[i] for i = 1:nassets(m)) +
                  sum((sum(rets[i,k,s]*u[i] for i = 1:nassets(m)) + θ[k,s])*probstates[k]*probscen(m, s, k)
                  for k = 1:nstates(m), s = 1:nscen(m)))

    cash = @constraint(jmodel, u0 + sum((1+transcost(m))*b[i] - (1-transcost(m))*d[i] for i = 1:nassets(m)) == inirf(m)).idx
    assets = Array(Int64,nassets(m))
    for i = 1:nassets(m)
        assets[i] = @constraint(jmodel, u[i] - b[i] + d[i] == iniassets(m, i)).idx
    end
    risk =  @constraint(jmodel,-(z - sum(probstates[k]*probscen(m, s, k)*y[k, s] for k = 1:nstates(m), s = 1:nscen(m))/(1-ap.α))
                          + transcost(m)*sum(b[i] + d[i] for i = 1:nassets(m)) <= ap.γ*initwealth(m)).idx

    @constraint(jmodel, trunc[k = 1:nstates(m), s = 1:nscen(m)], y[k,s] >= z - sum(rets[i,k,s]*u[i] for i = 1:nassets(m)))

    if stage == nstages(m)-1
        @constraint(jmodel, bound_cuts[k = 1:nstates(m), s = 1:nscen(m)], θ[k,s] ==  0)
    end

    status = JuMP.solve(jmodel)
    if status ≠ :Optimal
        writeLP(jumpmodel(sp),"prob.lp")
        error("Can't solve the problem status:",status)
    end
    m.stages[stage].subproblems[state] = Subproblem(jmodel, cash, assets, risk)
end

function createmodels!(model)
    for t = 1:nstages(model)-1
        for j = 1:nstates(model)
            createmodel!(model, t, j)
        end
    end
end

" Simulating forward states "
function simulatestates(model, states_forward::Vector{Int64}, rets_forward::Array{Float64,2})
  states_forward[1] = inistate(model)

  for t = 2:nstages(model)
    # Simulating states
    prob_trans = vec(transprob(model)[states_forward[t-1],1:nstates(model)])
    states_forward[t] = rand(Categorical(prob_trans))

    # Simulationg forward scenarios
    rand_idx = rand(Categorical(probscen(model)[1:nscen(model),states_forward[t]]))
    rets_forward[1:nassets(model),t] = returns(model)[t,1:nassets(model),states_forward[t],rand_idx];
  end
end

function forward!(model, states::Vector{Int64}, rets::Array{Float64,2};
        nstag = nstages(msddp(model)), real_transcost=0.0)
    m = msddp(model)

    # Initialize
    x_trial = zeros(Float64, nassets(m), nstag)
    x0_trial = zeros(Float64, nstag)
    x_trial[1:nassets(m), 1] = iniassets(m)
    x0_trial[1] = inirf(m)
    u_trial = zeros(nassets(m)+1, nstag)
    ap = assetspar(m)
    obj_forward = initwealth(m)
    for t = 1:nstag-1
        k = states[t]
        sp = subproblem(m, t, k)

        chgrrhs_low!(jumpmodel(sp), sp.cash, x0_trial[t])
        chgrrhs_low!(jumpmodel(sp), sp.risk, ap.γ*(sum(x_trial[:,t])+x0_trial[t]) )
        for i = 1:nassets(m)
            chgrrhs_low!(jumpmodel(sp), sp.assets[i], x_trial[i,t])
        end
        # Resolve subprob in time t
        status = solve_low(jumpmodel(sp))
        if status ≠ :Optimal
            writeLP(jumpmodel(sp),"prob.lp")
            error("Can't solve the problem status:",status)
        end
        # Evalute immediate benefit
        obj_forward += immediatebenefit_low(m, jumpmodel(sp), states[t], returns(m,t+1))

        # Update trials
        id_u0 = 1
        for i = 1:nassets(m)
            u_val = getvalue_low(jumpmodel(sp), i+id_u0)
            u_trial[i+1,t] = u_val
            x_trial[i,t+1] = (1+rets[i,t+1])*u_val
        end
        u_trial[1,t] = getvalue_low(jumpmodel(sp), 1)

        # If transactional cost is different from the optimization model
        if real_transcost != 0.0
            b = getindex(jumpmodel(sp),:b)
            d = getindex(jumpmodel(sp),:d)
            b_v = getvalue(b)
            d_v = getvalue(d)
            x0_trial[t+1] = - sum((1.0+real_transcost)*b_v) + sum((1.0-real_transcost)*d_v) + x0_trial[t]
        else
            x0_trial[t+1] = getvalue_low(jumpmodel(sp), 1)
        end
    end

    return x_trial, x0_trial, obj_forward, u_trial
end

function backward!(model, x_trial::Array{Float64,2}, x0_trial::Vector{Float64}, cutsfile::String)
    m = msddp(model)
    # Initialize
    α = ones(nstages(m)-1,nstates(m))
    β = ones(nassets(m)+1,nstages(m)-1,nstates(m))
    ap = assetspar(m)

    # Add cuts to t < T-1
    for t = nstages(m)-1:-1:2
        for j = 1:nstates(m)
            sp = subproblem(m, t, j)

            chgrrhs_low!(jumpmodel(sp), sp.cash, x0_trial[t])
            chgrrhs_low!(jumpmodel(sp), sp.risk, ap.γ*(sum(x_trial[:,t])+x0_trial[t]) )
            for i = 1:nassets(m)
                chgrrhs_low!(jumpmodel(sp), sp.assets[i], x_trial[i,t])
            end

            status = solve_low(jumpmodel(sp))
            if status ≠ :Optimal
                writeLP(jumpmodel(sp),"prob.lp")
                error("Can't solve the problem status:",status)
            end

            # Evalute custs
            λ0 = getdual_low(jumpmodel(sp), sp.cash)
            λ = zeros(nassets(m))
            for i = 1:nassets(m)
                λ[i] = getdual_low(jumpmodel(sp), sp.assets[i])
            end
            π = getdual_low(jumpmodel(sp), sp.risk)
            obj = getobjectivevalue_low(jumpmodel(sp))
            α[t,j] =  obj - (λ0 + ap.γ*π)*x0_trial[t] - sum([(λ[i] + ap.γ*π)*x_trial[i,t] for i = 1:nassets(m)])
            β[:,t,j] = vcat(λ0, λ) + ap.γ*π
        end

        for k = 1:nstates(m)
            addcut_low!(m, α, β, t, k)
        end
    end

    # Save cuts
    if cutsfile != ""
        open(cutsfile,"a") do file
            writecsv(file, [nassets(m)+1,nstages(m),nstates(m)]')
            writecsv(file, α)
            writecsv(file, β)
        end
    end
end

" Load cuts into the model "
function loadcuts!(model, file::String)
    open(file,"r") do f
        while true
            line = readline(f)
            line == nothing || line == "" && break
            items = split(line, ",")
            nassets_p1 = parse(Int, items[1])
            nstages = parse(Int, items[2])
            nstates = parse(Int, items[3])
            α = Array(Float64, nstages-1, nstates)
            β = Array(Float64, nassets_p1, nstages-1, nstates)
            for t = 1:nstages-1
                line = readline(f)
                items = split(line, ",")
                for j = 1:nstates
                    α[t,j] = parse(Float64,items[j])
                end
            end
            for j = 1:nstates
                for t = 1:nstages-1
                    for i = 1:nassets_p1
                        β[i,t,j] = parse(Float64,readline(f))
                    end
                end
            end
            # Add cuts to model
            for t = 1:nstages-1
                if t != 1
                    for k = 1:nstates
                        addcut_low!(msddp(model), α, β, t, k)
                    end
                end
            end
        end
    end
end

function addcut_low!(model, α::Array{Float64,2}, β::Array{Float64,3}, stage::Int, state::Int)
    jsp = jumpmodel(subproblem(model, stage-1, state))
    nvariables = CPLEX.num_var(jsp.internalModel.inner)
    u0_id = 1
    u_ids = collect(2:nassets(model)+1)

    coef = zeros(Float64, nstates(model)*nscen(model), nvariables)
    rhs = zeros(Float64, nstates(model)*nscen(model))
    ind_ini = nvariables - nstates(model)*nscen(model)

    for j = 1:nstates(model)
        for s = 1:nscen(model)
            ind_const = s + (j-1)*nscen(model)
            ind_θ = ind_ini + ind_const

            coef[ind_const, ind_θ] = 1
            coef[ind_const, u0_id] = -β[1,stage,j]
            for i = 1:nassets(model)
                coef[ind_const, u_ids[i]] = -β[i+1,stage,j]*(1+returns(model,stage+1,i,j,s))
            end
            rhs[ind_const] = α[stage,j]
        end
    end
    CPLEX.add_constrs!(jsp.internalModel.inner, coef, '<', rhs)
end

function addcut!(model, α::Array{Float64,2}, β::Array{Float64,3}, stage::Int, state::Int)
    jsp = jumpmodel(subproblem(model, stage-1, state))
    θ = getindex(jsp,:θ)
    u = getindex(jsp,:u)
    u0 = getindex(jsp,:u0)

    @constraint(jsp, [j = 1:nstates(model), s = 1:nscen(model)],
    θ[j,s] <= α[stage,j] + β[1,stage,j]*u0 + sum(β[i+1,stage,j]*(1+returns(model,stage+1,i,j,s))*u[i] for i = 1:nassets(model)))
end

function solve(model, p::SDDPParameters; cutsfile::String = "")
    # Delete cutsfile file
    if cutsfile != ""
        rm(cutsfile; force=true)
    end
    sd = param(model)
    ap = assetspar(model)
    x_trial = []
    x0_trial = []
    u_trial = []
    rets_forward = zeros(Float64, nassets(model), nstages(model))
    states_forward = Array(Int64, nstages(model))

    quantil = quantile(Normal(),sd.α_lower)

    gap = 100.0
    it = 0
    samplower_ini = p.samplower

    list_firstu = inialloc(model)
    list_uppers = [-1000]
    list_lowers = [-1000 -1000]
    if p.parallel
        lower = SharedArray(Float64, p.samplower)
    else
        lower = Array(Float64, p.samplower)
    end
    upper          = 9999999.0
    it_stable      = 0
    upper_last     = 9999999.0
    lower_conserv  = 0.0
    eps_upper      = 1e-6
    forwards_lower = 0 # number of forwards to evaluate lower bound
    while abs(gap) > p.gap && upper > eps_upper && it < p.max_iterations
        it += 1
        tic()
        for it_forward_backward = 1:p.nit_before_lower
            # Forward
            debug("Forward Step memuse $(memuse())")
            simulatestates(model, states_forward, rets_forward)
            x_trial, x0_trial, obj_forward, u_trial = forward!(model, states_forward, rets_forward)

            # Backward
            debug("Backward Step memuse $(memuse())")
            backward!(model, x_trial, x0_trial, cutsfile)


            # Evaluate upper bound
            sp = subproblem(model, 1, inistate(model))
            status = solve_low(jumpmodel(sp))
            if status ≠ :Optimal
                writeLP(jumpmodel(sp),"prob.lp")
                error("Can't solve the problem status:",status)
            end
            upper = getobjectivevalue_low(jumpmodel(sp)) + initwealth(model)
            u = getvalue_low(jumpmodel(sp), 2:nassets(model)+1)
            u0 = getvalue_low(jumpmodel(sp), 1)

            list_firstu = hcat(list_firstu,vcat(u0,u))

            debug("obj forward = $obj_forward, upper bound = $upper, stab upper $(abs(upper/upper_last -1)*100)")
            if p.fast_lower && (abs(upper/upper_last -1)*100 < p.diff_upper || upper < eps_upper || isnan(abs(upper/upper_last -1)*100))
                it_stable += 1
                if it_stable >= 5
                    if p.samplower < 30*samplower_ini
                        p.samplower = round(Int64,p.samplower + p.samplower_inc)
                        info("Stable upper bound, increasing samples for lower bound for $(p.samplower)")
                        if p.parallel
                            lower = SharedArray(Float64,p.samplower)
                        else
                            lower = Array(Float64,p.samplower)
                        end
                    end
                    break
                end
            else
                it_stable = 0
                p.samplower = samplower_ini
                if p.parallel
                    lower = SharedArray(Float64,p.samplower)
                else
                    lower = Array(Float64,p.samplower)
                end
            end
            upper_last = upper
        end

        # Lower bound
        debug("Evaluating the Lower Bound memuse $(memuse())")
        lower_conserv = 0
        gap = 100
        sumlower =0
        doub_samplower = false
        for forwards_lower = 1:p.samplower
            simulatestates(model, states_forward, rets_forward)
            x_trial, x0_trial, obj_forward, u_trial = forward!(model, states_forward, rets_forward)
            lower[forwards_lower] = obj_forward
            sumlower += obj_forward

            if p.fast_lower
                # Start testing gap_mean after samplower_ini/3 simulations
                if forwards_lower >= samplower_ini/3
                    meanlower = sumlower/forwards_lower
                    gap_mean = 100*(upper - meanlower)/upper
                    if gap_mean > p.gap*3 && it_stable < 10
                        meanlower = sumlower/forwards_lower
                        lower_conserv = (meanlower - quantil * std(lower[1:forwards_lower])/sqrt(forwards_lower))
                        gap = 100*(upper - lower_conserv)/upper
                        info("gap_mean $gap_mean is higher than $(p.gap*3) using $forwards_lower Forwards. Aborting lower evaluation.")
                        break
                        # If the gap_mean is lower then p.gap( the gap is almost close)
                        # doubles the number of scenarios used in lower
                    elseif doub_samplower == false && gap_mean < p.gap
                        doub_samplower = true
                        p.samplower = round(Int64,p.samplower + p.samplower_inc)
                        info("gap_mean < p.gap increasing samples for lower bound for $(p.samplower)")
                        if p.parallel
                            lower = vcat(lower,SharedArray(Float64,p.samplower_inc))
                        else
                            lower = vcat(lower,Array(Float64,p.samplower))
                        end
                    end
                end
                # Start testing gap after samplower_ini simulations
                if forwards_lower >= samplower_ini
                    meanlower = sumlower/forwards_lower
                    lower_conserv = (meanlower - quantil * std(lower[1:forwards_lower])/sqrt(forwards_lower))
                    gap = 100*(upper - lower_conserv)/upper
                    if abs(gap) < p.gap
                        info("SDDP ended: gap lower $gap is lower than $(p.gap) using $forwards_lower Forwards")
                        break
                    end

                    gap_mean = 100*(upper - meanlower)/upper
                    if gap_mean > p.gap && it_stable < 10
                        info("gap_mean $gap_mean is higher than $(p.gap) upper using $forwards_lower Forwards. Aborting lower evaluation.")
                        break
                    end
                end
            end
        end
        if p.file != ""
            save(string(p.file,"_MSDDP.jld"),"lower", lower[1:forwards_lower],"upper", upper,"lower_c", lower_conserv,"x",
            vcat(x0_trial',x_trial), "u", u_trial, "l_lower", list_lowers, "l_upper", list_uppers,
            "l_firsu",list_firstu)
        end
        if !p.fast_lower # Evaluate the lower bound always using samples for lower bound forwards
            meanlower = sumlower/forwards_lower
            lower_conserv = (meanlower - quantil * std(lower[1:forwards_lower])/sqrt(forwards_lower))
            gap = 100*(upper - lower_conserv)/upper
        end
        list_uppers = vcat(list_uppers,upper)
        list_lowers = vcat(list_lowers,[mean(lower[1:forwards_lower]) std(lower[1:forwards_lower])/sqrt(forwards_lower)])

        time_it = toq()
        info("lower bound conservative = $lower_conserv, upper bound = $upper, gap(%) = $gap, time it: $time_it")
        if it >= p.max_iterations
            info("SDDP ended: maximum number of iterations exceeded")
            break
        end
        if upper <= eps_upper
            info("SDDP ended: upper bound is zero")
            break
        end
    end

    if p.simu_lower
        file = jldopen("./output/allo_data_G$(string(ap.γ)[3:end])_C$(string(transcost(model))[3:end]).jld", "w")
        debug("Evaluating the Lower bound")
        states_forward_o = Array(Int64,nstages(model))
        rets_forward_o = zeros(nassets(model),nstages(model))
        addrequire(file, MSDDP)
        write(file,"SDDPParameters",param(model))
        write(file,"MKData",model.markov_data)
        for forwards_lower = 1:3
            simulatestates(model, states_forward_o, rets_forward_o)
            x_trial_o, x0_trial_o, obj_forward_o, u_trial_o = forward!(model, states_forward_o, rets_forward_o; parallel=p.parallel)
            write(file, "x$forwards_lower", vcat(x0_trial_o',x_trial_o))
            write(file, "u$forwards_lower", u_trial_o)
            write(file, "K$forwards_lower", states_forward_o)
        end
    end
    p.samplower = samplower_ini
    return lower[1:forwards_lower], upper, lower_conserv, model, vcat(x0_trial',x_trial), u_trial, list_lowers, list_uppers, list_firstu
end

function simulate_stateprob(model, rets::Array{Float64,2}, probret_state::Array{Float64,2}; real_transcost=0.0)

   states = Array(Int64,nstages(model)-1)
   for t = 1:nstages(model)-1
     states[t] = findmax(probret_state[:,t])[2]
   end

   return simulate(model, rets, states, real_transcost=real_transcost)
end

function simulate(model, rets::Array{Float64,2}, states::Array{Int64,1}; real_transcost=0.0)
  samps = size(rets,2)
  if samps != nstages(model)
    error("Return series has to have $(nstages(model)) and has $(nsamp) samples, use simulatesw if you want to really do that.")
  end
  x, x0, exp_ret, u = forward!(model, states, rets, real_transcost=real_transcost)

  return x, x0, exp_ret
end

# Simulate using sliding windows
function simulatesw(model, rets::Array{Float64,2}, states::Array{Int64,1}; real_transcost=0.0)
   mcopy = deepcopy(model)
   stages_test = size(rets,2)
   nstag = nstages(model)
   its=floor(Int,(stages_test-1)/(nstages(model)-1))

   all_x0, all_x = inialloct(mcopy)
   for i = 1:its
     rets_forward     = rets[:,(i-1)*(nstages(mcopy)-1)+1:(i)*(nstages(mcopy)-1)+1]
     states_forward_a = states[(i-1)*(nstages(mcopy)-1)+1:(i)*(nstages(mcopy)-1)+1]

     x, x0, expret, u = forward!(mcopy, states_forward_a, rets_forward, real_transcost=real_transcost)
     all_x = hcat(all_x, x[:,2:end])
     all_x0 = vcat(all_x0, x0[2:end])
     inialloc!(mcopy, x[:,end], x0[end])
   end

   # Simulate last periods
   diff_t = round(Int, stages_test-1 - (its*(nstages(mcopy)-1)))
   if diff_t > 0
     rets_forward_a   = zeros(nassets(mcopy),diff_t)
     states_forward_a = Array(Int64,diff_t)
     rets_forward_a   = rets[:,its*(nstages(mcopy)-1)+1:end]
     states_forward_a = states[its*(nstages(mcopy)-1)+1:end]
     x, x0, expret, u = forward!(mcopy, states_forward_a, rets_forward_a; nstag=diff_t +1, real_transcost=real_transcost)
     all_x = hcat(all_x, x[:,2:end])
     all_x0 = vcat(all_x0, x0[2:end])
   end

   return all_x, all_x0
 end

function simulate_percport(model, rets::Array{Float64,2}, x_p::Array{Float64,1})
   stages_test = size(rets,2)
   x = Array(Float64,nassets(model)+1, stages_test)
   x[:,1] = inialloc(model)
   cost = 0.0
   for t = 2:stages_test
     # Evaluate transaction costs
     total = sum(x[:,t-1])
     cost = 0.0
     for i = 2:nassets(model)+1
       cost += abs(x[i,t-1]-total*x_p[i])*transcost(model)
       x[i,t] = total*x_p[i]
     end
     x[1,t] = total*x_p[1] - cost

     # Evalute the return
     for i = 2:nassets(model)+1
       x[i,t] = (1+rets[i-1,t])*x[i,t]
     end

   end
   return x
 end

end # SDDP Module
