module MSDDP

using JuMP, CPLEX
using Distributions, MathProgBase
using Logging, JLD

# Types
export MKData, MAAParameters, SDDPParameters, ModelSizes, MSDDPModel

export solve, simulate, simulatesw, simulate_stateprob, simulatestates!, simulate_percport,  createmodels!, reset!,
loadcuts!, param, sizes, markov
export nstages, nassets, nstates, nscen, transcost, probscen
export setnstages!, setnstates!, setnassets!, setnscen!, setα!, setmarkov!, setγ!, setinistate!, settranscost!
export chgrrhs!, getdual, getduals
export memuse

export Subproblem, State
export AbstractMSDDPModel, AbstractSubproblem, AbstractState

abstract type AbstractMSDDPModel end
abstract type AbstractSubproblem end
abstract type AbstractState end


## Types ##
type State <: AbstractState
    x_trial::Array{Float64,2}
    x0_trial::Array{Float64,1}
end

risk_alloc(st::State) = st.x_trial
rf_alloc(st::State)   = st.x0_trial

type Subproblem <: AbstractSubproblem
    jmodel::JuMP.Model
    # Ids for the constraints
    cons::Vector{Int64}
    low::Bool # use low level api
    dic::Dict
end

function Subproblem(jmodel, cons, low)
    return Subproblem(jmodel, cons, low, Dict())
end

function Base.copy(sp::Subproblem)
    jmodel = copy(sp.jmodel)
    cons   = deepcopy(sp.cons)
    low    = copy(sp.low)
    dic    = deepcopy(sp.dic)

    new_sb = Subproblem(jmodel, cons, low, dic)
    return new_sb
end

type Stage
    subproblems::Vector{AbstractSubproblem}
end

" Markov Chain data "
type MKData
    inistate::Int64                         # Initial state
    transprob::Array{Float64,2}             # Transtition probability
    prob_scenario_state::Array{Float64,2}   # Probability of each scenario given a state
    ret::Array{Float64,4}                   # Return with stages, assets, states and samples
end

#  Utils for MKData
inistate(mkd::MKData)              = mkd.inistate
transprob(mkd::MKData)             = mkd.transprob
transprob(mkd::MKData, state::Int) = transprob(mkd)[state,:]
probscen(mkd::MKData)              = mkd.prob_scenario_state
probscen(mkd::MKData, scen::Int, state::Int) = probscen(mkd)[scen, state]


returns(mkd::MKData)               = mkd.ret
returns(mkd::MKData, stage::Int)   = returns(mkd)[stage,:,:,:]
returns(mkd::MKData, stage::Int, asset::Int, state::Int, sample::Int)  = returns(mkd)[stage, asset, state, sample]

setinistate!(mkd::MKData, state::Int) = mkd.inistate = state

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
    nit_before_lower::Int64         # Number of forwards and backwards before evaluate lower bound
    gap::Float64                    # Minimum percentage gap between the lower and upper bounds (%)
    α_lower::Float64                # Confidence level for the lower bound
    diff_upper::Float64             # Difference in the upper bound to define stabilization
    print_level::Int                # Log print level
    lowlevel_api::Bool              # Use or not low level CPLEX api
    parallel::Bool                  # Solve the SDDP in parallel
    fast_lower::Bool                # Use the fast evaluation of the lower bound
    file::String                    # Output JLD file
    stabtype::Int                   # Stabilization type, (default) 0 to use diff_upper as percetage or 1 to use as absolute value
end

type ModelSizes
    nstages::Int64
    nassets::Int64
    nstates::Int64
    nscen::Int64
end

type MSDDPModel <: AbstractMSDDPModel
    sizes::ModelSizes
    lpsolver::JuMP.MathProgBase.AbstractMathProgSolver
    asset_parameters::MAAParameters
    param::SDDPParameters
    markov_data::MKData
    stages::Vector{Stage}
    states::Vector{State}
    low::Bool
end


## Util functions ##

SDDPParameters(max_it::Int64, samplower::Int64, samplower_inc::Int64, nit_before_lower::Int64, gap::Float64, α_lower::Float64;
    diff_upper= 0.5, print_level=0, lowlevel_api=true, parallel=false, fast_lower=true, file= "" ) =
    SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower, diff_upper,
        print_level, lowlevel_api, parallel, fast_lower, file)

SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower, diff_upper,
    print_level, lowlevel_api, parallel, fast_lower, file) =
    SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower, diff_upper,
        print_level, lowlevel_api, parallel, fast_lower, file, 0)

" Verify stabilization of the upper bound "
function checkstab(newub::Float64, lastub::Float64, param::SDDPParameters)
    # Uses diff_upper as diference percentage
    param.stabtype == 0 &&
        return abs(newub/lastub -1)*100 < param.diff_upper
    # Uses diff_upper as absolute diference
    param.stabtype == 1 &&
        return abs(newub -lastub) < param.diff_upper
end

MSDDPModel(sizes, asset_parameters, param, markov_data, lpsolver, stages) =
    MSDDPModel(sizes, asset_parameters, param, markov_data, lpsolver, stages, [], true)

" Construct the MSDDPModel without ModelSizes "
function MSDDPModel(asset_parameters::MAAParameters,
        param::SDDPParameters,
        markov_data::MKData;
        nassets  = length(asset_parameters.iniassets),
        nstages  = size(markov_data.ret, 1),
        nstates  = size(markov_data.transprob, 1),
        nscen    = size(markov_data.prob_scenario_state, 1),
        lpsolver = ClpSolver(),
        low = true )
    MSDDPModel(ModelSizes(nstages, nassets, nstates, nscen),
        asset_parameters, param, markov_data, lpsolver, low)
end

" Construct the MSDDPModel using ModelSizes "
function MSDDPModel(msize::ModelSizes,
        asset_parameters::MAAParameters,
        param::SDDPParameters,
        markov_data::MKData,
        lpsolver::JuMP.MathProgBase.AbstractMathProgSolver,
        low::Bool)
    stages = Vector{Stage}(nstages(msize))
    for t = 1:nstages(msize)
        stages[t] = Stage(Vector{AbstractSubproblem}(nstates(msize)))
    end
    m = MSDDPModel(msize, lpsolver, asset_parameters, param, markov_data, stages, [], low)
    createmodels!(m)
    return m
end

" Copy stages vector "
function copy_stages(msize::ModelSizes, stages::Vector{Stage})
    new_stages = Vector{Stage}(nstages(msize))
    for t = 1:nstages(msize)
        new_stages[t] = Stage(Vector{AbstractSubproblem}(nstates(msize)))
    end

    for t = 1:nstages(msize)-1, j = 1:nstates(msize)
        new_stages[t].subproblems[j] = copy(stages[t].subproblems[j])
    end

    return new_stages
end

function Base.copy(model::MSDDPModel)
    sizes    = deepcopy(model.sizes)
    lpsolver = deepcopy(model.lpsolver)
    asset_pr = deepcopy(model.asset_parameters)
    param    = deepcopy(model.param)
    mar_data = deepcopy(model.markov_data)
    stages   = copy_stages(sizes, model.stages)
    states   = deepcopy(model.states)
    low      = copy(model.low)

    new_model = MSDDPModel(sizes, lpsolver, asset_pr, param, mar_data, stages, states, low)
    return new_model
end

" Reset the stages vector inside MSDDPModel and create models"
function reset!(m::MSDDPModel)
    stages = Vector{Stage}(nstages(m))
    for t = 1:nstages(m)
        stages[t] = Stage(Vector{AbstractSubproblem}(nstates(m)))
    end
    m.stages = stages
    m.states = []
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
sizes(m::MSDDPModel)   = m.sizes
markov(m::MSDDPModel)  = m.markov_data
assetspar(m::MSDDPModel)   = m.asset_parameters
low(m::MSDDPModel)     = m.low # user or not low level API

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
transcost(m::MSDDPModel)             = m.asset_parameters.transcost

stage(m::MSDDPModel, stage::Int)     = m.stages[stage]
subproblem(m::MSDDPModel, stage::Int, state::Int) = m.stages[stage].subproblems[state]

maximum(ap::MAAParameters)           = ap.maximum
jumpmodel(sp::Subproblem)            = sp.jmodel

setnstages!(m::MSDDPModel, n::Int)   = m.sizes.nstages = n
setnassets!(m::MSDDPModel, n::Int)   = m.sizes.nassets = n
setnstates!(m::MSDDPModel, n::Int)   = m.sizes.nstates = n
setnscen!(m::MSDDPModel, n::Int)     = m.sizes.nscen = n

setmarkov!(m::MSDDPModel, mk::MKData)= m.markov_data = mk
setγ!(m::MSDDPModel, γ::Float64)     = m.asset_parameters.γ = γ
setα!(m::MSDDPModel, α::Float64)     = m.asset_parameters.α = α
settranscost!(m::MSDDPModel, tc::Float64) = m.asset_parameters.transcost = tc

function memuse()
  pid = getpid()
  return string(round(Int,parse(Int,readstring(`ps -p $pid -o rss=`))/1024),"M")
end

############ hight level functions ############
" Change the RHS using the index of the constraint "
function chgrrhs_high!(sp::JuMP.Model, idx::Int64, rhs::Number)
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
    return nothing
end

function getdual_high(sp::JuMP.Model, idx::Int64)
    if length(sp.linconstrDuals) != MathProgBase.numlinconstr(sp)
        error("Dual solution not available. Check that the model was properly solved and no integer variables are present.")
    end
    return sp.linconstrDuals[idx]
end

function getduals_high(sp::JuMP.Model, idx::Vector{Int})
    if length(sp.linconstrDuals) != MathProgBase.numlinconstr(sp)
        error("Dual solution not available. Check that the model was properly solved and no integer variables are present.")
    end
    return sp.linconstrDuals[idx]
end

function immediatebenefit_high(model::AbstractMSDDPModel, sp::JuMP.Model, state::Int, rets::Array{Float64,3})
    b = getindex(sp,:b)
    d = getindex(sp,:d)
    u = getindex(sp,:u)
    stateprob = transprob(markov(model), state)
    @expression(sp, B_imed, - transcost(model)*sum(b[i] + d[i] for i = 1:nassets(model)) +
             sum((sum(rets[i,k,s]*u[i] for i = 1:nassets(model)) )*stateprob[k]*probscen(markov(model), s, k)
             for k = 1:nstates(model), s = 1:nscen(model)))
    return JuMP.getvalue(B_imed)
end

function addcut_high!(model::AbstractMSDDPModel, α::Array{Float64,2}, β::Array{Float64,3}, stage::Int, state::Int)
    jsp = jumpmodel(subproblem(model, stage-1, state))
    θ = getindex(jsp,:θ)
    u = getindex(jsp,:u)
    u0 = getindex(jsp,:u0)

    @constraint(jsp, [j = 1:nstates(model), s = 1:nscen(model)],
    θ[j,s] <= α[stage,j] + β[1,stage,j]*u0 + sum(β[i+1,stage,j]*
        (1+returns(markov(model),stage+1,i,j,s))*u[i] for i = 1:nassets(model)))
    return nothing
end


############ Low level functions ############
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

function getobjectivevalue_low(sp::JuMP.Model)
    CPLEX.get_objval(sp.internalModel.inner)
end

function getvalue_low(sp::JuMP.Model, idx::Int64)
    return CPLEX.get_solution(sp.internalModel.inner)[idx]
end

function getvalue_low(sp::JuMP.Model, ridx::UnitRange{Int64})
    vidx = collect(ridx)
    ret = Array{Float64}(length(vidx))
    values = CPLEX.get_solution(sp.internalModel.inner)
    for id = 1:length(vidx)
        ret[id] = values[vidx[id]]
    end
    return ret
end

function getvalue_low(sp::JuMP.Model, ridx::Array{Int64})
    ret = Array{Float64}(length(ridx))
    values = CPLEX.get_solution(sp.internalModel.inner)
    for id = 1:length(ridx)
        ret[id] = values[ridx[id]]
    end
    return ret
end

" Change constraint RHS using the Cplex low level function "
function chgrrhs_low!(sp::JuMP.Model, idx::Int, rhs::Number)
    cmodel = sp.internalModel.inner
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
    return nothing
end

" Evalute immediate benefit using low level Cplex api "
function immediatebenefit_low(model::AbstractMSDDPModel, sp::JuMP.Model, state::Int, rets::Array{Float64,3})

    u = getvalue_low(sp, 2:nassets(model)+1)
    b = getvalue_low(sp, 2+nassets(model):2*nassets(model)+1)
    d = getvalue_low(sp, 2*(1+nassets(model)):3*nassets(model)+1)

    stateprob = transprob(markov(model), state)
    B_imed = - transcost(model)*sum(b[i] + d[i] for i = 1:nassets(model)) +
             sum((sum(rets[i,k,s]*u[i] for i = 1:nassets(model)) )*stateprob[k]*probscen(markov(model), s, k)
             for k = 1:nstates(model), s = 1:nscen(model))
    return B_imed
end

function addcut_low!(model::AbstractMSDDPModel, α::Array{Float64,2}, β::Array{Float64,3}, stage::Int, state::Int)
    jsp = jumpmodel(subproblem(model, stage-1, state))
    nvariables = CPLEX.num_var(jsp.internalModel.inner)
    u0_id = 1
    u_ids = collect(2:nassets(model)+1)

    coef = zeros(Float64, nstates(model)*nscen(model), nvariables)
    rhs = zeros(Float64, nstates(model)*nscen(model))
    ind_ini = nvariables - nstates(model)*nscen(model)

    for j = 1:nstates(model), s = 1:nscen(model)
        ind_const = s + (j-1)*nscen(model)
        ind_θ = ind_ini + ind_const

        coef[ind_const, ind_θ] = 1
        coef[ind_const, u0_id] = -β[1,stage,j]
        for i = 1:nassets(model)
            coef[ind_const, u_ids[i]] = -β[i+1,stage,j]*(1+returns(markov(model),stage+1,i,j,s))
        end
        rhs[ind_const] = α[stage,j]
    end
    CPLEX.add_constrs!(jsp.internalModel.inner, coef, '<', rhs)
    return nothing
end

############ Functions to choose between low level api or high ############

function getdual(sp::Subproblem, idx::Int64)
    if sp.low
        return getdual_low(jumpmodel(sp), idx)
    end
    return getdual_high(jumpmodel(sp), idx)
end

function getduals(sp::Subproblem, idxs::Array{Int64})
    if sp.low
        return getduals_low(jumpmodel(sp), idxs)
    end
    return getduals_high(jumpmodel(sp), idxs)
end

function solve(sp::Subproblem)
    if sp.low
        return solve_low(jumpmodel(sp))
    end
    return JuMP.solve(jumpmodel(sp))
end

function getobjectivevalue(sp::Subproblem)
    if sp.low
        return getobjectivevalue_low(jumpmodel(sp))
    end
    return JuMP.getobjectivevalue(jumpmodel(sp))
end

function getvalue(sp::Subproblem, var::Symbol)
    if sp.low
        vars = jumpmodel(sp)[var]
        if typeof(vars) != JuMP.Variable
            idxs = Array{Int64}(length(vars))
            for i = 1:length(vars)
                idxs[i] = vars[i].col
            end
        else
            idxs = vars.col
        end
        return getvalue_low(jumpmodel(sp), idxs)
    end
    return JuMP.getvalue(jumpmodel(sp)[var])
end

function chgrrhs!(sp::Subproblem, idx::Int, rhs::Number)
    if sp.low
        return chgrrhs_low!(jumpmodel(sp), idx, rhs)
    end
    return chgrrhs_high!(jumpmodel(sp), idx, rhs)
end

function immediatebenefit(model, sp::Subproblem, state, rets::Array{Float64,3})
    if sp.low
        return immediatebenefit_low(model, jumpmodel(sp), state, rets)
    end
    return immediatebenefit_high(model, jumpmodel(sp), state, rets)
end

function addcut!(model, α::Array{Float64,2}, β::Array{Float64,3}, stage::Int, state::Int)
    if low(model)
        return addcut_low!(model, α, β, stage, state)
    end
    return addcut_high!(model, α, β, stage, state)
end

" Add subproblem to model "
function createmodel!(m::MSDDPModel, stage::Int, state::Int)
    rets = returns(markov(m), stage+1)
    probstates = transprob(markov(m), state)
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
        sum((sum(rets[i,k,s]*u[i] for i = 1:nassets(m)) + θ[k,s])*probstates[k]*probscen(markov(m), s, k)
        for k = 1:nstates(m), s = 1:nscen(m)))

    cash = @constraint(jmodel, u0 + sum((1+transcost(m))*b[i] - (1-transcost(m))*d[i] for i = 1:nassets(m)) == inirf(m)).idx
    assets = Array{Int64}(nassets(m))
    for i = 1:nassets(m)
        assets[i] = @constraint(jmodel, u[i] - b[i] + d[i] == iniassets(m, i)).idx
    end
    risk =  @constraint(jmodel,-(z - sum(probstates[k]*probscen(markov(m), s, k)*y[k, s] for k = 1:nstates(m), s = 1:nscen(m))/(1-ap.α))
                          + transcost(m)*sum(b[i] + d[i] for i = 1:nassets(m)) <= ap.γ*initwealth(m)).idx

    @constraint(jmodel, [k = 1:nstates(m), s = 1:nscen(m)], y[k,s] >= z - sum(rets[i,k,s]*u[i] for i = 1:nassets(m)))

    if stage == nstages(m)-1
        @constraint(jmodel, [k = 1:nstates(m), s = 1:nscen(m)], θ[k,s] ==  0)
    end

    status = JuMP.solve(jmodel)
    if status ≠ :Optimal
        writeLP(jmodel,"prob.lp";genericnames=false)
        error("Can't solve the problem status:",status)
    end
    m.stages[stage].subproblems[state] = Subproblem(jmodel, [cash; risk; assets], low(m))
end

function createmodels!(model::AbstractMSDDPModel)
    for t = 1:nstages(model)-1
        for j = 1:nstates(model)
            createmodel!(model, t, j)
        end
    end
end

function simulatestates!(states_forward::Vector{Int64}, rets_forward::Array{Float64,2}, model::AbstractMSDDPModel)
    simulatestates!(states_forward, rets_forward, sizes(model), markov(model))
end

" Simulating forward states "
function simulatestates!(states_forward::Vector{Int64}, rets_forward::Array{Float64,2}, sz::ModelSizes, mk::MKData)
  states_forward[1] = inistate(mk)

  for t = 2:nstages(sz)
    # Simulating states
    prob_trans = vec(transprob(mk)[states_forward[t-1],1:nstates(sz)])
    states_forward[t] = rand(Categorical(prob_trans))

    # Simulationg forward scenarios
    rand_idx = rand(Categorical(probscen(mk)[1:nscen(sz),states_forward[t]]))
    rets_forward[1:nassets(sz),t] = returns(mk)[t,1:nassets(sz),states_forward[t],rand_idx]
  end
end

function forward!(model::AbstractMSDDPModel, states::Vector{Int64}, rets::Array{Float64,2};
        nstag = nstages(msddp(model)), real_transcost=0.0, simulate=false)
    m = msddp(model)

    # Initialize
    x_trial = zeros(Float64, nassets(m), nstag)
    x0_trial = zeros(Float64, nstag)
    x_trial[1:nassets(m), 1] = iniassets(m)
    x0_trial[1] = inirf(m)
    u_trial = zeros(nassets(m)+1, nstag)
    ap = assetspar(m)
    obj_forward = 0.0
    for t = 1:nstag-1
        k = states[t]
        sp = subproblem(m, t, k)

        chgrrhs!(sp, sp.cons[1], x0_trial[t])
        chgrrhs!(sp, sp.cons[2], ap.γ*(sum(x_trial[:,t])+x0_trial[t]) )
        for i = 1:nassets(m)
            chgrrhs!(sp, sp.cons[i+2], x_trial[i,t])
        end
        # Resolve subprob in time t
        status = solve(sp)
        if status ≠ :Optimal
            writeLP(jumpmodel(sp),"prob.lp";genericnames=false)
            error("Can't solve the problem status:",status)
        end
        # Evalute immediate benefit
        obj_forward += immediatebenefit(m, sp, states[t], returns(markov(m),t+1))

        # Update trials
        u_val = getvalue(sp, :u)
        for i = 1:nassets(m)
            u_trial[i+1,t] = u_val[i]
            x_trial[i,t+1] = (1+rets[i,t+1])*u_val[i]
        end
        u_trial[1,t] = getvalue(sp, :u0)

        # If transactional cost is different from the optimization model
        if real_transcost != 0.0
            error("Adjust the issue first")
            b_v = getvalue(sp, :b)
            d_v = getvalue(sp, :d)
            x0_trial[t+1] = - sum((1.0+real_transcost)*b_v) + sum((1.0-real_transcost)*d_v) + x0_trial[t]
        else
            x0_trial[t+1] = getvalue(sp, :u0)
        end
    end

    st = State(x_trial, x0_trial)
    push!(model.states, st)
    return st, obj_forward, u_trial
end

function backward!(model::AbstractMSDDPModel, state::State, cutsfile::String)
    x_trial = risk_alloc(state)
    x0_trial = rf_alloc(state)

    m = msddp(model)
    # Initialize
    α = ones(nstages(m)-1,nstates(m))
    β = ones(nassets(m)+1,nstages(m)-1,nstates(m))
    ap = assetspar(m)

    # Add cuts to t < T-1
    for t = nstages(m)-1:-1:2
        for j = 1:nstates(m)
            sp = subproblem(m, t, j)

            chgrrhs!(sp, sp.cons[1], x0_trial[t])
            chgrrhs!(sp, sp.cons[2], ap.γ*(sum(x_trial[:,t])+x0_trial[t]) )
            for i = 1:nassets(m)
                chgrrhs!(sp, sp.cons[i+2], x_trial[i,t])
            end

            status = solve(sp)
            if status ≠ :Optimal
                writeLP(jumpmodel(sp),"prob.lp";genericnames=false)
                error("Can't solve the problem status:",status)
            end

            # Evalute custs
            λ0 = getdual(sp, sp.cons[1])
            λ = zeros(nassets(m))
            for i = 1:nassets(m)
                λ[i] = getdual(sp, sp.cons[i+2])
            end
            π = getdual(sp, sp.cons[2])
            obj = getobjectivevalue(sp)
            α[t,j] =  obj - (λ0 + ap.γ*π)*x0_trial[t] - sum([(λ[i] + ap.γ*π)*x_trial[i,t] for i = 1:nassets(m)])
            β[:,t,j] = vcat(λ0, λ) + ap.γ*π
        end

        for k = 1:nstates(m)
            addcut!(m, α, β, t, k)
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
    nothing
end

" Load cuts into the model "
function loadcuts!(model::AbstractMSDDPModel, file::String)
    open(file,"r") do f
        while true
            line = readline(f)
            line == nothing || line == "" && break
            items = split(line, ",")
            nassets_p1 = parse(Int, items[1])
            nstages = parse(Int, items[2])
            nstates = parse(Int, items[3])
            α = Array{Float64}( nstages-1, nstates)
            β = Array{Float64}( nassets_p1, nstages-1, nstates)
            for t = 1:nstages-1
                line = readline(f)
                items = split(line, ",")
                for j = 1:nstates
                    α[t,j] = parse(Float64,items[j])
                end
            end
            for j = 1:nstates, t = 1:nstages-1, i = 1:nassets_p1
                β[i,t,j] = parse(Float64,readline(f))
            end
            # Add cuts to model
            for t = 1:nstages-1
                if t != 1
                    for k = 1:nstates
                        addcut!(msddp(model), α, β, t, k)
                    end
                end
            end
        end
    end
end

""" Lower bound evaluation and convergence test """
function isconverged(model::AbstractMSDDPModel, p::SDDPParameters, upper::Float64,
        samplower_ini::Int, it_stable::Int)

    debug("Evaluating statistical the Lower Bound memuse $(memuse())")
    lower_conserv = 0.
    gap = 100.
    sumlower =0.
    doub_samplower = false
    quantil = quantile(Normal(),p.α_lower)

    rets_forward = zeros(Float64, nassets(model), nstages(model))
    states_forward = Array{Int64}( nstages(model))
    lowers = (p.parallel ? SharedArray{Float64}(p.samplower):
                Array{Float64}(p.samplower))

    forwards_lower = 0 # number of forwards to evaluate lower bound
    for forwards_lower = 1:p.samplower
        simulatestates!(states_forward, rets_forward, model)
        state, obj_forward, u_trial = forward!(model, states_forward, rets_forward)
        lowers[forwards_lower] = obj_forward
        sumlower += obj_forward

        if p.fast_lower
            # Start testing gap_mean after samplower_ini/3 simulations
            if forwards_lower >= samplower_ini/3
                meanlower = sumlower/forwards_lower
                gap_mean = 100*(upper - meanlower)/upper
                if gap_mean > p.gap*3 && it_stable < 10
                    meanlower = sumlower/forwards_lower
                    lower_conserv = (meanlower - quantil * std(lowers[1:forwards_lower])/sqrt(forwards_lower))
                    gap = 100*(upper - lower_conserv)/upper
                    info("gap_mean $gap_mean is higher than $(p.gap*3) using $forwards_lower Forwards. "*
                        "Aborting lower evaluation.")
                    return false, lowers[1:forwards_lower], lower_conserv, gap
                # If the gap_mean is lower then p.gap( the gap is almost close)
                # doubles the number of scenarios used in lower
                elseif doub_samplower == false && gap_mean < p.gap
                    doub_samplower = true
                    p.samplower = round(Int64,p.samplower + p.samplower_inc)
                    info("gap_mean < p.gap increasing samples for lower bound for $(p.samplower)")
                end
            end
            # Start testing gap after samplower_ini simulations
            if forwards_lower >= samplower_ini
                meanlower = sumlower/forwards_lower
                lower_conserv = (meanlower - quantil * std(lowers[1:forwards_lower])/sqrt(forwards_lower))
                gap = 100*(upper - lower_conserv)/upper
                if abs(gap) < p.gap
                    info("SDDP ended: gap lower $gap is lower than $(p.gap) using $forwards_lower Forwards")
                    return true, lowers[1:forwards_lower], lower_conserv, gap
                end

                gap_mean = 100*(upper - meanlower)/upper
                if gap_mean > p.gap && it_stable < 10
                    info("gap_mean $gap_mean is higher than $(p.gap) upper using $forwards_lower Forwards. "*
                        " Aborting lower evaluation.")
                    return false, lowers[1:forwards_lower], lower_conserv, gap
                end
            end
        else
            # Evaluate the lower bound always using samples for lower bound forwards
            meanlower = sumlower/forwards_lower
            lower_conserv = (meanlower - quantil * std(lowers[1:forwards_lower])/sqrt(forwards_lower))
            gap = 100*(upper - lower_conserv)/upper
            if abs(gap) < p.gap
                info("SDDP ended: gap lower $gap is lower than $(p.gap) using $forwards_lower Forwards")
                return true, lowers[1:forwards_lower], lower_conserv, gap
            end
        end
    end
    return false, lowers[1:forwards_lower], lower_conserv, gap
end

function solve(model::AbstractMSDDPModel, p::SDDPParameters; cutsfile::String = "", timelimit=Inf)
    # Delete cutsfile file
    if cutsfile != ""
        rm(cutsfile; force=true)
    end

    ap = assetspar(model)
    state = []
    u_trial = []
    rets_forward = zeros(Float64, nassets(model), nstages(model))
    states_forward = Array{Int64}( nstages(model))

    gap = 100.0
    it = 0
    samplower_ini = p.samplower

    list_firstu    = inialloc(model)
    list_uppers    = [-1000]
    list_lowers    = [-1000 -1000]
    upper          = 9999999.0
    it_stable      = 0
    upper_last     = 9999999.0
    lower_conserv  = 0.0
    eps_upper      = 1e-6
    lowers         = []

    timeini = time()
    while abs(gap) > p.gap && upper > eps_upper && it < p.max_iterations && (time() - timeini) < timelimit
        it += 1
        tic()
        for it_forward_backward = 1:p.nit_before_lower
            # Forward
            debug("Forward Step memuse $(memuse())")
            simulatestates!(states_forward, rets_forward, model)
            state, obj_forward, u_trial = forward!(model, states_forward, rets_forward)

            # Backward
            debug("Backward Step memuse $(memuse())")
            backward!(model, state, cutsfile)


            # Evaluate upper bound
            sp = subproblem(model, 1, inistate(markov(model)))
            status = solve(sp)
            if status ≠ :Optimal
                writeLP(jumpmodel(sp),"prob.lp";genericnames=false)
                error("Can't solve the problem status:",status)
            end
            upper = getobjectivevalue(sp)
            u = getvalue(sp,:u)
            u0 = getvalue(sp,:u0)

            list_firstu = hcat(list_firstu,vcat(u0,u))

            debug("obj forward = $(@sprintf("%.6f", obj_forward)), upper bound = $(@sprintf("%.6f", upper)),"*
                "stab upper $(@sprintf("%.6f", abs(upper/upper_last -1)*100))")
            if p.fast_lower
                if (checkstab(upper, upper_last, p) || upper < eps_upper || isnan(abs(upper/upper_last -1)*100))
                    it_stable += 1
                    if it_stable >= 5
                        if p.samplower < 30*samplower_ini
                            p.samplower = round(Int64,p.samplower + p.samplower_inc)
                            info("Stable upper bound, increasing samples for lower bound for $(p.samplower)")
                        end
                        upper_last = upper
                        break
                    end
                else
                    it_stable = 0
                    p.samplower = samplower_ini
                end
            end
            upper_last = upper
        end

        # Test convergence
        isconv, lowers, lower_conserv, gap = isconverged(model, p, upper, samplower_ini, it_stable)

        if p.file != ""
            save(string(p.file,"_MSDDP.jld"),"lowers", lowers,"upper", upper,"lower_c", lower_conserv,"x",
            vcat(rf_alloc(state)',risk_alloc(state)), "u", u_trial, "l_lower", list_lowers, "l_upper", list_uppers,
            "l_firsu",list_firstu)
        end

        list_uppers = vcat(list_uppers,upper)
        list_lowers = vcat(list_lowers,[mean(lowers) std(lowers)/sqrt(length(lowers))])

        time_it = toq()
        info("lower bound = $(@sprintf("%.6f", lower_conserv)), upper bound = $(@sprintf("%.6f", upper)),"*
            "gap(%) = $(@sprintf("%.4f", gap)), it: $it time it: $(@sprintf("%.4f", time_it))")
        if isconv
            break
        end
        if it >= p.max_iterations
            info("SDDP ended: maximum number of iterations exceeded")
            break
        end
        if upper <= eps_upper
            info("SDDP ended: upper bound is zero")
            break
        end
        gap = isnan(gap) ? Inf : gap
    end

    p.samplower = samplower_ini
    x_trial_  = length(lowers) != 0 ? vcat(rf_alloc(state)',risk_alloc(state)) : rf_alloc(state)'
    return lowers, upper, lower_conserv, x_trial_, u_trial, list_lowers, list_uppers, list_firstu
end

function simulate_stateprob(model::AbstractMSDDPModel, rets::Array{Float64,2}, probret_state::Array{Float64,2}; real_transcost=0.0)

   states = Array{Int64}(nstages(model)-1)
   for t = 1:nstages(model)-1
     states[t] = findmax(probret_state[:,t])[2]
   end

   return simulate(model, rets, states, real_transcost=real_transcost)
end

function simulate(model::AbstractMSDDPModel, rets::Array{Float64,2}, states::Array{Int64,1}; real_transcost=0.0)
  samps = size(rets,2)
  if samps != nstages(model)
    error("Return series has to have $(nstages(model)) and has $(nsamp) samples, u"*
        "se simulatesw if you want to really do that.")
  end
  state, exp_ret, u = MSDDP.forward!(model, states, rets, real_transcost=real_transcost)

  return risk_alloc(state), rf_alloc(state), exp_ret
end

# Simulate using sliding windows
function simulatesw(model::AbstractMSDDPModel, rets::Array{Float64,2}, states::Array{Int64,1};
    real_transcost=0.0, nstag = nstages(model), reestimate=false, checkgapbefore=false)
   mcopy = copy(model)
   stages_test = size(rets,2)
   its=floor(Int,(stages_test-1)/(nstag-1))

   all_x0, all_x = inialloct(mcopy)
   for i = 1:its
     indranges        = (i-1)*(nstag-1)+1:(i)*(nstag-1)+1
     rets_forward     = rets[:,indranges]
     states_forward_a = states[indranges]

     if reestimate
         debug("Reestimating SDDP, set state to $(states_forward_a[1]). It $(i) of $(its)")
         setinistate!(markov(mcopy), states_forward_a[1])
         if checkgapbefore
             # Test convergence
             isconv, lowers, lower_conserv, gap = isconverged(mcopy,
                param(mcopy), 0.0, param(mcopy).samplower, 0)

             # If din't converge go to solve
             !isconv && solve(mcopy, param(mcopy))
         else
             solve(mcopy, param(mcopy))
         end
     end

     state, = MSDDP.forward!(mcopy, states_forward_a, rets_forward;
        nstag=nstag, real_transcost=real_transcost, simulate=true)
     all_x = hcat(all_x, risk_alloc(state)[:,2:end])
     all_x0 = vcat(all_x0, rf_alloc(state)[2:end])
     inialloc!(mcopy, risk_alloc(state)[:,end], rf_alloc(state)[end])
   end

   # Simulate last periods
   diff_t = round(Int, stages_test-1 - (its*(nstag-1)))
   if diff_t > 0
     states_forward_a = Array{Int64}(diff_t)
     startind         = its*(nstag-1)+1
     rets_forward_a   = rets[:,startind:end]
     states_forward_a = states[startind:end]
     state, = MSDDP.forward!(mcopy, states_forward_a, rets_forward_a;
        nstag=diff_t +1, real_transcost=real_transcost, simulate=true)
     all_x = hcat(all_x, risk_alloc(state)[:,2:end])
     all_x0 = vcat(all_x0, rf_alloc(state)[2:end])
   end

   return all_x, all_x0
end

function simulate_percport(model::AbstractMSDDPModel, rets::Array{Float64,2}, x_p::Array{Float64,1})
   stages_test = size(rets,2)
   x = Array{Float64}(nassets(model)+1, stages_test)
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
