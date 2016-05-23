tests = ["hmm",
         "lhs",
         "msddp",
         "hmm_msddp",
         "inputs",
         "realsimulate"]

for t in tests
    fp = "$(t).jl"
    println("running $(fp) ...")
    evalfile(fp)
end
