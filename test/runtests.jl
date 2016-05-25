tests = ["lhs",
         "msddp",
         "hmm_msddp",
         "inputs",
         "realsimulate"]

for t in tests
    fp = "$(t).jl"
    println("running $(fp) ...")
    @time evalfile(fp)
end
