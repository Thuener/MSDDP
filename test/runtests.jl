tests = ["lhs",
         "msddp",
         "hmm_msddp",
         "inputs",
         "realsimulate",
         "sdp"]

@sync @parallel for t in tests
    fp = "$(t).jl"
    println("running $(fp) ...")
    @time evalfile(fp)
end
