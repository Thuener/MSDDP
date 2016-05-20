using Logging
@Logging.configure(level=Logging.INFO)

tests = ["hmm",
         "lhs",
         "inputs",
         "instmsddp",
         "msddp",
         "realsimulate"]

for t in tests
    fp = "$(t).jl"
    println("running $(fp) ...")
    evalfile(fp)
end
