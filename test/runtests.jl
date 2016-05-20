using Logging
@Logging.configure(level=Logging.INFO)

Logging.info("Testing H2SDDP.jl")
include("H2SDDP.jl")
Logging.info("Testing InstH2SDDP.jl")
include("InstH2SDDP.jl")
Logging.info("Testing C++inputs.jl")
include("C++inputs.jl")
Logging.info("Testing RealSimulate.jl")
include("RealSimulate.jl")
