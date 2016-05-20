module HMM
export train_hmm

using PyCall
@pyimport numpy as np
@pyimport hmmlearn.hmm as hl_hmm

function train_hmm(data, n_states, lst; cov_type="full",init_p="stmc")
	model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,init_params=init_p)
	model[:fit](data,lst)
end

end # end HMM
