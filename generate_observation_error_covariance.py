# 
# load satellite track observation data (of fine and coarse resolutions)
# compute covariance matrix
# the method here is like the eof computation
#
from relative_import_header import *
from data_assimilation_utilities import data_assimilation_workspace as da_workspace
import numpy as np
from scipy.linalg import ldl

wspace = da_workspace('/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin')

cdata = np.load(wspace.output_name('obs_data.npz', 'cPDESolution'))
fdata = np.load(wspace.output_name('obs_data_fs.npz', 'PDESolution'))

data = np.zeros((len(cdata), cdata['obs_data_0'].shape[0]))
#data = np.zeros((len(cdata), 10))
for index in range(len(cdata)):
    _key = 'obs_data_{}'.format(index)
    #print(fdata[_key].shape, data[index].shape)

    assert len(data[index, :]) == cdata[_key].shape[0]
    assert len(data[index, :]) == fdata[_key].shape[0]

    data[index,:] += fdata[_key] - cdata[_key]

obs_cov = np.matmul(data.T, data)
#obs_cov /= len(cdata)  # sample cov averaging
normalisation_const = len(cdata)

#print(obs_cov)

#obs_cov_inv = np.linalg.pinv(obs_cov, rcond=1e-15, hermitian=True)

#print(np.matmul(obs_cov_inv, obs_cov))

#np.save(wspace.output_name('obs_cov_pseudo_inv', 'ParamFiles'), obs_cov_inv)
print("matrix_rank: ", np.linalg.matrix_rank(obs_cov))
lu, d, perm = ldl(obs_cov)
d_serial = np.diag(d)
print(np.max(d_serial))
print(np.min(d_serial))
rank = len(d_serial[d_serial > 1e-7])
#rank = np.linalg.matrix_rank(obs_cov)

# within rank, we want to subsample 200 satellite observation points
subsample_indices = None
max_sub_rank = 51 #250
total_number_samples_per_track = 50 # 3 * max_sub_rank 
if rank > max_sub_rank:
    _step = int(np.ceil(total_number_samples_per_track / max_sub_rank))
else:
    _step = 1
    max_sub_rank = rank

#print(max_sub_rank)
subsample_indices = np.arange(0, total_number_samples_per_track, _step)    
#subsample_indices = np.arange(0, 200)


print("d_serial rank:" , rank, max_sub_rank)
print('perm shape: ', perm.shape)

perm_inv = np.zeros(perm.shape, dtype=np.int64)
for i in range(len(perm)):
    perm_inv[perm[i]] = i

P = np.eye(len(perm))[perm,:]

#cov_sub_matrix_unnormalised = P.dot(obs_cov).dot(P.T)[:rank, :rank]
cov_sub_matrix_unnormalised = P.dot(obs_cov).dot(P.T)[subsample_indices][:,subsample_indices]
#print(cov_sub_matrix_unnormalised.shape)
cov_sub_matrix_inv = np.linalg.inv(cov_sub_matrix_unnormalised)  # no need for normalisation const here
#print("norm of cov_sub_matrix_inv:", np.linalg.norm(cov_sub_matrix_inv, np.inf))
print("2-norm of cov_sub_matrix_inv:", np.linalg.norm(cov_sub_matrix_inv, 2))

obs_cov /= normalisation_const
print("obs_cov shape", obs_cov.shape)
print("obs_cov_sub_shape", cov_sub_matrix_unnormalised.shape)
if 1:
    np.save(wspace.output_name('obs_cov', 'ParamFiles'), obs_cov)
    np.save(wspace.output_name('obs_cov_sub_matrix_inv', 'ParamFiles'), cov_sub_matrix_inv)
    np.save(wspace.output_name('obs_cov_sub_matrix', 'ParamFiles'), cov_sub_matrix_unnormalised / normalisation_const)
    #np.save(wspace.output_name('sub_obs_site_indices', 'ParamFiles'), perm[:rank])
    #subsample_indices = np.arange(0, 1791, 1) 
    np.save(wspace.output_name('sub_obs_site_indices', 'ParamFiles'), subsample_indices)
    #print(perm_inv[:rank])
    #np.savetxt(wspace.output_name('sub_obs_site_indices.csv', 'ParamFiles'), perm_inv[:rank], delimiter=',')
    #
_proper_indices = np.arange(0,rank)
_diff = perm[:rank] - _proper_indices
print("swapped positions:", _diff[_diff > 0])

if 0:
    d_ = np.copy(d)

    sum_ = 0
    num_off_diag = 0
    _sum_max = 0    # maximum nonzero value in the off diag
    _sum_min = 100  # minimum nonzero value in the off diag
    _sum_max_ij = [0,0]

    for i in range(len(d_)):
        for j in range(len(d_)):
            if not (i == j):
                sum_ += d_[i,j]
                num_off_diag +=1

                if d_[i,j] > _sum_max:
                    _sum_max_ij = [i, j]
                    _sum_max = d_[i,j]

                if d_[i,j] > 0:
                    _sum_min = min(d_[i,j], _sum_min)
            else:
                pass

    print("sum_:", sum_)
    print("_sum_max, _sum_min:",_sum_max, _sum_min)
    print("_sum_max and ij: ", _sum_max, _sum_max_ij)
    print(d_[_sum_max_ij[0], _sum_max_ij[1]], _sum_max_ij[0], _sum_max_ij[1])



