import numpy as np
from scipy.sparse import coo_matrix

def nan_to_csr(data, mask):
  
  indices = np.nonzero((mask == 1))
  sparse_data_coo = coo_matrix((data[indices], indices), shape=data.shape)
  sparse_data_csr = sparse_data_coo.tocsr()
  
  return sparse_data_csr