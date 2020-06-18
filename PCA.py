import numpy as np
from numpy.linalg import eig

x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]

# Calculate the convariance matrix
cov_mat = np.stack((x, y), axis = 0)  
C = np.cov(cov_mat)
print(C)

eig_values, eig_vectors = eig([[C[0][0], C[0][1]], [C[1][0], C[1][1]]])
print(eig_vectors)
print(eig_values)