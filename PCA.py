import numpy as np
x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]

cov_mat = np.stack((x, y), axis = 0)  
print(cov_mat)
print(np.cov(cov_mat)) 