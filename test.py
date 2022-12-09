import numpy as np

a1 = np.arange(5)

print(a1.reshape(-1,1) @ a1.reshape(1,-1))