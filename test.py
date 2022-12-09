import numpy as np

a1 = np.arange(25).reshape(5,5)

cord_1 = np.array([2,2])
diff = np.array([1,1])
print(a1[tuple(cord_1+diff)],a1)