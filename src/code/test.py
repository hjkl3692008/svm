import numpy as np

import svm_tools as st
import smo

x = np.array([[3,3],[4,3],[1,1]])
c = np.array([1,1,-1])
a = np.array([0.25,0,0.25])

w = smo.update_w(x, c, a)
print(w)
print(len(w))
print(np.dot(w,w))

x = np.array([3,3])
b = np.dot(w, x)
print(b)


L, H = smo.upper_lower_bound(10, 2, 3, 1, -1)
print(L)
print(H)

n = np.array([1,1,-1])
nm = np.vstack((c, n))
print(nm)
