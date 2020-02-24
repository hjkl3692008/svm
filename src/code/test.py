import numpy as np

import svm_tools as st

x = np.array([[3,3],[4,3],[1,1]])
c = np.array([1,1,-1])
a = np.array([0.25,0,0.25])

w = st.calculate_w(x, c, a)
print(w)

x = np.array([3,3])
b = np.dot(w, x)
print(b)