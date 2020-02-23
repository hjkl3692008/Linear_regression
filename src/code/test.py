import numpy as np
import linear_tools as lt

x = np.array([1,2,3,4,5])
y = np.array([1,1,1,1,1])
# x = np.vstack((x, np.ones(x.shape)))
# w = np.array([0,0])
#
# k = np.dot(w.T, x)
# loss = y - k
# m = np.dot(loss, x.T) / 2
#
# print(k)
# print(m)


house_data = np.vstack((y, x))

w = lt.linear_regression(house_data.T)
print(w)
