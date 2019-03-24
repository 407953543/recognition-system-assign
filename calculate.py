import numpy as np

x = np.array([[2,3,2],[1,2,6],[5,7,6]], dtype=np.float64)
mean = np.mean(x, axis=1).reshape(3,1)
print(x)
print(mean)
x[0] -= mean[0]
x[1] -= mean[1]
x[2] -= mean[2]
x = x.T
print(x)
scatterMatrix = np.dot(x.T, x)
print(scatterMatrix)
e2 = np.array([[-0.0681],[0.9948],[0.0763]], dtype=np.float64)
e3 = np.array([[-0.4882],[0.0334],[-0.8721]], dtype=np.float64)
g12 = np.dot(x[0], e2)
g13 = np.dot(x[0], e3)
print(mean+g12*e2+g13*e3)
g22 = np.dot(x[1], e2)
g23 = np.dot(x[1], e3)
print(g22)
print(g23)
print(mean+g22*e2+g23*e3)
g32 = np.dot(x[2], e2)
g33 = np.dot(x[2], e3)
print(g32)
print(g33)
print(mean+g32*e2+g33*e3)
x4 = np.array([2-mean[0],2-mean[1],4-mean[2]], dtype=np.float64).T
g42 = np.dot(x4, e2)
g43 = np.dot(x4, e3)
print(g42)
print(g43)
print(mean+g42*e2+g43*e3)