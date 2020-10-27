import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def featureMappingExample():
	numPnt= 500
	data  = np.random.multivariate_normal([1,1], np.eye(2), numPnt)
	t = np.sqrt(np.sum(data*data, axis =1))
	c = np.zeros(t.shape)
	c[t > 1] = 1
	c[t <= 1] = 0
	fig = plt.figure()
	plt.scatter(data[:,0],data[:,1],c=t, linewidth=0)
	fig = plt.figure()
	plt.scatter(data[:,0],data[:,1],c=c, linewidth=0)

	p = np.empty([numPnt,3])
	for i in range(len(t)):
		p[i,:] = (data[i,0]**2, math.sqrt(2)*data[i,0]*data[i,1], data[i,1]**2)

	fig = plt.figure()
	ax = fig.add_subplot(*[1,1,1], projection='3d')
	ax.scatter(p[:,0], p[:,1], p[:,2],c=c,linewidth=0) 
	plt.show();


if __name__ == '__main__':
	featureMappingExample()
