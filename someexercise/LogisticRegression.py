import numpy as np
import random
import matplotlib.pyplot as plt


def gradientDescent(x,y,theta,alpha,m,numIterations):
	#alpha是学习率，m是实例个数，numIterations是迭代次数
    xTrans=x.transpose()
    for i in range(0,numIterations):
		hypothesis = np.dot(x, theta)

	return theta




def genData(numPoints,bias,variance):
	x=np.zeros(shape=(numPoints,2))
	y=np.zeros(shape=numPoints)

	for i in range(0,numPoints):
		x[i][0]=1;
		x[i][1]=i;

		y[i]=(i+bias)+random.uniform(0,1)*variance
	return x,y


x,y=genData(100,20,10)
#==============================================================================
# print(x)
# plt.plot(x)
# plt.plot(y)
#==============================================================================
