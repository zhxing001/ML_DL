# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:08:43 2017
一些关于反向传播算法的练习程序
@author: zhxing
"""
#1.  设置输入
x=-2;y=5;z=-4
#进行前向传播，即计算函数的值
q=x+y    #q  3
f=q*z    #f  -12

dqdx=1;
dqdy=1;    #q对x和y的梯度

#进行逆向传播，首先处理f=q*z
dfdz=q   #所以关于z的梯度是3
dfdq=z   #所以关于q的梯度是-4

dfdx=dfdq*dqdx
dfdy=dfdq*dqdy

print("dfdz,dfdx,dfdy分别是：")
print(dfdz,dfdx,dfdy)

#2.--sigmoid函数的反向传播
import math
w=[2,-3,-3]
x=[-1,-2]
dot=w[0]*x[1]+w[1]*x[1]+w[2]
f=1.0/(1+math.exp(-dot))

ddot=(1-f)*f    #函数求导
#回传
dx=[w[0]*ddot,w[1]*ddot]    
dw=[x[0]*ddot,x[1]*ddot,1*ddot]
print(dx)
print(dw)


import numpy as np
W=np.random.randn(5,10)
X=np.random.randn(10,3)
D=W.dot(X)

#假设我们得到了D的梯度
dD=np.random.randn(*D.shape)
dW=dD.dot(X.T)
dX=W.T.dot(dD)
print(dW)
print(dX)