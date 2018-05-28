# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:39:18 2018

@author: zhxing
"""

#boosting tree  李航《统计学习方法》  P150
import numpy as np
def f(x):
    if x<6.5:
        return 6.24
    else:
        return 8.91


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
ms=np.zeros(9)
y1=np.zeros(10)
for i in range(1,10):
    R1=y[0:i]             #分块
    R2=y[i:10]
    c1=R1.mean()          #均值
    c2=R2.mean()
    ms[i-1]=np.dot(R1-c1,R1-c1)+np.dot(R2-c2,R2-c2)
    
for i in range(0,10):
    y1[i]=f(y[i])
    
e=y-y1
print(e)
    


    