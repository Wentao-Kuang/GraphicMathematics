# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:34:38 2015

@author: Kuang
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as pl


"""
compute the distance between two points
"""
def distance(x1,x2):
    distance=abs(x1-x2)
    return distance

"""
Gaussian kernel
R(x)=exp((-(x-c)^2)/2*sigma^2) 
when variance=100 2*sigma^ = 200
"""
def gaussian(distance):
    return np.exp(-(distance/200)**0.00005)

"""
Get the weight of the linear system Ax=b
"""     
def get_weight(x,y):
    l=x.shape[0]
    A=np.zeros(shape=(l,l))
    for i in range(0,l):
        for j in range(0,l):
            A[i,j]=gaussian(distance(x[i],x[j])) 
    weight=np.linalg.solve(A,y)
    print(".................")
    print(A)
    print(y)
    print(weight)
    return weight
"""
give two test arrey
1. Get the weight first, 
2. get y array of RBF_gaussian use x=[0,1,2,3,4..200] and weight, y=xw.
3. finally plot then points and gaussian line 
"""
def test_1_dimension():
    x=np.array([20.,40.,60.,80.,160.,170.])
    y=np.array([2.,7.,8.,9.,8.5,10.])
    w=get_weight(x,y) # get the weight
#    print(w)
    x_g=np.arange(0,200,1) # make a arrey form 1 to 200 to hold the curve
    y_g=np.zeros(200)
    for i in range(0,x_g.shape[0]):
        for l in range(0,x.shape[0]):
            y_g[i]+=(gaussian(distance(x_g[i],x[l])))*w[l]
    pl.ylim((0,14))
    pl.plot(x,y,"or",ms=5)
    pl.plot(x_g,y_g,"-k")
    pl.show()
    
test_1_dimension()

"""
multiquadric kernel
R(x)=sqrt(((x-c)^2)/2*sigma^2+1) 
assume 2*sigma^2 = 10
"""
def multiquadric(distance):
    return np.sqrt((distance/10)**2 + 1)
"""
get multiquadric weight
"""
def get_weight_multiquadric(x,y):
    l=x.shape[0]
    A=np.zeros(shape=(l,l))
    for i in range(0,l):
        for j in range(0,l):
            A[i,j]=multiquadric(distance(x[i],x[j])) 
    weight=np.linalg.solve(A,y)
    return weight
"""
test multiquadric kernel
"""
def test_1_dimension_multiquadric():
    x=np.array([20.,40.,60.,80.,160.,170.])
    y=np.array([2.,7.,8.,9.,8.5,10.])
    w=get_weight_multiquadric(x,y) # get the weight
#    print(w)
    x_g=np.arange(0,200,1) # make a arrey form 1 to 200 to hold the curve
    y_g=np.zeros(200)
    for i in range(0,x_g.shape[0]):
        for l in range(0,x.shape[0]):
            y_g[i]+=(multiquadric(distance(x_g[i],x[l])))*w[l]
    pl.ylim((0,14))
    pl.plot(x,y,"or",ms=5)
    pl.plot(x_g,y_g,"-k")
    pl.show()
    
test_1_dimension_multiquadric()


"""
compute the distance of a n-dimension
"""
def n_distance(x1,x2):
    distance=0.0
    if(x1.shape==x2.shape):
        for i in range(0,x1.shape[0]):
            distance+=(x1[i]-x2[i])**2
    else:
        print("Please input the same shape matrix")
    if(distance!=0):
        distance=np.sqrt(distance)
    else:
        distance=0
    return distance
    
"""
get the weight
"""
def n_get_weight(x,y):
    l=x.shape[0]
    A=np.zeros(shape=(l,l))
    for i in range(0,l):
        for j in range(0,l):
            A[i,j]=gaussian(n_distance(x[i],x[j])) 
#    print(A)
    weight=np.linalg.solve(A,y)

    return weight
    
"""
test 
1.input a 2 dimension example 
    n_xy = np.array([[0,0],[100,0],[0,100],[100,100],[50,50]])
    n_z = np.array( [ 0., 0, 0, 0, 10 ])
"""
def test_ndimension():
#    n_xy=np.array([[40,20],[50,30],[70,80],[80,100]])
#    n_z=np.array([2,7,5,9])
    n_xy = np.array([[0,0],[100,0],[0,100],[100,100],[50,50]])
    n_z = np.array( [ 0., 0, 0, 0, 10 ])
    w=n_get_weight(n_xy,n_z)
    test_x=np.arange(0,100,1)
    test_y=np.arange(0,100,1)
    test_z=np.zeros((100,100))
    for i in range(0,test_z.shape[0]):
        for j in range(0,test_z.shape[0]):
            for l in range(0,n_xy.shape[0]):
                test_z[i][j]+=(gaussian(n_distance(np.array([i,j]),n_xy[l])))*w[l]  
    X, Y = np.meshgrid(test_x, test_y)
#    print(test_z.shape)
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, test_z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pl.show()
    im = pl.imshow(test_z, cmap='hot')
    pl.colorbar(im, orientation='horizontal')
    pl.show()
test_ndimension()