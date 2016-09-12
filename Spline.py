# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:43:08 2015

@author: Kuang
"""

import numpy as np
import matplotlib.pyplot as pl
import scipy.sparse as sp_
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

"""


Spline curve pass (25,1) (50,-1) (60,0.3)
"""
def spline(derivatives):
    #make a 100*100 matrix as derivative operator
    F=np.zeros((100,100))
    for i in range(1,99):
        F[i][i-1]=-1
        F[i][i]=2
        F[i][i+1]=-1
    if derivatives=="finite":       
        F[0][0]=2
        F[0][1]=-1
        F[99][98]=-1
        F[99][99]=2
    elif derivatives=="dirichlet":
        F[0][0]=1
        F[0][1]=-1
        F[99][98]=-1
        F[99][99]=1
        F=0.5*F+0.5*np.dot(F,F)
    elif derivatives=="neumann":
        F[0][0]=0
        F[0][1]=0
        F[99][98]=0
        F[99][99]=0
        F=0.5*F+0.5*np.dot(F,F)
    else:
        print("derivatives don't exist")
    #put derivative operator and constraints into a new matrix 
    N=np.zeros((103,100))
    x=0
    y=0
    N[x:x+100,y:y+100]=F
    N[100][25]=1
    N[101][50]=1
    N[102][60]=1
    #make b vector with constraints
    b=np.zeros(103)
    b[1]=1
    b[2]=2



    
    b[100]=1
    b[101]=-1
    b[102]=0.3
    #normal equation: N*N.T*x=N.T*b -->  x=np.solve((N*N.T),(N.T*b))
    y=np.linalg.solve(np.dot(N.T,N),np.dot(N.T,b)) 
    return y
"""
Test: Spline curve pass (25,1) (50,-1) (60,0.3)
"""
#derivatives="finite"
#derivatives="dirichlet"
derivatives="neumann"
y=spline(derivatives)
x=np.arange(0,100,1)
pl.plot(25,1,"ro")
pl.plot(50,-1,"ro")
pl.plot(60,0.3,"ro")
pl.plot(x,y,"-k")
pl.title(derivatives+" boundary condition_Kuang")
pl.show()



"""


Spline implemented by sparse matrix
"""
def spline_sparse():
    #make a 100*100 matrix as derivative operator
    M=sp_.diags([-1,2,-1],[-1,0,1],shape=(100,100))
    #make a 3*100 matrix wit constaints   
    row=np.array([0,1,2])
    col=np.array([25,50,60])
    data=np.array([1,1,1])
    nc=sp_.coo_matrix((data,(row,col)),shape=(3,100))
    #put derivative operator and constraints into a new matrix 
    N=sp_.vstack([M,nc])
    #make a vector with constraint
    b=sp_.coo_matrix(([1,-1,0.3],([100,101,102],[0,0,0])))
    N=N.tocsr()
    b=b.tocsr()
    #solve linear system by nurmal equation
    y=spsolve((N.T).dot(N),(N.T).dot(b))
    return y
"""
test:Spline implemented by sparse matrix
"""    
y=spline_sparse()
x=np.arange(0,100,1)
pl.plot(25,1,"ro")
pl.plot(50,-1,"ro")
pl.plot(60,0.3,"ro")
pl.plot(x,y,"-k")
pl.title("spline with sparse matrix_Kuang")
pl.show()    


"""


2D spline 
points [3][2]=1; [5][4]=-1; [8][7]=0.5; [9][3]=-0.5

"""
def spline_2d():
    #make a 100*100 matrix as derivative operator with 2D bodundary
    M1=sp_.eye(100,100,k=-3,format='lil')
    M2=sp_.eye(100,100,k=-1,format='lil')
    M3=sp_.eye(100,100,format='lil')
    M4=sp_.eye(100,100,k=1,format='lil')
    M5=sp_.eye(100,100,k=95,format='lil')
    M = -M1 - M2 + 4*M3 -M4 -M5
    M[0,0]=2
    M[1,1]=3
    M[2,2]=2
    M[2,3]=0
    M[3,2]=0
    M[3,3]=3
    M[96,96]=3
    M[96,97]=0
    M[97,96]=0
    M[97,97]=2
    M[98,98]=3
    M[99,99]=2
    #set the constraints 
    nc=sp_.lil_matrix((4,100))
    nc[0,23]=100
    nc[1,45]=100
    nc[2,78]=100
    nc[3,39]=100
    N=sp_.vstack([M,nc])
    #make b vector with constraints    
    b=np.zeros(104)
    b[100]=1*100
    b[101]=-1*100
    b[102]=0.5*100
    b[103]=-0.5*100
    N.tocsr()
    #solve linear system by nurmal equation
    z_=spsolve((N.T).dot(N),(N.T).dot(b)) 
    #retrive 2D z=(x,y) result
    z=np.zeros((10,10))
    for i in range(0,10):
        for j in range(0,10):
            z[i,j]=z_[i*10+j]
    return z
    
"""
Test: 2d Spline and plot it
"""
z=spline_2d()
x_=np.arange(0,10,1)
y_=np.arange(0,10,1)
x,y = np.meshgrid(x_,y_)
fig=pl.figure()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.plot([3,5,8,9],[2,4,7,3],[1,-1,0.5,-0.5],"o",color="red")
pl.show()



