import numpy as np
import matplotlib.pylab as py
import math

"""
plot the path of the deathstar.
"""
def draw_path(x,y):
    A=np.array([0,-0.1])
    V=np.array([2,1])
    P=np.array([0,0])
    timestep = 0.8
    for i in range(0,60):
        V = V + timestep * A
        P = P + timestep * V
        end,N=intersection(x,y,V,P)
        if end:
            py.plot(P[0],P[1],"bs" )
            last_P = P - timestep * V            
            Q=getprojection(last_P,P,N)
            R=last_P+2*(Q-last_P)
            V=R-P
        else:
            py.plot(P[0],P[1],"ro")
"""
seperate the points to two vector.
"""
def sep_points(rectangle):
    n=rectangle.shape[0]
    x=np.zeros(n)
    y=np.zeros(n)
    for i in range(0,n):
        x[i]=rectangle[i][0]
        y[i]=rectangle[i][1]
    return x,y
"""
pute the points into one matrix.
"""    
def put_points(x,y):
    rectangle=np.zeros((x.shape[0],2))
    for i in range(0,x.shape[0]):
        rectangle[i][0]=x[i]
        rectangle[i][1]=y[i]
    return rectangle
"""
draw the points and connect them by black lines.
"""
def draw_rectangle(x,y):
    py.plot(x,y,linewidth=2,color='black')  
"""
move the rectangle to another location
"""
def move_rectangle(x,y,move):
   x=x+move[0]
   y=y+move[1]
   return x,y
"""
rotate the rectangle by an angle.
"""
def rotate_rectangle(x,y,angle):
    theta=(angle/180.)*np.pi
    euler_angle=np.zeros((2,2))
    euler_angle[0,0]=math.cos(theta)
    euler_angle[0,1]=-(math.sin(theta))
    euler_angle[1,0]=math.sin(theta)
    euler_angle[1,1]=math.cos(theta)
    rectangle=put_points(x,y)
    for i in range(0,rectangle.shape[0]):
        rectangle[i]=np.dot(rectangle[i],euler_angle)
    x,y=sep_points(rectangle)
    return x,y
"""
get the vertexes array and eages array .
"""
def vertex_eage(x,y):
    n=x.shape[0]
    Q=np.zeros((n,2))
    dQ=np.zeros((n-1,2))
    for i in range(0,n):
        Q[i][0]=x[i]
        Q[i][1]=y[i]        
    for i in range(0,n-1):    
        dQ[i]=Q[i+1]-Q[i]
    return Q,dQ

"""
find the intersection point of the path and rectangle.
aV-b*dQ=[[V0,-dQ0],[V1,-dQ1]]*[a,b]=Q-P
"""
def intersection(x,y,V,P):
    end=False
    Q,dQ=vertex_eage(x,y)
    N=[]
    for i in range(0,dQ.shape[0]):
        A=np.zeros((2,2))
        A[0,0]=V[0]
        A[0,1]=-dQ[i][0]
        A[1,0]=V[1]
        A[1,1]=-dQ[i][1]
        b=Q[i]-P
        a=np.linalg.solve(A,b)
        if(0<a[0]<1 and 0<a[1]<1):
            end=True
            N=getnormal(Q,i).ravel()
    return end,N
"""
return the normal vector for side i
"""
def getnormal(deathstar,i):
    deathstar=np.asarray(deathstar)
    deathstar=deathstar.T
    dp=deathstar[:,i+1]-deathstar[:,i]
    normal=np.array([dp[1],-dp[0]])
    normal=normal.reshape(2,1)
    return normal/np.linalg.norm(normal)
    
"""
get the projection point of two vector
projection of vector a,b = a.b.b/|b|^1 , if lenth of N = 1, then |b|=1
"""
def getprojection(P,B,N):
    a=P-B
    b=N
    print(b)
    Q=np.dot(np.dot(a,b),b)+B
    return Q


"""
test the functions
"""
# points of the rectangle
x=np.array([0,0,10,10,0])
y=np.array([0,10,10,0,0])

x,y=rotate_rectangle(x,y,70)
x,y=move_rectangle(x,y,[40,-4.5])
draw_rectangle(x,y)
draw_path(x,y)
py.axes().set_aspect('equal','datalim')
py.title("angle: 70 location of scare: x=40,y=-4.5")
py.show()