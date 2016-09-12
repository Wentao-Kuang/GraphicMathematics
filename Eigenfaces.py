# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:43:22 2015

@author: Kuang
"""
import numpy as np
import matplotlib.pyplot as pl
import pylab as pyl
import math
from PIL import Image

"""
function to convert an image into a column vector
"""
def img2vec(im):
    (x,y)=im.size
    im = im.resize((x,y), Image.BICUBIC)
    v = np.reshape(im, (x*y, 1))
    v = np.double(v) / 255.
    v = v.reshape(x*y,)
    return v
    
"""
functon to scales the pixels back into the 0..255 range 
then call function vec2img to convert to image.
"""
def vec2im2(e,xres,yres):
    scale = 1. / (np.max(e) - np.min(e))
    e = e - np.min(e)
    e = e * scale
    imrecovered = vec2img(e,xres,yres)
    return imrecovered
 
"""
function to convert a column vector back into an image.
"""   
def vec2img(v,xres,yres):
    v = v.reshape((yres,xres))
    v = np.fix(255.*v)
    v8 = np.zeros((yres,xres), np.uint8) 
    v8[:,:] = v[:,:] 
    imrecovered = Image.fromarray(v8)
    return imrecovered    

"""
function to import the images and convert them into vectors then put the vectors in a matrix.
"""
def load_pictures(xres,yres,picture_numbers,PROJECT_PATH):   
    npixels=xres*yres
    IMS=np.zeros((picture_numbers,npixels))
    for i in range(0,picture_numbers):
        picture_name=str(i+1)+'.png'
        path=PROJECT_PATH+picture_name
        im = Image.open(path)
        im=im.convert('L') #convert images to a L mode.
        v=img2vec(im) 
        IMS[i]=v    
    IMS=IMS.T
    return IMS

"""
function to get the mean
then remove the mean of this matrix
finally get the eigenvalues and eigenvectors.
"""
def egienfaces(IMS,picture_numbers):    
    m=np.mean(IMS,axis=1)
    for i in range(0,IMS.shape[1]):
        IMS[:,i]-=m
    U,S,Vt=np.linalg.svd(IMS)
    x=np.arange(0,picture_numbers,1)
    pl.plot(x,S/math.sqrt(15),"-k")
    pl.show()
    return U
    
"""
function to convert egienvectors to images and save them by calling vectim2()
"""
def save_pictures(PROJECT_PATH,IMS,U,xres,yres):
    for i in range(0,IMS.shape[1]):
        im=vec2im2(U[:,i],xres,yres)
        img_name=str(i)+".png"
        im.save(PROJECT_PATH+"results/"+img_name)
        
"""
function to generate the Fourier approximation to a paticular face
"""
def Fourierfaces(m,U,PROJECT_PATH):
    path=PROJECT_PATH+"15.png"
    im=Image.open(path)
    f=img2vec(im)
    p=np.zeros(f.shape)
    for i in range(0,15):
        p+=np.dot(np.dot((f-m).T,U[:,i]),U[:,i])
        vec2im2(p+m,59,65).save(PROJECT_PATH+"results2/"+str(i)+".png")


"""
test eginfaces of assigments 2
"""
PROJECT_PATH = 'C:\\Users\\Kuang\\Desktop\\comp471\\assignment2\\faces1\\' 
picture_numbers=15
xres=59
yres=65
IMS=load_pictures(xres,yres,picture_numbers,PROJECT_PATH)
U=egienfaces(IMS,picture_numbers)
save_pictures(PROJECT_PATH,IMS,U,xres,yres)

"""
test eginfaces of Obama
"""
PROJECT_PATH = 'C:\\Users\\Kuang\\Desktop\\comp471\\assignment2\\faces2\\'
picture_numbers=10
xres=60
yres=60
IMS=load_pictures(xres,yres,picture_numbers,PROJECT_PATH)
U=egienfaces(IMS,picture_numbers)
save_pictures(PROJECT_PATH,IMS,U,xres,yres)

"""
test the Fourier approximation faces.
"""
PROJECT_PATH = 'C:\\Users\\Kuang\\Desktop\\comp471\\assignment2\\faces1\\'
xres=59
yres=65
picture_numbers=15
IMS=load_pictures(xres,yres,picture_numbers,PROJECT_PATH)
m=np.mean(IMS,axis=1)
U=egienfaces(IMS,picture_numbers)
Fourierfaces(m,U,PROJECT_PATH)
