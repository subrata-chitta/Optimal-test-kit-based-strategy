

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:26:37 2020

@author: subrata
"""

import numpy as np
from numpy import *
from scipy import stats
from scipy.integrate import odeint
import pylab as pl
import random
import scipy.io
import math
import networkx as nx
from collections import Counter
#node number
N=500
#population of nodes
H=np.zeros([N],float)
for i in range(0,N):
    H[i]=10000.
#load adjacency matrix    
A=np.loadtxt('scale_free_network_500_14.txt',float)
#create graph of adjacency matrix
G=nx.Graph(A)
# degree of the graph
degree = A.sum(axis=1)
row_sums= A.sum(axis=1)
#normalized adjacency matrix
A1=A/row_sums[:,np.newaxis]
#number of nodes randomly chosen 
N1=100
def createRandomSortedList(num,start = 0, end = 499):
    arr=[]
    tmp= random.randint(start,end)
    for x in range(num):
        while tmp in arr:
            tmp= random.randint(start,end)
        arr.append(tmp)
    arr.sort()
    return arr

x1=createRandomSortedList(N1,0,499)
c_c=nx.closeness_centrality(G)
bet_c=nx.betweenness_centrality(G)
clu_c=nx.clustering(G)

bet=[]
clu=[]
for i in range(0,N):
    bet.append(bet_c[i])
    clu.append(clu_c[i])
deg_1=[]
bet_1=[]
clu_1=[]
for i in range(0,N):
    deg_1.append([degree[i],i])
    bet_1.append([bet_c[i],i])
    clu_1.append([clu_c[i],i])
deg_1.sort()
bet_1.sort()
clu_1.sort() 
x2=[]
x3=[]
x4=[]
for i in range(0,N1):
    x2.append(deg_1[-N1:][i][1])
    x3.append(bet_1[-N1:][i][1])
    x4.append(clu_1[-N1:][i][1])
a=np.zeros([N,1],float)
for i in range(0,N1):
    a[x1[i]]=(1./N1)

a03=np.zeros([N,1],float)
for i in range(0,N1):
    a03[x2[i]]=degree[x2[i]]
 
a07=np.zeros([N,1],float)
for i in range(0,N1):
    a07[x4[i]]=clu[x4[i]]

a06=np.zeros([N,1],float)
for i in range(0,N1):
    a06[x3[i]]=bet[x3[i]]    
a3=[x/np.sum(a03) for x in a03]    
a6=[x/np.sum(a06) for x in a06]
a7=[x/np.sum(a07) for x in a07]


v=np.zeros([5,N],float)
for i in range(0,N):
    v[1,i]=a[i][0]
    v[2,i]=a3[i][0]
    v[3,i]=a6[i][0]
    v[4,i]=a7[i][0]



R0=np.arange(0.0,6,.3)
#create different transmission rates 
b=[x*0.01 for x in R0]
     
def f(y,t):
    dy = np.empty(5*N+1,float)
    s = y[0:5*N:5]
    e = y[1:5*N:5]
    i_s = y[2:5*N:5]
    h = y[3:5*N:5]
    r = y[4:5*N:5]
    k = y[2500]
    S1 = np.dot(A1,s)
    S2 = np.dot(A1,e)
    S3 = np.dot(A1,i_s)
    S4 = np.dot(A1,r)
    for i in range(0,N):
        dy[5*i] = ((-B*s[i]*i_s[i])/H[i] )+K*(S1[i]-s[i])
        dy[5*i+1] = ((B*s[i]*i_s[i])/H[i]) -sigma*e[i] + K*(S2[i]- e[i])
        dy[5*i+2] = sigma*e[i] - (c0+(c1*R[i]*k))*i_s[i] + K*(S3[i] - i_s[i])
        dy[5*i+3] = (c0+(c1*R[i]*k))*i_s[i] - gamma*h[i] -delta*h[i]
        dy[5*i+4] = gamma*h[i] + K*(S4[i] - r[i])
    dy[2500] = xi*(np.sum(i_s)) - ki*k
    return dy

os_n=[]  
for k in range(0,len(b)):
    #choosing identicaly distributed kit
    R=v[1]
    print(k)
    B=b[k]
    sigma=0.1
    delta=0.0
    gamma=0.07
    c0=0.01
    c1=0.0001
    tf = 1500
    xi = 0.02
    ki = 0.1
    K=0.05
    y0 = np.zeros(5*N+1,float)
    y0[0:5*N:5]=10000.
    y0[1:5*N+1:5]=0
    y0[2:5*N+2:5]=0
    y0[3:5*N+3:5]=0  
    y0[4:5*N+4:5]=0
    y0[0]=y0[20]=10000-20
    y0[1]=y0[21]=13
    y0[2]=y0[22]=7
    y0[3]=0.
    y0[4]=0.
    y0[2500]=10
    t1= np.linspace(0, tf, tf*100)
    [sol1,info] = odeint(f,y0,t1,full_output=1,printmessg=1)

    i_3=[]
    for i in range(0,tf*100):
        i_3.append(np.sum(sol1[i,4:5*N:5])/5000000.)
    os_n.append(i_3[-1])    
l=['SI']
markers=["b*"]
l2=['b']
ind3=(0,0.02,0.04,0.06)
ind4=(0,0.2,0.4,0.6,0.8,1,1.02) 
pl.figure(1,figsize=(6,6))
pl.plot(b,os_n,l2[0],linewidth=5,label=l[0])
pl.xlabel(r'$\beta$',fontsize=35)
pl.ylabel(r'$Z_{FOS}$',fontsize=35)
pl.yticks(ind4,('0','0.2','0.4','0.6','0.8','1',''),fontsize=25)
pl.xticks(ind3,('0','0.02','0.04','0.06'),fontsize=25)
pl.rcParams['axes.linewidth']=3
pl.tick_params(direction= 'in', length=7,width=3)
pl.legend(loc='lower right',fontsize=20)
pl.tight_layout()  

