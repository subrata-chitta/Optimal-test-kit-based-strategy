# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:53:14 2020

@author: subrata
"""
import numpy as np
from numpy import *
from scipy import stats
from scipy.integrate import odeint
import pylab as pl
import random
import math
import networkx as nx
#from collections import Counter


N=500
H=np.zeros([N],float)
for i in range(0,N):
    H[i]=10000.
def createRandomSortedList(num,start = 0, end = 499):
    arr=[]
    tmp= random.randint(start,end)
    for x in range(num):
        while tmp in arr:
            tmp= random.randint(start,end)
        arr.append(tmp)
    arr.sort()
    return arr    
f1=[2,3,4,5,6,7]
inf_peak=[]
out_size=[]  
for q in range(0,len(f1)):
    G = nx.barabasi_albert_graph(N,f1[q])
    A = nx.to_numpy_matrix(G)
    A = np.array(A)    
    degree = A.sum(axis=1)
    print(np.mean(degree))
    row_sums= A.sum(axis=1)
    A1=A/row_sums[:,np.newaxis]
    N1=50
    x1=createRandomSortedList(N1,0,499)
        
    e_v_c=nx.eigenvector_centrality(G)
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

    B=[]
    for i in range(0,N):
        B.append(0.03)

    v=np.zeros([5,N],float)
    for i in range(0,N):
        v[1,i]=a[i][0]
        v[2,i]=a3[i][0]
        v[3,i]=a6[i][0]
        v[4,i]=a7[i][0]

#
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
            dy[5*i] = ((-B[i]*s[i]*i_s[i])/H[i] )+K*(S1[i]-s[i])
            dy[5*i+1] = ((B[i]*s[i]*i_s[i])/H[i]) -sigma*e[i] + K*(S2[i]- e[i])
            dy[5*i+2] = sigma*e[i] - (c0+(c1*R[i]*k))*i_s[i] + K*(S3[i] - i_s[i])
            dy[5*i+3] = (c0+(c1*R[i]*k))*i_s[i] - gamma*h[i] -delta*h[i]
            dy[5*i+4] = gamma*h[i] + K*(S4[i] - r[i])
        dy[2500] = xi*(np.sum(i_s)) - ki*k
        return dy
    i_peak=[] 
    z_fos=[]
    for k in range(0,5):
        print(k)
        R=v[k]
        sigma=0.1
        delta=0.0
        gamma=0.07
        c0=0.01
        c1=0.0001
        tf = 3000
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
        z_fos.append(i_3[-1]) 
        out_size.append(i_peak)
        
  
        i_4=[]
        for i in range(0,tf*100):
            i_4.append(np.sum(sol1[i,2:5*N:5])/5000000.)
        i_peak.append(np.max(i_4)) 
        inf_peak.append(i_peak)
m1=inf_peak[0:30:5]        
n1=out_size[0:30:5]
m=np.zeros([len(f1),5],float)
for  i in range(0,len(f1)):
    for j in range(0,5):
        m[i,j]=m1[i][j]
n=np.zeros([len(f1),5],float)
for  i in range(0,len(f1)):
    for j in range(0,5):
        n[i,j]=n1[i][j]        
ma_on=[4,6,8,10,12,14]
markers=["ko-","b*-","gs-","rp-","D-"]
marker=["--"]
k=[4,6,8,10,12,14]
ind3=(0.6,0.8,1,1.2)
pl.figure(1,figsize=(6,6))
pl.plot(k,m[:,0],'k',linewidth=3)
pl.plot(ma_on,m[:,0],markers[0],label='SW',markersize=12,markeredgecolor='k')
pl.plot(k,m[:,1],'b',linewidth=3)
pl.plot(ma_on,m[:,1],markers[1],label='SI',markersize=16,markeredgecolor='b')
pl.plot(k,m[:,2],'g',linewidth=3)
pl.plot(ma_on,m[:,2],markers[2],label='SD',markersize=10,markeredgecolor='g')
pl.plot(k,m[:,3],'r',linewidth=3)
pl.plot(ma_on,m[:,3],markers[3],label='SB',markersize=14,markeredgecolor='r')
pl.plot(k,m[:,4],'-',color='orange',linewidth=3)
pl.plot(ma_on,m[:,4],markers[4],label='SC',color='#FFA500',markersize=10,markeredgecolor='orange')
ind1=(4,8,12,16)
ind2=(0,0.1,0.2,0.3)
pl.xlabel(r'$\langle d \rangle$',fontsize=35)
pl.ylabel(r'$I_{\rm max}$',fontsize=35)
pl.xticks(ind1,('4','8','12','16'),fontsize=30)
pl.yticks(ind2,('0','0.1','0.2','0.3'),fontsize=30)
pl.rcParams['axes.linewidth']=2
pl.tick_params(direction= 'in', length=7,width=3)
#pl.legend(bbox_to_anchor=(0.0,1.01),loc=7,ncol=5,borderaxespad=0, frameon=False,fontsize=16)
pl.tight_layout()

pl.figure(2,figsize=(6,6))
pl.plot(k,n[:,0],'k',linewidth=3)
pl.plot(ma_on,n[:,0],markers[0],label='SW',markersize=12,markeredgecolor='k')
pl.plot(k,n[:,1],'b',linewidth=3)
pl.plot(ma_on,n[:,1],markers[1],label='SI',markersize=16,markeredgecolor='b')
pl.plot(k,n[:,2],'g',linewidth=3)
pl.plot(ma_on,n[:,2],markers[2],label='SD',markersize=10,markeredgecolor='g')
pl.plot(k,n[:,3],'r',linewidth=3)
pl.plot(ma_on,n[:,3],markers[3],label='SB',markersize=14,markeredgecolor='r')
pl.plot(k,n[:,4],'-',color='orange',linewidth=3)
pl.plot(ma_on,n[:,4],markers[4],label='SC',color='#FFA500',markersize=10,markeredgecolor='orange')
pl.xlabel(r'$\langle d \rangle$',fontsize=35)
pl.ylabel(r'$Z_{\rm FOS}$',fontsize=35)
pl.xticks(ind1,('4','8','12','16'),fontsize=30)
pl.yticks(ind3,('0.6','0.8','1',''),fontsize=30)
pl.rcParams['axes.linewidth']=2
pl.tick_params(direction= 'in', length=7,width=3)
#pl.legend(frameon=False,loc=7,fontsize=16)
pl.tight_layout()