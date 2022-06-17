#!/usr/bin/python
#
# By Xiaotong Li, Karel Adamek, and Wesley Armour
#
from __future__ import division
import os
import numpy
import cupy
import scipy.io
from mat4py import loadmat
import time
from cuRandomSVD import *

"""
@INPUT:
    the first element of each mask, mask matrix, incomplete matrix
    
@OUTPUT:
    complete matrix, reconstruction error
"""

def gpuSVT(A,mask,tau=None,delta=None,epsilon=1e-2,rel_improvement=-0.01,max_iterations=5000):
    
    Y = cupy.zeros_like(A)
    recon_error = cupy.zeros(max_iterations)

    if not tau:
        tau = 5 * cupy.sum(cupy.asarray(A.shape)) / 2
    if not delta:
        delta = 1.2 * cupy.sum(mask) / cupy.prod(cupy.asarray(A.shape))

    r_previous = 0

    for k in range(max_iterations):
        if k == 0:
            X = cupy.zeros_like(A)
        else:
            sk = r_previous + 1
            
            if sk < min(len(Y),len(Y[0]))/2-9:
                p = min(sk, Y.shape[1]-1)+20
            else:
                p = min(len(Y),len(Y[0])) - sk
            S = cupy.zeros(int(min(sk, Y.shape[1]-1)))
            U = cupy.zeros((int(A.shape[0]),int(min(sk, Y.shape[1]-1))))
            V = cupy.zeros((int(min(sk, Y.shape[1]-1)),int(A.shape[1])))
            cS, cU, cV = cuRandomSVD(Y, min(sk, Y.shape[1]-1), p, nIterations=1)
            for i in range(int(min(sk, Y.shape[1]-1))):
                S[i] = cS[i]
                for j in range(Y.shape[0]):
                    U[j][i] = cU[j][i]
                for h in range(Y.shape[1]):
                    V[i][h] = cV[h][i]
            
            while cupy.min(S) >= tau:
                sk = sk + 5
                
                if sk < min(len(Y),len(Y[0]))/2-9:
                    p = min(sk, Y.shape[1]-1)+20
                else:
                    p = min(len(Y),len(Y[0])) - sk
                
                S = cupy.zeros(int(min(sk, Y.shape[1]-1)))
                U = cupy.zeros((int(A.shape[0]),int(min(sk, Y.shape[1]-1))))
                V = cupy.zeros((int(min(sk, Y.shape[1]-1)),int(A.shape[1])))
                cS, cU, cV = cuRandomSVD(Y, min(sk, Y.shape[1]-1), p, nIterations=1)
                for i in range(int(min(sk, Y.shape[1]-1))):
                    S[i] = cS[i]
                    for j in range(Y.shape[0]):
                        U[j][i] = cU[j][i]
                    for h in range(Y.shape[1]):
                        V[i][h] = cV[h][i]
                
            shrink_S = cupy.maximum(S - tau, 0)
            r_previous = cupy.count_nonzero(shrink_S)
            diag_shrink_S = cupy.diag(shrink_S)
            X = cupy.dot(cupy.dot(U, diag_shrink_S), V)
        Y += delta * mask * (A - X)

        recon_error[k] = cupy.linalg.norm(mask * (X - A)) / cupy.linalg.norm(mask * A)
        if recon_error[k] < epsilon:
            print('recon_error small enough with k',k)
            break

    return X, recon_error

if __name__ == "__main__":
          
    Reference_Number = '_gpuSVT'
    
    Ini = scipy.io.loadmat('ini.mat')
    pyini = Ini['ini']
    pyini = cupy.array(pyini)
    
    Ve = scipy.io.loadmat('pymask.mat')
    pymask = Ve['pymask']
    pymask = cupy.array(pymask)
    for i in range(2):
        pymask[i][0][0] = pyini[0][i]
    
    Le = scipy.io.loadmat('pyinp.mat')
    pyR = Le['pyinp']
    pyR = cupy.array(pyR)
    
    
    for i in range(2):
        print('Start SVT MC ...')
        
        R = pyR[i]
        mask = pymask[i]
	    
        tic = time.perf_counter()
        nnR, recon_error = gpuSVT(R, mask)
        toc = time.perf_counter()
        print(f"SVT in {toc - tic:0.4f} seconds")
        
        scipy.io.savemat('nnR'+Reference_Number+str(i)+'.mat',mdict={'nnR':nnR.get()})
        scipy.io.savemat('recon_error'+Reference_Number+str(i)+'.mat',mdict={'recon_error':recon_error.get()})
    
        print("SVT MC finished.")
    
