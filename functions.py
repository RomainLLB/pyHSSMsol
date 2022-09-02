# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:04:04 2019

@author: rlecuyer
"""
import numpy as np
from scipy.linalg import hankel, expm
from tqdm import tqdm
from numpy.linalg import lstsq,norm
from scipy.integrate import simps 
from scipy import signal

   
    
def radiation(num_bodies, t, w, B, threshold_value=10):
    n=B.shape[0]
    K=np.zeros([n,n,t.size])
    with tqdm(total=n*n) as pbar:
        for i in range(n):
            for j in range(n):
                pbar.set_description("IRF calculation of K[{:2d},{:2d}]".format(i+1,j+1))
                Bij=B[i,j,:]
                
                if np.max(np.abs(Bij)) > threshold_value:
                
                    for k in range(t.size):
                        K[i,j,k]=2/np.pi*simps(y=Bij*np.cos(w*t[k]),x=w)
                        
                else:
                    pass
                    
                pbar.update(1)
    K[:,:,0] *= 0.5
    return(K)
    
    
def state_space(t, K, order=10, threshold_value=10):
    '''Function to calculate the state space reailization of the wave
    radiation IRF.

    Args:
        order : int
            Maximum order of the state space reailization fit
        r2_thresh : float
            The R^2 threshold used for the fit. THe value must be between 0
            and 1

    Returns:
        No variables are directily returned by thi function. The following
        internal variables are calculated:

        Ass :
            time-invariant state matrix
        Bss :
            time-invariant input matrix
        Css :
            time-invariant output matrix
        Dss :
            time-invariant feedthrough matrix
        k_ss_est :
            Impusle response function as cacluated from state space
            approximation
        status :
            status of the realization, 0 - zero hydrodynamic oefficients, 1
            - state space realization meets R2 thresholdm 2 - state space
            realization does not meet R2 threshold and at ss_max limit

    Examples:
    '''
    n1 = K.shape[0]
    n2 = K.shape[1]
    
    dt = t[2] - t[1]
    r2bt = np.zeros([n1, n2, t.size])
    k_ss_est = np.zeros(t.size)
    A = np.zeros([n1, n2, order, order])
    B = np.zeros([n1, n2, order, 1])
    C = np.zeros([n1, n2, 1, order])
    D = np.zeros([n1, n2])
    irk_bss = np.zeros([n1, n2, t.size])
    it = np.zeros([n1, n2])
    r2t = np.zeros([n1, n2])
    
    maxval = n1 * n2
    with tqdm(total=maxval) as pbar:
        
        for i in range(n1):
            for j in range(n2):
                pbar.set_description('State space calculation for ' + 'K[{:2d},{:2d}]'.format(i+1,j+1) )    

                r2bt = np.linalg.norm(K[i, j, :] - K.mean(axis=2)[i, j])
                
                ss = order 
                
                
                if r2bt != 0.0 and np.max(np.abs(K[i, j, :])) > threshold_value: #skipping zeros and noises
                    
                        
                    # Perform Hankel Singular Value Decomposition
                    y = dt * K[i, j, :] 
                    h = hankel(y[1::])
                    u, svh, v = np.linalg.svd(h)
                    
                    u1 = u[0:t.size - 2, 0:ss]
                    v1 = v.T[0:t.size - 2, 0:ss]
                    u2 = u[1:t.size - 1, 0:ss]
                    sqs = np.sqrt(svh[0:ss].reshape(ss, 1))
                    invss = 1 / sqs
                    ubar = np.dot(u1.T, u2)
                    
                    # Discrete modele
                    a = ubar * np.dot(invss, sqs.T)
                    b = v1[0, :].reshape(ss, 1) * sqs
                    c = u1[0, :].reshape(1, ss) * sqs.T
                    d = y[0]
                    
                    # # Continuous modele from discrete model
                    # # (T/2*I + T/2*A)^{-1}         = 2/T(I + A)^{-1}
                    # iidd = np.linalg.inv(dt/2 * np.eye(ss) + dt/2 * a)
                    # # (A-I)2/T(I + A)^{-1}         = 2/T(A-I)(I + A)^{-1}
                    # ac = np.dot(a - np.eye(ss), iidd)
                    # # (T/2+T/2)*2/T(I + A)^{-1}B   = 2(I + A)^{-1}B
                    # bc = (dt/2  + dt/2 ) * np.dot(iidd, b)
                    # # C * 2/T(I + A)^{-1}          = 2/T(I + A)^{-1}
                    # cc = np.dot(c, iidd)
                    # # D - T/2C (2/T(I + A)^{-1})B  = D - C(I + A)^{-1})B
                    # dc = d + dt/2 * np.dot(np.dot(c, iidd), b)
                    
                    alpha = 1
                    beta = 2/dt
                    
                    # Continuous modele from discrete model
                    ac = beta * np.dot(np.linalg.inv(alpha*np.eye(ss) + a),(a - alpha*np.eye(ss)))
                    bc = np.sqrt(2*alpha*beta) * np.dot(np.linalg.inv(alpha*np.eye(ss) + a), b)
                    cc = np.sqrt(2*alpha*beta) * np.dot(c, np.linalg.inv(alpha*np.eye(ss) + a))
                    dc = d - np.dot(np.dot(c, np.linalg.inv(alpha*np.eye(ss) + a)), b)
                    
                    
                    for jj in range(t.size): 
                    
                        # Calculate impulse response function from state space approximation
                        # k_ss_est[jj] = np.dot(np.dot(cc, expm(ac * dt * jj)), bc)
                        k_ss_est[jj] = np.dot(np.dot(cc, expm(ac * t[jj])), bc)

                                            
                    # Calculate 2 norm of the difference between know and estimated
                    # values impulse response function
                    R2TT = np.linalg.norm(K[i, j, :] - k_ss_est)
                    # Calculate the R2 value for impulse response function
                    R2T = 1 - np.square(R2TT / r2bt)
                
                    A[i, j, :, :] = ac[:,:]
                    B[i, j, :, 0] = bc[:, 0]
                    C[i, j, 0, :] = cc[0, :]
                    D[i, j] = dc
                    irk_bss[i, j, :] = k_ss_est
                    r2t[i, j] = R2T*100.0
                    it[i, j] = ss
                    
                else:
                    
                    A[i, j, :, :] = 0.0
                    B[i, j, :, 0] = 0.0
                    C[i, j, 0, :] = 0.0
                    D[i, j] = 0.0
                    irk_bss[i, j, :] = np.zeros([t.size])
                    r2t[i, j] = 100
                    it[i, j] = 0.0 

                
                pbar.update(1)
                
    B = np.reshape(B,(n1,n2,order))
    C = np.reshape(C,(n1,n2,order))
                
    return(A,B,C,D,irk_bss,r2t)
    



def prony(x,y,nb_exp=None,nnb_exp=None):
    """
    Curve fitting using Prony's series'
    """
    deltax=np.mean(np.diff(x))#t[1]-t[0]
    N=len(x)-1 # number of sample
    
    # number of exponential
    if nb_exp==None:
        n=int(np.floor(N/2))  
    else:
        n=nb_exp   
    
    if np.all(y==0.0) :
        
        alpha = np.zeros([n],dtype=np.complex)
        beta = np.zeros([n],dtype=np.complex)
        yprony = np.zeros([x.size])
        corres = 1.0
        
        return(alpha,beta,yprony,corres)
        
    
    # Step 1
    Y=np.zeros((N-n+1))
    Ymat=np.zeros((N-n+1,n))
    for i in range(N-n+1):
        Y[i]=y[n+i]
        for j in range(n):
            Ymat[i,j]=y[n+(i-1)-j]  
    asol=lstsq(Ymat,Y,rcond=None)[0]       #solving AX=B   
    
    
    # Step 2
    p=np.concatenate((np.array([1]),-asol),axis=0)
    p=np.transpose(p).tolist()       
    z=np.roots(p)
    bbbeta=np.log(z)/deltax 
    
    # Step 3
    Y2=np.zeros((N))
    Zmat=np.zeros((N,n),dtype=np.complex)
    for i in range(N):
        Y2[i]=y[i]
        for j in range(n):
            Zmat[i,j]=z[j]**(i)
    aaalpha=lstsq(Zmat,Y2,rcond=None)[0]    #solving AX=B       
    
    
    ###########################################################################
    ############# Reduction of the number of exponentials #####################
    ###########################################################################
    
    
    if nnb_exp==None: 
        alpha=np.reshape(aaalpha,n)
        beta=np.reshape(bbbeta,n)
    
    else:       
        aalpha=np.reshape(aaalpha,n)
        bbeta=np.reshape(bbbeta,n)        
        alpha=np.zeros((nnb_exp),dtype=np.complex)
        beta=np.zeros((nnb_exp),dtype=np.complex)
        critere=np.abs(np.real(aalpha))
        indices=np.reshape(np.argsort(critere,axis=0),n).tolist()
        indices.reverse()
        for i in range(nnb_exp):
            indice=indices[i]
            alpha[i]=aalpha[indice]
            beta[i]=bbeta[indice]  
            
    # Interpolation 
    yprony=np.zeros((N+1))
    yp=0
    for i in range(len(alpha)):
        yp=alpha[i]*np.exp(beta[i]*x)
        yprony += np.real(yp)
        
    # Matching rate calculation
    r2bt = norm(y - y.mean())
    R2TT = norm(y - yprony)
    corres = 1 - np.square(R2TT / r2bt)
    corres=max(0,corres)
        
    
    return(alpha,beta,yprony,corres)