# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:39:42 2019

@author: rlecuyer
"""

import os
import h5py
import numpy as np    
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from functions import radiation, prony
from scipy.integrate import simps
from matplotlib.backends.backend_pdf import PdfPages


wdir=os.path.abspath('')


dim=3 # 3 for 3D and 2 for 2D
threshold = 10 # threshold value under which the signal is a numerical noise
w_lim = 100.0 # max pulsation for the extrapolation

###############################################################################
##################### Read HydroStar Data #####################################
###############################################################################

num_bodies=1
path = 'HydroStar_Data'
tab = np.loadtxt(wdir + '\\' + path + '\\Cylindre_CM_11.dat')
n_direction = int((tab.shape[1]-1)/2)
wave_dir = np.linspace(0,180,n_direction)
n_w = tab.shape[0]
w = tab[:,0]
dw=w[1]-w[0]
tabf = np.loadtxt(wdir + '\\' + path + '\\Cylindre_F1_1.dat')
n_wf = tabf.shape[0]
Aw=np.zeros([6*num_bodies,6*num_bodies,n_w])
Bw=np.zeros([6*num_bodies,6*num_bodies,n_w])
ex_mag=np.zeros([6*num_bodies,n_direction,n_wf])
ex_phase=np.zeros([6*num_bodies,n_direction,n_wf])
with tqdm(total=6*num_bodies*6*num_bodies, desc = "Reading HYDROSTAR's data") as pbar:
    for ii in range(6*num_bodies):
        path_f = wdir + '\\' + path + '\\Cylindre_F1_{}.dat'.format(ii+1)
        tab_f = np.loadtxt(path_f)
        ex_mag[ii,:,:] = np.transpose(tab_f[:,1:n_direction+1])
        ex_phase[ii,:,:] = np.transpose(tab_f[:,n_direction+1:]*np.pi/180)
        for jj in range(6*num_bodies):    
            path_m = wdir + '\\' + path + '\\Cylindre_CM_{}{}.dat'.format(ii+1,jj+1)
            tab_m = np.loadtxt(path_m)
            Aw[ii,jj,:] = tab_m[:,1]
            path_d = wdir + '\\' + path + '\\Cylindre_CA_{}{}.dat'.format(ii+1,jj+1)
            tab_d = np.loadtxt(path_d)
            Bw[ii,jj,:] = tab_d[:,1]
            pbar.update(1)
ex_re = ex_mag*np.cos(ex_phase)
ex_im = ex_mag*np.sin(ex_phase)

###############################################################################
# Delete the irregular frequencies
###############################################################################

Ainf = Aw[:,:,-1]
indice = 300
w = np.delete(w,np.arange(indice,n_w))
Aw = np.delete(Aw,np.arange(indice,n_w),axis=2)
Bw = np.delete(Bw,np.arange(indice,n_w),axis=2)
ex_mag = np.delete(ex_mag,np.arange(indice,n_wf),axis=2)
ex_phase = np.delete(ex_phase,np.arange(indice,n_wf),axis=2)
ex_re = np.delete(ex_re,np.arange(indice,n_wf),axis=2)
ex_im = np.delete(ex_im,np.arange(indice,n_wf),axis=2)

           
###############################################################################
# Extrapolation of the radiation damping
###############################################################################

def loi_puissance(w,Bw,wlim = 10.0):
    dw = w[1]-w[0]
    wtot = np.arange(w[0],wlim+dw,dw)
    wfit=np.arange(w[-1],wlim,dw)+dw
    indice=w.size-np.where(w==2)[0][0]
    B=np.zeros([6*num_bodies,6*num_bodies,wtot.size])
    for i in range(6*num_bodies):
        for j in range(6*num_bodies):            
            bij=Bw[i,j,:]
            if np.all(bij!=0.0) and np.max(np.abs(bij)) > threshold:
                if np.abs(bij[-1]) < threshold: # a déjà atteint la convergence
                    B[i,j,:] = np.concatenate((bij,np.zeros([wfit.size])),axis=0) 
                elif np.mean(bij) > 0.0:
                    xij = np.log(w[-indice:])                    
                    yij=np.log(bij[-indice:])
                    fit = np.polyfit(xij,yij,1)
                    a = fit[0]
                    b = fit[1]
                    signe = 1
                    bijfit=signe*np.exp(b)*wfit**a
                    B[i,j,:] = np.concatenate((bij,bijfit),axis=0)
                else:
                    xij = np.log(w[-indice:])                    
                    yij=np.log(-bij[-indice:])
                    fit = np.polyfit(xij,yij,1)
                    a = fit[0]
                    b = fit[1]
                    signe = -1
                    bijfit=signe*np.exp(b)*wfit**a
                    B[i,j,:] = np.concatenate((bij,bijfit),axis=0)
            elif np.max(np.abs(bij)) < threshold:
                B[i,j,:] = np.zeros([wtot.size])
            else:
                B[i,j,:] = np.concatenate((bij,np.zeros([wfit.size])),axis=0)
    return(wtot,B)
    
[wtot,B] = loi_puissance(w,Bw,wlim = w_lim)

# IRF computation
t=np.linspace(0.,100.,1001)
Kw=radiation(num_bodies, t, w, Bw);Kw[:,:,0]*=0.5
K=radiation(num_bodies, t, wtot, B);K[:,:,0]*=0.5

# Added mass computation
Minf=np.zeros([6*num_bodies,6*num_bodies])
A=np.zeros([6*num_bodies,6*num_bodies,wtot.size])
with tqdm(total=6*num_bodies*6*num_bodies, desc='Added mass extrapolation') as pbar:
    for i in range(6*num_bodies):
        for j in range(6*num_bodies):
            mij=Aw[i,j,:]
            if np.max(np.abs(mij)) < 1.0:
                pass
            else:
                kij=K[i,j,:]
                mijinf=np.zeros([w.size])
                
                (alpha,beta,yprony,corres) = prony(t,kij,nb_exp=50,nnb_exp=20)
                integrale=np.zeros([w.size])
                                    
                for l in range(alpha.size):
                    yp = alpha[l]/(beta[l]**2+wtot**2)
                    A[i,j,:] -= np.real(yp)   # A-Ainf
                    
                Minf[i,j] = np.mean(mij - A[i,j,:w.size])
                A[i,j,:] += Minf[i,j] # A-Ainf+Ainf
                
            pbar.update(1)
                
A[i,j,:] += Minf[i,j]                



###############################################################################
# Write the h5 file
###############################################################################

nom='simulation_input.h5'

    
data_for_saving={'A':A,'B':B,'Aw':Aw,'Bw':Bw,\
                 'omega':w,'wave_dir':wave_dir,\
                 'num_bodies':num_bodies,'Minf': Minf,\
                 'ex_mag':ex_mag,'ex_phase':ex_phase,\
                 'dimension':dim,'wtot':wtot}


path = wdir+'\\'+nom

if os.path.exists(path):
    os.remove(path)
    
with h5py.File(path, 'a') as mon_fichier:
    
    for key in data_for_saving.keys():
        arr=data_for_saving[key]
        data_dir=key
        mon_fichier.create_dataset(data_dir,data=arr)

        
    mon_fichier.flush()




plot_size = 10

###############################################################################
# Added mass plot
###############################################################################
    


with PdfPages('added_mass.pdf') as pdf:
    
    for ii in range(Aw.shape[0]):
        for jj in range(Aw.shape[1]):
            
            if np.max(np.abs(Aw[ii,jj,:])) > threshold:
    
                fig, ax = plt.subplots()
                plt.subplots_adjust(bottom=0.2)
                
                y1=Aw[ii,jj,:]
                y2=A[ii,jj,:]
                y3=Minf[ii,jj]*np.ones([wtot.size])
                y4=Ainf[ii,jj]*np.ones([wtot.size])
                
                l1, = plt.plot(w, y1, lw=2)
                l2, = plt.plot(wtot , y2, lw=2 ,ls='--')
                l3, = plt.plot(wtot , y3, lw=2, ls='-')
                l4, = plt.plot(wtot , y4, lw=2, ls='--')
                
                plt.legend(['Added mass \n without \n extrapolation',
                            'Added mass \n with \n extrapolation',
                            '$\overline{A} (\omega) \quad = \quad A_{\infty}$',
                            '$Hydrostar \quad A_{\infty}$']
                            ,fontsize=plot_size,framealpha=1)#loc="lower left")
                           
                plt.minorticks_on()
                plt.grid(which="major",ls="-")
                plt.grid(which="minor",ls=":")
                plt.xlabel(r'$\omega \quad (rad.s^{-1})$',fontsize=plot_size)
                plt.ylabel(r'$Added \quad mass \quad (kg)$',fontsize=plot_size)
                ax.set_title(r'$A_{'+str(ii+1)+','+str(jj+1)+'}$',fontsize=plot_size)
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
                
                plt.xscale('log')
                plt.xlim(wtot[0],wtot[-1])
                
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(plot_size)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(plot_size)
                
                plt.pause(0.1)
                plt.tight_layout()
                
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
            else:
                pass
            

###############################################################################
# Radiation damping plot
###############################################################################

with PdfPages('radiation_damping.pdf') as pdf:
    
    for ii in range(Bw.shape[0]):
        for jj in range(Bw.shape[1]):

            if np.max(np.abs(Bw[ii,jj,:])) > threshold:
                
                fig2, ax2 = plt.subplots()
                plt.subplots_adjust(bottom=0.2)
                
                y4=Bw[ii,jj,:]
                y5=B[ii,jj,:]
                l4, = plt.plot(w, y4,lw=2)
                l5, = plt.plot(wtot, y5,lw=2, ls=':')
                plt.legend(['Radiation damping \n without extrapolation','Radiation damping \n with extrapolation']
                           ,fontsize=plot_size,framealpha=1)
                plt.minorticks_on()
                plt.grid(which="major",ls="-")
                plt.grid(which="minor",ls=":")
                plt.xlabel('$\omega \quad (rad.s^{-1})$',fontsize=plot_size)
                plt.ylabel('$Damping \quad coefficient \quad (N.m^{-1}.s)$',fontsize=plot_size)
                ax2.set_title('$B_{'+str(ii+1)+','+str(jj+1)+'}$',fontsize=plot_size)
                ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
                plt.xscale('log')
                plt.xlim(wtot[0],wtot[-1])

                
                for tick in ax2.xaxis.get_major_ticks():
                    tick.label.set_fontsize(plot_size)
                for tick in ax2.yaxis.get_major_ticks():
                    tick.label.set_fontsize(plot_size)
                
                plt.pause(0.1)
                plt.tight_layout()
    
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
            else:
                pass
