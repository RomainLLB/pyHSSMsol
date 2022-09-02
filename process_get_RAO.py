# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:56:32 2019

@author: rlecuyer
"""

import numpy as np
import pyHSSMsol
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.integrate import simps

wdir=os.path.abspath('')
name_input_file='simulation_input.h5'
name_output_file='simulation_output.h5'

###############################################################################
# Read the h5 file
###############################################################################
with h5py.File(wdir+'\\'+name_input_file, 'r') as mon_fichier:
    omega=mon_fichier['omega'][()]
    ex_phase=mon_fichier['ex_phase'][()]
    ex_mag=mon_fichier['ex_mag'][()]
    Aw=mon_fichier['Aw'][()]
    Bw=mon_fichier['Bw'][()]
    Arad=mon_fichier['Arad'][()]
    Brad=mon_fichier['Brad'][()]
    Crad=mon_fichier['Crad'][()]
    Drad=mon_fichier['Drad'][()]
    Minf=mon_fichier['Minf'][()]
    wave_dir=mon_fichier['wave_dir'][()]
    num_bodies=mon_fichier['num_bodies'][()]
    NN=mon_fichier['order'][()] # ordre du modèle d'état
    dim = mon_fichier['dimension'][()] # dimension 2 pour 2D et 3 pour 3D

###############################################################################
# Simulation's parameters
###############################################################################


nbs = 1 # number of solids    
n = 3*(dim-1)*num_bodies*nbs # number of DoF
n_hydro = 3*(dim-1)*num_bodies # number of DoF concerned by the state space model

rho=1025
g=9.81

nb_period = 30
T=12
wave_direction=180
h=5e-2 # time step


###############################################################################
# Functions
###############################################################################


# Excitation force
def f_waves(n_hydro,T,omega,ex_mag,ex_phase,angle,wave_dir):
    w=2*np.pi/T
    H=np.zeros((n_hydro))
    phi=np.zeros((n_hydro))
    k=np.where(wave_dir==angle)
    for i in range(n_hydro):
    	H[i]=np.interp(w,omega,np.reshape(ex_mag[i,k,:],omega.size))
    	phi[i]=np.interp(w,omega,np.reshape(ex_phase[i,k,:],omega.size))
    return(H,phi)


def serie_fourier(t,f,T,n):

    duration=t[-1]-t[0]
    an=np.zeros([n+1])
    bn=np.zeros([n+1])
    an[0] = np.mean(f)  # 1/duration * simps(f,t)
    bn[0] = 0
    
    for i in range(1,n+1):
        an[i] = 2/duration * simps(f*np.cos(i*t*2*np.pi/T),t)
        bn[i] = 2/duration * simps(f*np.sin(i*t*2*np.pi/T),t)
            
    harm=np.linspace(0,n,n+1)*1/T
    mag=np.sqrt(an**2+bn**2)
    phase=np.angle(an-1j*bn)
    return(harm,an,bn,mag,phase)

def rao(time,function,T,nbT=1):
    
    dt = np.mean(np.diff(t))
    nt = int(np.ceil(nbT*T/dt))
    
    t_sf = np.linspace(t[-1]-nbT*T, t[-1], nt)
    f_sf = np.interp(t_sf, time, function)

    harm,an,bn,mag,phase=serie_fourier(t_sf,f_sf,T,n=1)

    return(mag[1],phase[1])


###############################################################################
# Mechanical carcateristics
###############################################################################

R=5 # cylinder radius
H=20 # cylinder height


# mass
m1=795868 
#inertia
I1=1.153*1e7
I2=m1*R**2/2

#Mass matrix
if dim==2:
    MA = np.array([[m1, 0, 0],
                   [0, m1, 0],
                   [0, 0, I1]])
elif dim==3:
    MA = np.array([[m1, 0, 0, 0, 0, 0],
                   [0, m1, 0, 0, 0, 0],
                   [0, 0, m1, 0, 0, 0],
                   [0, 0, 0, I1, 0, 0],
                   [0, 0, 0, 0, I1, 0],
                   [0, 0, 0, 0, 0, I2]])

    
# Initial conditions
theta1_0=0;x1_0=0;z1_0=-7.5;dx1_0=0;dz1_0=0;dtheta1_0=0

# Stiffness matrix
K33=rho*g*np.pi*R**2
K34=0.0;K35=0.0;K45=0.0
K44=rho*g*np.pi/4*R**2*(R**2+H**2/4)
K55=K44

if dim==2:
    Kh = np.array([[0.0, 0.0, 0.0],
                   [0.0, K33, K35],
                   [0.0, K35, K55]])
elif dim==3:
    Kh = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, K33, K34, K35, 0.0],
                   [0.0, 0.0, K34, K44, K45, 0.0],
                   [0.0, 0.0, K35, K45, K55, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


# State space vector 
X0=np.zeros([2*n+NN*(n_hydro)**2])
if dim==2:
    X0[n:2*n]=[x1_0,z1_0,0]
elif dim==3:
    X0[n:2*n]=[x1_0,0,z1_0,0,0,0]


# Dictionnary for comunicating with the solver
parameters={"n":n,"n_hydro":n_hydro,"NN":NN}

# stiffness matrix
parameters["Kh"] = Kh 
 
# modèle de radiation
parameters["Arad"] = Arad 
parameters["Brad"] = Brad 
parameters["Crad"] = Crad 
parameters["Drad"] = Drad

# mass and damping
parameters["MA"] = MA + np.array(Minf) 
parameters["Blin"] = np.zeros([n,n]) 
parameters["Bquad"] = np.zeros([n,n]) 


from tqdm import tqdm


RAOfreq = np.zeros([n,omega.size],dtype=np.complex128)
for i in range(omega.size):
    wi = omega[i]
    aa = Kh - wi**2*(MA+Aw[:,:,i]) + 1j*Bw[:,:,i]
    bb = ex_mag[:,-1,i]*np.exp(1j*ex_phase[:,-1,i])
    RAOfreq[:,i] = np.linalg.lstsq(aa,bb,rcond=None)[0]



RAOm=np.zeros([n,omega.size])
RAOp=np.zeros([n,omega.size])
with tqdm(total=omega.size) as pbar:
    
    for j in range(omega.size):
        
        T = 2*np.pi/omega[j]
        
        pbar.desc="RAO calculation in progress for w = {0:.2f} s".format(omega[j])
        
        if T>10:
            nb_period = 30
        else:
            nb_period = 50
        
        # time vector 
        duration=nb_period*T
        periods = 2*np.pi/omega
        t=np.arange(0,duration,h)
        
        # excitation force parameters
        angle = wave_direction
        H, phi = f_waves(n_hydro,T,omega,ex_mag,ex_phase,angle,wave_dir)
        parameters["T"] = T
        parameters["H"] = H 
        parameters["phi"] = phi
        parameters["tr"] = 15*T
        
        [Xsol,dXsol,fhoule]=pyHSSMsol.RK4(t, X0, parameters) 
        
        
        for i in range(n):
            
            y = Xsol[n+i,:]
            RAOm[i,j], RAOp[i,j] = rao(t,y,T,nbT=1)
            
            
        pbar.update(1)


# Plots

if n == 3:
    titre=['Surge','Heave','Pitch']
    ylabel = ['RAO in m/m','RAO in m/m','RAO in rad/m']
elif n == 6:
    titre=['Surge','Sway','Heave','Roll','Pitch','Yaw']
    ylabel = ['RAO in m/m','RAO in m/m','RAO in m/m','RAO in rad/m','RAO in rad/m','RAO in rad/m']



for i in range(n):
    x = omega
    fig, ax = plt.subplots(2,1)
    ax[0].plot(x,RAOm[i,:],'.-',lw=1)
    ax[1].plot(x,RAOp[i,:],'.-',lw=1)
    
    ax[0].plot(x,np.abs(RAOfreq[i,:]),'k--',lw=1)
    ax[1].plot(x,np.angle(RAOfreq[i,:]),'k--',lw=1)
        
    ax[0].legend(["SSM RAO","HydroStar RAO"])
    plt.minorticks_on()
    
    for k in range(ax.size):
        ax[k].grid(which="major",ls="-")
        ax[k].grid(which="minor",ls=":")

    plt.xlabel('$\omega_{wave}$ in rad/s')
    plt.ylabel(ylabel[i])
    plt.suptitle(titre[i])

