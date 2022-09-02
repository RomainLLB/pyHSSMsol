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
    
    
# time vector and excitation force parameters
duration=nb_period*T
periods = 2*np.pi/omega
t=np.arange(0,duration,h)
indice1 = np.argmin(np.abs(wave_dir-wave_direction))
indice2 = np.argmin(np.abs(periods-T))
H = np.array(ex_mag[:,indice1,indice2])
phi = np.array(ex_phase[:,indice1,indice2])

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

# excitation force parameters
parameters["T"] = T
parameters["H"] = H 
parameters["phi"] = phi
parameters["tr"] = 15*T # ramp function for excitation force

# simulation
[Xsol,dXsol,frad]=pyHSSMsol.RK4(t, X0, parameters)


# Plots
motion_keys=["surge","sway","heave","roll","pitch","yaw"]
motion_units=["(m)","(m)","(m)","(rad)","(rad)","(rad)"]
motions=dict()

for i,key in enumerate(motion_keys):
    motions[key]=Xsol[n+i]


for i,key in enumerate(motion_keys):

    fig, ax = plt.subplots()
    ax.plot(t, motions[key])
    ax.set_xlabel("t (s)")
    ax.set_ylabel(key+" "+motion_units[i])
    
    plt.minorticks_on()
    ax.grid(which="major",ls="-")
    ax.grid(which="minor",ls=":")
    
    ax.set_xlim(t[0],t[-1])
    ax.set_title(key)


plt.show()