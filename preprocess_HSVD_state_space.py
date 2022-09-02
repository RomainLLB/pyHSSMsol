# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:39:42 2019

@author: rlecuyer
"""

#from bemio.io import nemoh
import os
import h5py
import numpy as np    
from functions import state_space, radiation


wdir=os.path.abspath('')
nom='simulation_input.h5'
path = wdir+'\\'+nom

with h5py.File(path, 'r') as my_file:
    w=my_file['omega'][()]
    wtot=my_file['wtot'][()]
    A=my_file['A'][()]
    B=my_file['B'][()]
    num_bodies=my_file['num_bodies'][()]
    dim=my_file['dimension'][()]
    Aw=my_file['Aw'][()]
    Bw=my_file['Bw'][()]
    wave_dir=my_file['wave_dir'][()]
    Minf=my_file['Minf'][()]
    ex_mag=my_file['ex_mag'][()]
    ex_phase=my_file['ex_phase'][()]
n = 3*(dim-1)*num_bodies 

wdir=os.path.abspath('')


order=10
threshold=10 
tend=100 
nbt=1001


###############################################################################
##################### State space model for radiation #########################
###############################################################################

t=np.linspace(0,tend,nbt)
dt=np.mean(np.diff(t))
K=radiation(num_bodies, t, wtot, B)
[Arad, Brad, Crad, Drad, Krad, corres] = state_space(t, K, order=order, threshold_value=threshold)



###############################################################################
# Overwrite the h5 file
###############################################################################


    
data_for_saving={'Arad':Arad,'Brad':Brad,'Crad':Crad,'Drad':Drad,\
                 'A':A,'B':B,'Aw':Aw,'Bw':Bw,\
                 'omega':w,'wave_dir':wave_dir,\
                 'num_bodies':num_bodies,'Minf': Minf,\
                 'ex_mag':ex_mag,'ex_phase':ex_phase,\
                 'dimension':dim,'wtot':wtot,'order':order}
                 


path = wdir+'\\'+nom

if os.path.exists(path):
    os.remove(path)
    
with h5py.File(path, 'a') as my_file:
    
    for key in data_for_saving.keys():
        arr=data_for_saving[key]
        data_dir=key
        my_file.create_dataset(data_dir,data=arr)

        
    my_file.flush()


