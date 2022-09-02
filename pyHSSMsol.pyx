cimport cython
cimport scipy.linalg.cython_lapack
from libc.stdlib cimport malloc, free
from libc.math cimport cos, sin, exp, tan, pi, atan, tanh, fabs, acos, fmod
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from tqdm import trange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double fonction_rampe(double t, double tr):
    if t < tr:
        return(0.5*(1+cos(pi+(pi*t)/tr)))
    elif t >= tr:
        return(1)       


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cfhoule(int n_hydro, double ti, double tr, double T,double[::1] H,double[::1] phi,double[::1] fh):
    # Valeur de la rampe
    # cdef double tr = 10
    cdef double Rf = fonction_rampe(ti, tr)
    # Calcul de l'effort
    cdef int i=0
    for i in range(n_hydro):
        fh[i]=Rf*H[i]*cos(+2*pi/T*ti+phi[i])
	

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef csolve_dgesv(double[:,::1] A , double[::1] B):
	
    cdef int N = A.shape[0]
    cdef int NRHS = 1
    cdef int LDA = A.shape[0]
    cdef int LDB = A.shape[0]
    cdef int info = 0
	
    cdef int* piv_pointer = <int*>malloc(sizeof(int)*N)
    if not piv_pointer:
        raise MemoryError()	

    try:		
        scipy.linalg.cython_lapack.dgesv(&N,&NRHS,&A[0,0],&LDA,piv_pointer,&B[0],&LDB,&info)	
        if info!=0:
            raise NameError('error in dgesv')
    except NameError:
        raise NameError('error in dgesv')

    finally:
        free(piv_pointer)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cstate_space_function(double ti,double[::1] X0,double[::1] X,double[::1] dX, dict c_parameters,# double[:,::1] B,double[:,::1] C,double[:,::1] tB,double[:,::1] tC,
    double[:,::1] A1,double[:,::1] A2,double[::1] fh,double[::1] frad,double[::1] ftot):
	
	# Entrées de la simulation
    cdef double[:,::1] MA = c_parameters["MA"]
    cdef double[:,::1] Kh = c_parameters["Kh"]
    cdef double[:,::1] Blin = c_parameters["Blin"]
    cdef double[:,::1] Inn = c_parameters["Inn"]
    cdef double[:,:,:,::1] Arad = c_parameters["Arad"]
    cdef double[:,:,::1] Brad = c_parameters["Brad"]
    cdef double[:,:,::1] Crad = c_parameters["Crad"]
    cdef double[:,::1] Drad = c_parameters["Drad"]
    
	
    cdef int n=c_parameters["n"],NN=c_parameters["NN"], n_hydro=c_parameters["n_hydro"]
    cdef int i=0, j=0, k=0, l=0
    cdef double rho=1025.,g=9.81	
	

	###########################################################################
	# Les efforts internes et externes
	###########################################################################

	#Effort de radiation 
    frad[:]=0.0 #reinitialisation obligatoire ici aussi
    if NN != 0:
        for i in range(n_hydro):
            for j in range(n_hydro):
                frad[i] += Drad[i,j]*X[j]
                for k in range(NN):
                    frad[i] += Crad[i,j,k]*X[2*n+i*n*NN+j*NN+k]
                
    # effort d'excitation de la houle 
    cdef double tr = c_parameters["tr"]
    cdef double T = c_parameters["T"]
    cdef double[::1] H = c_parameters["H"]
    cdef double[::1] phi = c_parameters["phi"]
    
    cfhoule(n_hydro, ti, tr, T, H, phi, fh)
    
    
	#Total force 
    for i in range(n):
        ftot[i]=fh[i]-frad[i]
    ftot[n:2*n]=0.0

	
    #Hydrostatic force
    for i in range(n_hydro):
        for j in range(n_hydro):
            ftot[i] -= Kh[i,j]*(X[n+j]-X0[n+j])
          
          
    #Linear damping
    for i in range(n_hydro):
        for j in range(n_hydro):
            ftot[i] -= Blin[i,j]*(X[j]-X0[j])
            
    #Quadratic damping
    for i in range(n_hydro):
        for j in range(n_hydro):
            ftot[i] -= Blin[i,j]*(X[j]-X0[j])*fabs(X[j]-X0[j])
	
	
    # Matrix A1
    A1[:,:]=0.0 
    A1[0:n,0:n]=MA[:,:]
    A1[n:2*n,n:2*n]=Inn[:,:]
	
	# dXdyn
    for i in range(2*n):
        dX[i] = ftot[i]
    for i in range(n):
        dX[n+i] += X[i] 
    csolve_dgesv(A1,dX[0:2*n])

	# dXrad
    if NN != 0:
        for i in range(n_hydro):
            for j in range(n_hydro):
                for k in range(NN):              
                    dX[2*n+i*n*NN+j*NN+k]=Brad[i,j,k]*X[j]                
                    for l in range(NN):
                        dX[2*n+i*n*NN+j*NN+k] += Arad[i,j,k,l]*X[2*n+i*n*NN+j*NN+l]
                    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RK4(t,X0,parameters):
    # Définition des entiers et des réels pour le calcul
    cdef int n_hydro=parameters["n_hydro"], n=parameters["n"], NN=parameters["NN"]
    cdef double ti=0
    cdef int ntot=X0.shape[0]
    cdef int nt=t.shape[0]
    cdef int i,j,k
    cdef double h
    # Ajout de la matrice identité dans le dictionnaire de parameters
    Inn=np.eye(n)
    parameters["Inn"]=Inn
    #Définition des matrices et des vecteurs pour le calcul
    Xsol=np.zeros([ntot,nt])
    Xsol[:,0]= X0[:]
    dXsol=np.zeros([ntot,nt-1])
    F_tot=np.zeros([n,nt-1])
    k1=np.zeros([ntot])
    k2=np.zeros([ntot])
    k3=np.zeros([ntot])
    k4=np.zeros([ntot])
    X2=np.zeros([ntot])
    X3=np.zeros([ntot])
    X4=np.zeros([ntot])		
    A1=np.zeros([2*n,2*n])	
    A2=np.zeros([2*n,2*n])	
    fh=np.zeros([n_hydro])	
    frad=np.zeros([n_hydro])	
    ftot=np.zeros([2*n])	
    dXdyn=np.zeros([2*n])	
    dXrad=np.zeros([NN*n_hydro**2])	
    # Définition des memory_views (copie bas niveau liée à un objet python) pour une meilleure performance de calcul
    cdef double[:,::1] Xsol_v=Xsol
    cdef double[:,::1] dXsol_v=dXsol
    cdef double[:,::1] F_tot_v=F_tot
    cdef double[::1] k1_v=k1 	
    cdef double[::1] k2_v=k2
    cdef double[::1] k3_v=k3
    cdef double[::1] k4_v=k4
    cdef double[::1] X2_v=X2
    cdef double[::1] X3_v=X3
    cdef double[::1] X4_v=X4	
    cdef double[::1] X0_v=X0
    cdef double[::1] X_v=X0.copy()
    cdef double[:,::1] A1_v=A1
    cdef double[:,::1] A2_v=A2
    cdef double[::1] fh_v=fh
    cdef double[::1] frad_v=frad
    cdef double[::1] ftot_v=ftot

   
    # c_Définition du dictionnaire des paramètres
    cdef dict c_parameters = parameters
    
    # Début de la résolution temporelle dans une boucle par RK4	
    for i in range(nt-1):
    
        h=t[i+1]-t[i]
        ti=t[i]
        cstate_space_function(ti, X0_v, X_v,k1_v,c_parameters,  A1_v, A2_v, fh_v,frad_v,ftot_v)
        dXsol_v[:,i]=k1_v[:]
        F_tot_v[:,i]=frad_v[0:n]
		
        for k in range(ntot):
            X2_v[k]=X_v[k]+h/2.0*k1_v[k]
        ti=t[i]+0.5*h
        cstate_space_function(ti, X0_v, X2_v, k2_v,c_parameters,  A1_v, A2_v, fh_v,frad_v,ftot_v)

        for k in range(ntot):
            X3_v[k]=X_v[k]+h/2.0*k2_v[k]
        ti=t[i]+0.5*h
        cstate_space_function(ti, X0_v, X3_v,k3_v,c_parameters,  A1_v, A2_v, fh_v,frad_v,ftot_v)

        for k in range(ntot):
            X4_v[k]=X_v[k]+h*k3_v[k]
        ti=t[i]+h
        cstate_space_function(ti, X0_v, X4_v,k4_v,c_parameters,  A1_v, A2_v, fh_v,frad_v,ftot_v)

        for k in range(ntot):
            X_v[k]+=h/6.0*(k1_v[k]+2.0*k2_v[k]+2.0*k3_v[k]+k4_v[k])
        Xsol_v[:,i+1]=X_v[:]
		
    return(Xsol,dXsol,F_tot)	