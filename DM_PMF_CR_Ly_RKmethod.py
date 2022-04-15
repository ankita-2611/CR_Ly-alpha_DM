#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:42 2021

@author: ankita
"""

#%%

from time import time
start = time()
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.special as sp1
from scipy.interpolate import interp1d

zf = 10.0                     # Final redshift
zi = 1010.0                       # Initial redshift
del_z = - 0.05                 # Step-size

z = np.arange(zi, zf, del_z)

'---------------------------'
'--COSMOLOGICAL PARAMETERS--'
'-------(Planck 2018)-------'
'---------------------------'

Omega_b_h2 = 0.02242                 # Physical baryon density parameter
Omega_c_h2 = 0.11933                 # Physical dark matter density parameter
W  = -1                              # Equation state of dark energy 
Omega_b = 0.04897                    # Baryon density parameter
Omega_c = 0.26067                    # Dark Matter density parameter
Omega_m = 0.3111                     # Matter density parameter
Omega_lambda = 0.6889                # Dark Energy density parameter
Omega_k = 0.0
h = 0.6766
H0 = (100*h)*(1e3/3.086e22)          # Hubble constant in s^-1, originally km s-1 Mpc-1

"NOTE:- Omega_k = 0 for LCDM model"
"NOTE:- Omega_rad = 0 if we consider the above values"

'--------------------------------------------'
'------OTHER CONSTANTS (all in SI unit)------'
'--------------------------------------------'

hp = 6.626e-34                      # Planck constant
c = 3.0e8                           # Speed of light
me = 9.1e-31                        # Mass of electron
kB = 1.38e-23                       # Boltzmann constant
sigma_T = 6.6524e-29                # Thomson scattering cross-section 
sigma_SB = 5.67e-8                  # Stefan-Boltzmann constant
T0 = 2.725                          # CMB Temperature at z = 0
del_E = 1.632e-18                   # Ly_alpha transition energy
E2 = 5.44e-19                       # Energy of 2s orbital of hydrogen
E1 = 2.176e-18                      # Energy of 1s orbital of hydrogen
G = 6.67e-11                        # Gravitational constant

'--------------------------------------------'
'------OTHER CONSTANTS (all in CGS unit)-----'
'--------------------------------------------' 

G_CGS = 6.67e-8                     # in cm3 gm-1 sec-2
Mpc = 3.085677581e24                # in centimetres
gauss = 1.0e-4                      # in tesla
H0_CGS = (100*h)*(1e3/3.086e22)     # 100*h*10**5.*Mpc**-1., same in SI and CGS unit s^-1
kB_CGS = 1.38e-16                   # cm2 g s-2 K-1 or egr/K
m0 = 1.6726e-24                     # gm
c_cgs = 3e10                        # cm s^-1
m_p = 938.272                       # rest mass energy of proton in MeV
m_p_cgs = m_p* 1.602e-6 #0.00150534 # rest mass energy of proton in erg
mp = 1.6726e-24                     # mass of proton in gram

'---------------------parameters of X-ray model----------------------------------'

E1s = 2.176e-11                     # in erg
Xray_ON = 0.0                       # write '1' to enable decaying turbulence heating, '0' to disable it

source_model = 2                    # 1 = STARBURST ! 2 = SNR ! 3 = MINIQUASARS !
f_star = 1.0                        #in units of 0.01      # The data of Xray heating by Raghu is for f_star = 0.01. Change the value accordingly

'-------------------free parameters in DM-b int model--------------------------------'

mx = 0.1                             # Mass of dark matter particle in GeV
mb = 0.938                          # Mass of baryonic particle in GeV
sigma_45 = 2. #2. #278.2559402207126   # Interaction cross-section in unit of 10^-45 m^2

'----------------------parameters in PMF model------------------------------------'

B0 = 0.3e-9                         # magnetic field gauss
E1s = 2.176e-11                     # in erg
AD_ON = 0.                          # write 1 to enable ambipolar diffusion heating
DT_ON = 0.                          # write 1 to enable decaying turbulence heating

'--------------------parameters of cosmic ray model--------------------------------------'

CR_ON = 1.
q = 2.2
eta = 7
       
eps_popIII = 0.1
eps_popII = 0.2

f_SN_PopIII = 1/145
f_SN_PopII = 0.02

E_SN_PopIII = 6.90e49 * 145  # in erg
E_SN_PopII = 1e51            # in erg

CR_heat_popII = 0.2       #heating fraction due to PopII stars

'----------------------------'
'''HELIUM FRACTION'''
'----------------------------'

Y = 0.24
f_He = 0.079

'-------------------------------------------------------------'
'-------CONSTANTS FOR COLLISIONAL COUPLING COEFFEICIENTS------'
'-------------------------------------------------------------'

T_star = 0.068             
A_10 = 2.85e-15          # Einstein coefficient for spontaneous emission

'-----------------------------------------'
'''FUNCTION TO DETERMINE CMB TEMPERATURE'''
'-----------------------------------------'

def T_CMB(red):
    T_gamma = T0*(1.0 + red)
    return T_gamma

'------------------------------------------'
'''FUNCTION TO DETERMINE HUBBLE PARAMETER'''
'------------------------------------------'
'We have assumed a matter dominated universe'

def H(red):
    H_z= H0*(Omega_m*(1.0 + red)**3.0)**0.5
    return H_z

'-------------------------------------------------'
'''FUNCTION TO DETERMINE HUBBLE PARAMETER IN CGS'''
'-------------------------------------------------'
'We have assumed a matter dominated universe'

def H_CGS(red):
    return H0_CGS*(Omega_m*(1.0 + red)**3.0)**0.5
    
'---------------------------------------------------------'
'''FUNCTION TO DETERMINE NEUTRAL HYDROGEN NUMBER DENSITY'''
'---------------------------------------------------------'

def nH(red):
    NH = 8.403*Omega_b_h2*(1.0 + red)**3.0              #m-3
    return NH

def nH_CGS(red):
    return 8.403e-6*Omega_b_h2*(1.0 + red)**3.0          #cm-3

'-----------------------------------------------------------'
'''FUNCTION TO DETERMINE RECOMBINATION COEFFICIENT (alpha)'''
'-----------------------------------------------------------'

def alpha_e(Tg): #SI unit
    a = 4.309
    b = -0.6166
    cp = 0.6703
    d = 0.5300
    F = 1.14             # Fudge factor
    t = Tg/(1.0e4)
    
    alpha = F*1.0e-19*((a*t**b)/(1.0 + cp*t**d))    #m3 s-1
    return alpha

'------------------------------------------------------------'
'''FUNCTION TO DETERMINE PHOTOIONIZATION COEFFICIENT (beta)'''  
'------------------------------------------------------------'
 
def beta_e(T_gamma): #SI unit, note that T_gamma has been used to calculate beta as suggested in chluba, 2015, mnras
    return alpha_e(T_gamma)*2.4093e21*T_gamma**(1.5)*np.exp(-39420.289/T_gamma)


'---------------------------'
'''FUNCTION TO DETERMINE C'''
'---------------------------'

def C1(red,x, Tg): #unit less
    K = 7.1898e-23/(H(red))
    Lambda = 8.22458 
    
    Cr = (1.0 + K*Lambda*(1.0 - x)*nH(red))/(1.0 + K*(Lambda + beta_e(Tg))*(1.0 - x)*nH(red))
    return Cr


'----------------------------------------------------'
'''FUNCTION TO DETERMINE DARK MATTER ENERGY DENSITY'''
'----------------------------------------------------'

def rho_DM(red):  #SI unit
    rho_DM_eng =  (Omega_m-Omega_b)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_DM_eng

'-----------------------------------------------'
'''FUNCTION TO DETERMINE MATTER ENERGY DENSITY'''
'-----------------------------------------------'

def rho_M(red): #SI unit
    rho_M_eng = (Omega_m)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_M_eng    

'-----------------------------------------------'
'''FUNCTION TO DETERMINE BARYON ENERGY DENSITY'''
'-----------------------------------------------'

def rho_B(red): #SI unit
    rho_B_eng = (Omega_b)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_B_eng

'---------------------------------------------'
'''FUNCTION TO DETERMINE BARYON MASS DENSITY'''
'---------------------------------------------'

def rho_b_CGS(red): #CGS unit
    return (Omega_b)*((3*H0_CGS**2)/(8*np.pi*G_CGS))*(1.0 + red)**3

'------------------------------'
'''FUNCTION TO DETERMINE u_th'''
'------------------------------'

def u_th(Tg, Tx): #SI unit m/s
    return c*2.936835e-7*(((Tg/mb) + (Tx/mx))**(0.5))

'------------------------------'
'''FUNCTION TO DETERMINE F(r)'''
'------------------------------'
  
def F_r(vel_xb, Tg, Tx): #unit less
    u_therm = u_th(Tg, Tx)
    rv = vel_xb/u_therm
    F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
    return F

'----------------------------------------'
'''FUNCTION TO DETERMINE F(r)/Vel_xb^2'''
'----------------------------------------'
def Fr_by_velxb2(vel_xb, Tg, Tx): #depends on the unit of Vel_xb^2
    u_therm = u_th(Tg, Tx)
    rv = vel_xb/u_therm
    if rv >= 0.1:
        F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
        F = F/vel_xb**2
        return F
    else:
        F = np.sqrt(2.0/np.pi)*(rv/3.0 - rv**3.0/10.0 + rv**5.0/56.0)
        F = F/u_therm**2
        return F
        

'----------------------------------------'
'''FUNCTION TO DETERMINE F(r)/Vel_xb'''
'----------------------------------------'
def Fr_by_velxb(vel_xb, Tg, Tx): #depends on the unit of Vel_xb^2
    u_therm = u_th(Tg, Tx)
    rv = vel_xb/u_therm
    if rv >= 0.1:
        F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
        F = F/vel_xb
        return F
    else:
        F = np.sqrt(2.0/np.pi)*(rv**2.0/3.0 - rv**4.0/10.0 + rv**6.0/56.0)
        F = F/u_therm
        return F
        

'---------------------------------------'
'''FUNCTION TO DETERMINE THE DRAG TERM'''
'---------------------------------------'

def Drag_vxb(vel_xb, red, Tg, Tx): #SI unit
    D_Vxb2 = 1.*2.63e+7*h*(Omega_m**0.5)*((1.0 + red)**0.5)*sigma_45*Fr_by_velxb2(vel_xb, Tg, Tx)/(mb + mx)
    return D_Vxb2


'---------------------------------------'
'''FUNCTION TO DETERMINE Q_b_coupling'''
'---------------------------------------'

def Q_b_coupling(Tx, Tg, red, vel_xb): #SI unit
    u_therm = u_th(Tg, Tx)
    rv = vel_xb/u_therm
    Q_b1 = (2.0/3.0)*2.10e+7*(Omega_m - Omega_b)*(h**2.0)*((1.0 + red)**0.5)*sigma_45*np.exp(-(rv**2)/2.0)*(mb/((mb + mx)**2.0))*(Tx - Tg)/(h*(Omega_m**0.5)*u_therm**3)/1.
    return Q_b1

'----------------------------------'
'''FUNCTION TO DETERMINE Q_b_drag'''
'----------------------------------'

def Q_b_drag(Tx, Tg, red, vel_xb): #SI unit
    Q_b_d = 1.*2.26e+3*(Omega_m - Omega_b)*(h**2.)*((1.0+red)**0.5)*sigma_45*(mb*mx/((mb + mx)**2.0))*Fr_by_velxb(vel_xb, Tg, Tx)/(h*Omega_m**0.5) 
    return Q_b_d


'-----------------------------'
'''FUNCTION TO DETERMINE Q_x'''
'-----------------------------'

def Q_x_coupling(Tx, Tg, red, vel_xb): #SI unit
    u_therm = u_th(Tg, Tx)
    rv = vel_xb/u_therm
    Q_x1 = (2.0/3.0)*2.10e+7*Omega_b*(h**2.0)*((1.0 + red)**0.5)*sigma_45*np.exp(-(rv**2)/2.0)*(mx/((mb + mx)**2.0))*(Tg - Tx)/(h*(Omega_m**0.5)*u_therm**3)/1.
    return Q_x1


'----------------------------------'
'''FUNCTION TO DETERMINE Q_x_drag'''
'----------------------------------'

def Q_x_drag(Tx, Tg, red, vel_xb): #SI unit
    Q_x_d = 1.*2.26e+3*Omega_b*h**2*(1.0+red)**0.5*sigma_45*(mb*mx/((mb + mx)**2.0))*Fr_by_velxb(vel_xb, Tg, Tx)/(h*Omega_m**0.5)  
    return Q_x_d

'----------------------------------'
'''FUNCTION TO DETERMINE MAG FIELD AND RELEVANT'''
'----------------------------------'

def B(red):
    return B0*(1. + red)**2.

def rho_B_pmf(red):
    return B0**2.*(1. + red)**4./(8.0*np.pi) # CGS unit erg/cm^3

k_max = 286.91*(1.0e-9/B0)  # 1/Mpc unit
#L = 1.0/k_max    # Mpc unit

def l_d(red):
    return (1.075e22)*(B0/1.0e-9)/(1+red)  # cm unit
    
'----------------------------------------'
'''FUNCTION TO DETERMINE SPECTRAL INDEX'''
'----------------------------------------'

n = -2.9
m = 2.0*(n+3.0)/(n+5.0)

'-------------------------------------------------------'
'''FUNCTION TO DETERMINE COLLISIONAL IONIZATION COEFF.'''
'-------------------------------------------------------'

def k_ion(Tg): #CGS unit,  cm 3 s −1, denoted as gamma in sethi, subramanian paper, 2005
    U = np.abs(E1s/(kB_CGS*Tg))
    return 0.291e-7*U**0.39*np.exp(-U)/(0.232+U)    #cm3s-1 adopted from Section B, Minoda et. al. 2017 paper

'----------------------------------'
'''FUNCTION TO DETERMINE TIME'''
'----------------------------------'

tibytd = 14.8*(1.0e-9/B0)*(1.0/k_max)  #t = t_i/t_d, k_max should be in unit of Mpc^-1

'--------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE RATE OF ENERGY INPUT DUE TO AMBIPOLAR DIFFUSION'''
'--------------------------------------------------------------------------'

#def L_square(red):
#    return (rho_B_pmf(red)/l_d(red))**2.

def f_L(n):
    return 0.8313*(1 - 1.02e-2*n)*(n)**1.105

def curl_B(red, n, rho_B_red): # in CGS unit
    return ((rho_B_red/l_d(red))**2.)*f_L(n+3.0)

def gamma_AD(red, Tg, x, n, rho_B_red): # in CGS unit #((1./8.*np.pi)**2.)*
    return (0.126*((1.-x)*(rho_B_red**2.)*((1.0e-9/B0)**2.)*f_L(n+3.0))/(x*(Tg**0.375)*(h**4.)*(Omega_b**2.)*(1.+red)**4.))         #in cm-1 g s-3

def pmf_AD(red, Tg, x, n, rho_B_red):
    #print(red)
    return ((2.236e37/(1.+red)**9.5)*((1.-x)*(rho_B_red**2.)*((1.0e-9/B0)**2.)*f_L(n+3.0))/(x*(Tg**0.375)*(h**7.)*(Omega_b**3.)*(Omega_m**0.5)*(1.+f_He+x)))

def rho_AD(red, Tg, x, n, rho_B_red):
    return ((3.93e16*(1.0-x)*((1.0e-9/B0)**2.0)*f_L(n+3.0)*rho_B_red**2.)/(x*((1.0+red)**6.5)*(h**5)*np.sqrt(Omega_m)*(Omega_b**2.0)*(Tg**0.375)))
    
'------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE RATE OF ENERGY INPUT DUE TO DECAYING TURBULENCE (DT)'''
'------------------------------------------------------------------------------'

def gamma_DT(red, rho_B_red):
    return (4.86e-18*m*(np.log(1+tibytd))**m*rho_B_red*h*Omega_m**0.5*(1.+red)**1.5)/((np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1))

def pmf_DT(red, x, rho_B_red):
    return (8.623e20/(1.+red)**4.)*(m*(np.log(1+tibytd))**m*rho_B_red)/(Omega_b*h**2.*(1.+f_He+x)*(np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1))

def rho_DT(red, rho_B_red):
    return ((1.5*m*rho_B_red*(np.log(1.+tibytd))**m)/((1.+red)*(np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1)))




'--------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE RATE OF ENERGY INPUT DUE TO COSMIC RAYS'''
'--------------------------------------------------------------------------'

'''Contribution from Pop III stars'''
'-------------------------------------'


beta_i = 0.001
beta_f = 0.999
del_beta = 0.004

beta_arr = np.arange(beta_i, beta_f, del_beta)
  
 
Ek_i = 0.001                            # 1 keV in terms of MeV
Ek_f = 1e9                              # 10^6 GeV in terms of MeV

Ek_arr = np.linspace(Ek_i, Ek_f, num=len(beta_arr))

zi_CR_popIII = 10.0
zf_CR_popIII = 50.0
z_cal_popIII = np.arange(zf_CR_popIII, zi_CR_popIII, del_z)
  

#def R(red):
#    return 7.5e-4* eta* red**(2.5)* np.exp(-0.8* (red - 3.5)) # Mo yr-1 Mpc-1

z_popIII, SFR_popIII = np.loadtxt('new_SFRD_popIII_NLW4800_self_consistent_with_feedbackfromPopII_fromz50_delz0.05.txt', unpack=True)
def R_popIII(red):
    count = int((red - zf_CR_popIII)/del_z)
    star_form = SFR_popIII[count]
    return star_form

#E_SN = 1e51 #6.90e49 * 145  # in erg
def E_cr_dot(red):
    return 1e-30* eps_popIII* R_popIII(red)* (E_SN_PopIII/1e51)* f_SN_PopIII* (1.+red)**3.


E_store=[]
for i in range(len(z_cal_popIII)):
    E_val = (1e-30* eps_popIII* (E_SN_PopIII/1e51)* R_popIII(z_cal_popIII[i])* f_SN_PopIII* (1.+z_cal_popIII[i])**3.) / (H_CGS(z_cal_popIII[i]) * (1.+z_cal_popIII[i]))
    E_store.append(E_val)



def beta_E(Ek):
    Ep = (1.*Ek)/m_p
    return (np.sqrt(Ep**2. + 2.*Ep))/(1.*Ep + 1)

def Ek_beta(beta):
    return m_p * ((1. - beta**2.)**(-0.5) - 1.)



#Beta_inte = lambda b,q: b**(-q)* ((1-b**2.)**(-0.5) - 1)* (1-b**2.)**(0.5*(q-3.))
#I_beta, err = scint.quad(Beta_inte, beta_E(Ek_i), beta_E(Ek_f), args=(q))

#Beta_inte(beta_arr, q):
y_beta = beta_arr**(-q)* ((1. - beta_arr**2.)**(-0.5) - 1)* (1. - beta_arr**2.)**(0.5*(q-3.))
x_beta = np.linspace(beta_E(Ek_i), beta_E(Ek_f), num=len(beta_arr))
I_beta = scint.simps(y_beta, x_beta)
#return I_beta
    
    
def N_0dot(red, beta, q):
    return E_cr_dot(red)/(m_p_cgs* I_beta)
    
    
def dn_dz(red, beta, q):
    check_E = Ek_beta(beta)
    if (check_E < 1.*Ek_i): # or check_E > Ek_f):
        ret_val = 0.
    else:
        ret_val = (-3.086e17* N_0dot(red, beta, q)* (((1. - beta**2.)/beta**2.)**(0.5*q)))/ ((1.+ red)**(2.5)* h * np.sqrt(Omega_m) * (1. - beta**2.)**(1.5)) 
    return ret_val


def e_collision(red, Tg, x, beta):
    x_m = 0.0286* np.sqrt(Tg/(2e6))
    return 8.4808e-4* ((Omega_b * h)/Omega_m**0.5) * (1. + red)**0.5* x * ((beta* (1 - beta**2.)**1.5)/(x_m**3. + beta**3.))
    

def H_ionization(red, Tg, x, beta):
    if(beta < 0.01):
        th_beta = 0.
    else:
        th_beta = 1. #+ 0.0185* np.log(beta)
    return 5.0314e-4* ((Omega_b * h)/Omega_m**0.5) * (1. + red)**0.5* (1. - x)* (1. + 0.0185* np.log(beta)* th_beta) * ((2.* beta* (1 - beta**2.)**1.5)/(1e-6 + 2.* beta**3.)) 
#(1. + 0.0185* np.log(beta)* th_beta)



th_b = np.zeros(len(beta_arr))
#def theta_fn(beta):
for j in range(len(beta_arr)):
    if(beta_arr[j] < 0.01):
        th_b[j] = 0.
    else:
        th_b[j] = 1. #+ 0.0185* np.log(beta_arr[j])
#return th_b   
'''
def H_ionization_arr(red, Tg, x, beta):
    return 5.0314e-4* ((Omega_b * h)/Omega_m**0.5) * (1. + red)**0.5* (1. - x)*  (1. + 0.0185* th_b* np.log(beta)) * ((2.* beta* (1 - beta**2.)**1.5)/(1e-6 + 2.* beta**3.))
#(1. + 0.0185* th_b* np.log(beta))
'''

def ad_expansion(red, beta):
    return (beta* (1. - beta**2.))/(1. + red)


def dbeta_dz(red, Tg, x, beta):
    return e_collision(red, Tg, x, beta) + H_ionization(red, Tg, x, beta) + ad_expansion(red, beta)

'''
def dE_dz(red, Tg, x, beta):
    return (e_collision(red, Tg, x, beta) + H_ionization_arr(red, Tg, x, beta)) * m_p_cgs * beta * ((1. - beta**2.)**(-1.5))
'''

N_cr_eq = np.zeros(len(beta_arr))
def N_cr_injected(red, beta, q):
    N_cr_eq = dn_dz(red, beta, q) * del_z
    return N_cr_eq



#N_cr_intp = interp1d(beta_arr, N_cr_eq)
row_len = int((zf - zf_CR_popIII)/del_z) + 1
N_cr = np.zeros((row_len, len(beta_arr)))
N_cr_1 = np.zeros((row_len, len(beta_arr)))
N_cr_2 = np.zeros((row_len, len(beta_arr)))
N_cr_prime = np.zeros((row_len, len(beta_arr)))
#beta_new_evolve = np.zeros(len(beta_arr))
beta_new_cal = np.zeros(len(beta_arr))
#N_cr.append(N_cr_evolve(red, q))



flag = -1

def N_cr_cal(red, Tg, x, q):
    count = int((red - zf_CR_popIII)/del_z)
    #print(red)
    #print("count=%d" %count)
    global flag
    #print ("global_flag=%d" %flag)
    if (count < 0):
        ydata = 0.0
    elif (count == flag):
        ydata = N_cr[count,:]
    else:
        #print('loop is running')
        if(count == 0):
            for j in range (len(beta_arr)):
                #beta_cal = beta_i + j* del_beta
                N_cr[count,j] = (N_cr_injected(red, beta_arr[j], q)) #*(-1.* del_z)
            ydata = (N_cr[count,:])
            #print(ydata)
        else:         
            for j in range(len(beta_arr)):
                #beta_cal = beta_i + j* del_beta
                N_cr_1[count,j] = (N_cr_injected(red, beta_arr[j], q)) #*(-1.* del_z)
                    
                beta_new_evolve = beta_arr[j] + dbeta_dz((red - del_z), Tg, x, beta_arr[j])* del_z
                if(beta_new_evolve > 0.0):
                    beta_new_cal[j] = beta_new_evolve
                    N_cr_prime[count-1, j] = N_cr[count-1, j] * (beta_new_evolve/(beta_arr[j])) #((beta_arr[j])/beta_new_evolve) 
                else:
                    beta_new_cal[j] = 0.0
                    N_cr_prime[count-1, j] = 0.0
                        
            N_cr_intp = interp1d(beta_new_cal, N_cr_prime[count-1,:])
            for j in range(len(beta_arr)):
                search_b = beta_arr[j]
                if (search_b < min(beta_new_cal) or search_b > max(beta_new_cal)):
                    N_inp = 0.0
                else:
                    N_inp = N_cr_intp(search_b)
                N_cr_2[count,j] = (N_inp *((1. + red)/(1. + red - del_z))**3.) #*(-1.* del_z)
                
                N_cr[count,j] = N_cr_1[count,j] + N_cr_2[count,j]
            ydata = N_cr[count,:]
        
    flag = count
    #print ("local_flag=%d" %flag)
    return ydata #/(-1.* del_z)

dE_store = np.zeros(len(beta_arr))
def dE_dz(red, Tg, x, beta_arr):
    xm = 0.0286* np.sqrt(Tg/(2e6))
    e_coll = 1.28e-6* ((Omega_b * h)/Omega_m**0.5) * (1. + red)**0.5* x * ((beta_arr**2.)/(xm**3. + beta_arr**3.))
    H_ion = 7.57e-7* ((Omega_b * h)/Omega_m**0.5) * (1. + red)**0.5* (1. - x)* (1. + 0.0185* th_b* np.log(beta_arr)) * ((2.* beta_arr**2.)/(1e-6 + 2.* beta_arr**3.))
    E_val = e_coll + H_ion
    for j in range(len(beta_arr)):
        E_check = Ek_beta(beta_arr) *(1.602e-6)
        if (((-1.*del_z)* E_val[j] - E_check[j]) > 0.0):
            dE_store[j] = (E_check[j]) /(-1.*del_z)
        else:
            dE_store[j] = E_val[j] 
    return dE_store
    #return (e_collision(red, Tg, x, beta) + H_ionization_arr(red, Tg, x, beta)) * m_p_cgs * (beta/((1. - beta**2.)**(1.5)))


#gamma_store_final = []
y_beta_gamma = np.zeros(len(beta_arr))
#eta1, eta2, eta3 = 5./3, 1.17, 1.43
gamma_store = []
z_store = []
def gamma_CR(red, Tg, x, q):
    #eta1, eta2, eta3 = 5./3, 1.17, 1.43
    count = int((red - zf_CR_popIII)/del_z)
    if(count < 0):
        val = 0.0
    else:
        y_beta_gamma = (dE_dz(red, Tg, x, beta_arr)* N_cr_cal(red, Tg, x, q))
        gamma_I_beta = scint.simps(y_beta_gamma, beta_arr)
        val = 1.743 * gamma_I_beta
        gamma_store.append(val)
        z_store.append(red)
        #val = (5./8)* eta1* eta2* eta3* gamma_I_beta
    return val

    
    
def Q_CR_popIII(red, Tg, x, q):
    return ( (5.75e20* gamma_CR(red, Tg, x, q))/(Omega_b_h2* (1.+ f_He+ x)* (1. + red)**(3.)) )
    #return (1.77e23* gamma_CR(red, Tg, x, q))/(Omega_b_h2* np.sqrt(Omega_m)* h* (1.+ f_He+ x)* (1.+red)**(5.5))


sigma_HI = np.zeros(len(beta_arr))
for j in range(len(beta_arr)):
    sigma_cal = (6.2 + np.log10(beta_arr[j]**2./(1. - beta_arr[j]**2.)) - 0.43* beta_arr[j]**2.)/beta_arr[j]
    if(beta_arr[j] >= 0.026):
        sigma_HI[j] = sigma_cal
    else:
        sigma_HI[j] = 0.0     
#sigma_HI = (6.2 + np.log10(beta_arr**2./(1. - beta_arr**2.)) - 0.43* beta_arr**2.)/beta_arr

def x_e_CR_popIII(red, Tg, x, q):
    count = int((red - zf_CR_popIII)/del_z)
    if(count < 0):
        val1 = 0.0
    else:
        y_beta_xe = (sigma_HI* N_cr_cal(red, Tg, x, q))
        xe_I_beta = scint.simps(y_beta_xe, beta_arr)
        val1 = (1.14e8 * (1. - x)* xe_I_beta)/(h* np.sqrt(Omega_m)* (1. + red)**(2.5))
    return val1
 

cr_new = []
red_new = []


'''Contribution from Pop II stars'''
'-------------------------------------'

zi_CR_popII = 1.0
zf_CR_popII = 30.0
z_cal_popII = np.arange(zf_CR_popII, zi_CR_popII, del_z)

z_popII, SFR_popII = np.loadtxt('SFRD_popII_interpolated_delz0.05.txt', unpack=True)

def R_popII(red):
    count = int((red - zf_CR_popII)/del_z)
    star_form = SFR_popII[count]
    return star_form


def Edot_popII(red):
    return (eps_popII* (E_SN_PopII/1e51)* R_popII(red)* f_SN_PopII) #(1e-30* (1.+red)**3.) it will get cancelled out in the calculation of Q_CR_popII

def Q_CR_popII(red):
    if (red > 30.):
        popII_return = 0.
    else:
        popII_return = (5.75e-10* CR_heat_popII* Edot_popII(red))/(H_CGS(red)* (1.+red)* Omega_b_h2)
    return (popII_return)



'-------CALCULATION OF Ly-alpha COUPLING COEFFICIENT------'

z_ly, J_ly = np.loadtxt('new_J_ly_alpha_PopIII_NLy4800_and_PopII_fromz50_delz0.05.txt', unpack=True)
    
f_alpha = 0.4162
S_alpha = 1.0

J_alpha = np.zeros(19200) #(1010-50)/del_z; 96000 for del_z 0.01 & 19200 for del_z 0.05
J_ly_alpha = np.append(J_alpha, J_ly)



'----------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE GAS KINETIC TEMPERATURE (Tg) AND IONIZATION FRACTION (x)'''
'----------------------------------------------------------------------------------'

def func(r,red):
    Tg = r[0]
    x = r[1]
    Tx = r[2]
    V_xb = r[3]
    rho_B_red = r[4]
    
    f_Tg = ((2.0*Tg)/(1.0 + red)) - ((2.70877e-20*(T_CMB(red) - Tg)*(1.0 + red)**(1.5)*(x/(1.0 + f_He + x)))/(H0*np.sqrt(Omega_m))) - Q_b_coupling(Tx, Tg, red, V_xb) - Q_b_drag(Tx, Tg, red, V_xb)/1. - AD_ON*pmf_AD(red, Tg, x, n, rho_B_red) - DT_ON*pmf_DT(red, x, rho_B_red) - CR_ON*(Q_CR_popIII(red, Tg, x, q) + Q_CR_popII(red))  
    f_x =  (C1(red,x,Tg)*(alpha_e(Tg)*x**2*nH(red) - beta_e(T_CMB(red))*(1.0 - x)*np.exp(-118260.87/T_CMB(red))))/(H(red)*(1.0 + red))  - (k_ion(Tg)*nH(red)*(1.-x)*x)/(H(red)*(1.0 + red)) - (x_e_CR_popIII(red, Tg, x, q))
    f_Tx = ((2.0*Tx)/(1.0 + red)) - Q_x_coupling(Tx, Tg, red, V_xb) - Q_x_drag(Tx, Tg, red, V_xb)/1. 
    f_Vxb = (V_xb/(1.0 + red)) + Drag_vxb(V_xb, red, Tg, Tx)
    f_rho_B = (4.*rho_B_red)/(1.0 + red) + AD_ON*rho_AD(red, Tg, x, n, rho_B_red) + DT_ON*rho_DT(red, rho_B_red) #+ 1.*(1.*gamma_DT(red,rho_B_red)/(H_CGS(red)*(1.+red)) + 0.*gamma_AD(red,Tg,x,n,rho_B_red)/(H_CGS(red)*(1.+red)))
#    print(red, f_x, f_Vxb)
#    cr_new.append(Q_CR(red, Tg, x, q))
#    red_new.append(red)
    return np.array([f_Tg, f_x, f_Tx, f_Vxb, f_rho_B], float)


'-------------------------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE THE VALUE OF PROBABILITY DISTRIBUTION FOR EACH INTIAL RELATIVE VELOCITY'''
'-------------------------------------------------------------------------------------------------'

def prob_func(v_xb):
    return 4*np.pi*v_xb**2*(np.exp((-3*v_xb**2)/(2*Vrms**2)))*(1.0/((2.0/3)*np.pi*Vrms**2)**(1.5))
 


Vxb = np.arange(2.0e-6*c, 5.0e-4*c, 2.0e-5*c)       # Range of initial relative velocties
Vrms = 1.0e-4*c                                     # RMS velocity
T_21_xb = []                                        # Array to store T_21 data for each initial conditions (array within and array)
P_Vxb = []                                          # the value of the probabilty distribution for each initial velocity
x_frac_vxb = []
T_gas_vxb = []
T_dark_vxb = []
T_spin_vxb = []

x_c_store = []
x_alpha_store = []
x_tot_store = []

xfrac_standard_17 = 0.0001958782928613447

h_step = del_z

for vxb in Vxb:
        
    T_gas = [] 
    x_points = [] 
    T_dark = [] 
    Vel_xb = [] 
    B_energy_density = [] 
    
    '----------------------------------------------------'
    '''---------------INITIAL CONDITIONS---------------'''
    '---- (Tg = T_CMB at z = 1010 and x = 0.05497) ----'''
    '-----(Tx = 0.0 at z = 1010 and V_xb_0 = 1e-4*c)---'''
    '----------------------------------------------------'
    
    r = np.array([T_CMB(zi), 0.05497, 0.0, vxb, rho_B_pmf(zi)], float)     # Initial conditions for Tg, x, Tx and V_xb # 0.95919324
    #r = np.array([164, 2.5e-4, 0.0, vxb, rho_B_pmf(zi)], float)             # Initial condition at z=100
    #r = np.array([34, 2e-4, 0.0, vxb, rho_B_pmf(zi)], float)               # Initial condition at z=40

    '''------SOLVING THE EQUATIONS--------'''
    
    for red in z:
        T_gas.append(r[0])                              # Stores Tg as an array in K
        x_points.append(r[1])                           # Stores x as an array 
        T_dark.append(r[2])                            # Stores Tx as an array in K
        Vel_xb.append(r[3])                             # Stores V_xb as an array in m/s
        B_energy_density.append(r[4])                  # Stores magnetic energy density 
        #print(red, r[0])
        k1 = h_step * func(r , red)
        #k2 = h_step * func(r + 0.5*k1, red + 0.5*h_step) 
        #k3 = h_step * func(r + 0.5*k2, red + 0.5*h_step) 
        #k4 = h_step * func(r + k3, red + h_step)
        r = r + k1 #((k1 + 2.*k2 + 2.*k3 + k4)/6.)

    
    
    '''r = scint.odeint(func, r0, z, atol=1e-4, rtol=1e-6, hmax=abs(del_z), hmin=0.0)#, mxstep=2)               # Solving the coupled differential equation
    T_gas = r[:,0]                              # Stores Tg as an array in K
    x_points = r[:,1]                           # Stores x as an array 
    T_dark = r[:,2]                             # Stores Tx as an array in K
    Vel_xb = r[:,3]                             # Stores V_xb as an array in m/s
    B_energy_density = r[:,4]'''
    
    #B_z = np.sqrt(8.*np.pi*np.ndarray(B_energy_density))
    #mag_field0 = B0*(1. + z)**2.
    
    '----------CMB TEMPERATURE---------'

    T_gamma = T0*(1.0 + z)
    
    
    '-------CALCULATION OF COLLISIONAL COUPLING COEFFICIENT------'
    
    K_HH = 3.1e-17* (np.asarray(T_gas)**(0.357))* np.exp(-32.0/np.asarray(T_gas))     # Using the fitting formula
    nHI = 8.403*Omega_b_h2*(1.0 + z)**3.0
    
    C_10 = K_HH*nHI
    
    x_c = (4.* T_star* C_10)/(3.* A_10* T_gamma)     # Collisional coupling coefficient 
    
    
    '-------CALCULATION OF TOTAL COUPLING COEFFICIENT------'
    
    #S_alpha = ( np.exp(-0.37*(1.0 + z)**0.5 * np.asarray(T_gas)**(-2./3.)) )/(1.0 + 0.4/np.asarray(T_gas))
    x_alpha = (0.0494* T_star* f_alpha* S_alpha* J_ly_alpha)/(A_10* T_gamma)
    
    x_tot = x_c + x_alpha 
    
    x_c_store.append(x_c)
    x_alpha_store.append(x_alpha)
    x_tot_store.append(x_tot)
    
    '------CALCULATION OF SPIN TEMPERATURE------'
    
    T_spin = ((1.0 +  x_tot)* T_gas* T_gamma)/(x_tot*T_gamma + T_gas)
    #T_spin = 57.

    T_21 = .023*((0.15/Omega_m)*((1.0 + z)/10))**(0.5)*(Omega_b*h/0.02)*(1.0 - (T_gamma/T_spin))  #T_21 brightness temperature in mK
    
    
    P_Vxb.append(prob_func(vxb))
    T_gas_vxb.append(np.array( T_gas))
    T_dark_vxb.append(np.array( T_dark))
    T_21_xb.append(np.array( T_21))
    x_frac_vxb.append(np.array( x_points))
    T_spin_vxb.append(np.array( T_spin))

    
'''------The following steps does a statistical average of the T_21 signal over the velocity distribution---'''

#mag_field = (np.array(B_z)/mag_field0)
    
T_gas_avg = []
T_dark_avg = []
T_b_avg = []
x_frac_avg = []
T_spin_avg = []
for i in range(len(P_Vxb)):
    T_gas_avg.append(T_gas_vxb[i]*P_Vxb[i])
    T_dark_avg.append(T_dark_vxb[i]*P_Vxb[i])
    T_b_avg.append(T_21_xb[i]*P_Vxb[i])
    x_frac_avg.append(x_frac_vxb[i]*P_Vxb[i])
    T_spin_avg.append(T_spin_vxb[i]*P_Vxb[i])
    #T_b_avg.append(0.1*P_Vxb[i])
    
T_gas_avg = sum(T_gas_avg)/(1.0*sum(P_Vxb))
T_dark_avg = sum(T_dark_avg)/(1.0*sum(P_Vxb))
T_21_avg = sum(T_b_avg)/(1.0*sum(P_Vxb))
x_frac_avg = sum(x_frac_avg)/(sum(P_Vxb))
T_spin_avg = sum(T_spin_avg)/(1.0*sum(P_Vxb))

Temp_data1 = np.array([z, T_gamma, T_gas_avg, T_21_avg, x_frac_avg, T_spin_avg])#, T_dark_avg, mag_field, mag_field0])
Temp_data = Temp_data1.T

'''
gamma_store_final = []
gamma_arr = np.array(gamma_store)
for i in range(0, len(z_store), 2):
    gamma_store_final.append(gamma_arr[i])

'''
'''
data_file = []
i = 0

for i in range (len(Temp_data)):
    if (i <= len(Temp_data)):
        data_file.append(Temp_data[i])
        i = i + 5
'''

#np.savetxt('new_temp-xfracs_PopIII_NL4800_PopII_withoutDM_withoutCR.txt', Temp_data) 
np.savetxt('temp-xfracs_PopIII_NL4800_PopII_withDM_mx0.1GeV_sigma2_q'+str(q)+'epsIII'+str(eps_popIII)+'epsII'+str(eps_popII)+'QpopII'+str(CR_heat_popII)+'data.txt', Temp_data)

'''
fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=[7,10], gridspec_kw={'hspace': 0, 'height_ratios': [1, 1]})
#fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=[8,10], gridspec_kw={'hspace': 0, 'height_ratios': [2, 1, 1]})

#plt.subplot(3,1,1)
axs[0].loglog(z, T_gas_avg, 'b-', label='$Standard$') #m_{\chi}=0.3\,GeV, \sigma_{45}=5, 
axs[0].loglog(z, T_gamma, 'k:')
axs[0].set_yscale('log')
#plt.xlim(10, 1010)
#plt.xticks('bottom')
#plt.ylim(1, 10000)
axs[0].set_ylabel("Temperature(K)", size=15)
#plt.legend(shadow=True)
axs[0].minorticks_on()
axs[0].grid(which='major', linestyle='-')
axs[0].grid(which='minor', linestyle='--')
plt.show()

#plt.subplot(3,1,2)
axs[1].loglog(z, x_frac_avg, 'b-')
axs[1].set_yscale('log')
#plt.xlim(10, 1010)
#plt.ylim(.00001, 1)
axs[1].set_ylabel("$x_e$", size=15)
axs[1].set_xlabel("Redshift(z)", size=15)
axs[1].minorticks_on()
axs[1].grid(which='major', linestyle='-')
axs[1].grid(which='minor', linestyle='--')
plt.show()
'''
'''
#E_input = (np.array(gamma_store_final))/(np.array(E_store))

plt.yscale('log')
plt.plot(z_cal, E_store, label='$E_{cr}$')
plt.plot(z_cal, gamma_store_final, label='$\Gamma_{cr}$')
'''
'''
plt.loglog(z, T_gamma, 'k:', label='$T_{\gamma}$')
plt.loglog(z, T_gas_avg, '--', label='$T_g$')
plt.loglog(z, T_spin_avg, '-.', label='$T_s$')

'''
#plt.yscale('log')
plt.plot(z, T_21_avg)#, label='PopIII+II')
plt.xlim(10,25)
#plt.loglog(z, x_frac_avg, label='xe') #%(np.abs(del_z)))

plt.legend(loc='upper right')
plt.xlabel('z')
plt.ylabel('$Temperature$ in K')
plt.show()

#plt.savefig('Tg_Ts_morePopIII_PopII.png', bbox_inches='tight', dpi=200)

end = time()
print("Time taken: %g sec" %(end - start))

'''
plt.loglog(z, x_c, 'k:', label='$x_c$,PopIII+II')
plt.loglog(z, x_alpha, '--', label='$x_a$,PopIII+II')
plt.loglog(z, x_tot, '-.', label='$x_tot$,PopIII+II')
plt.legend()
plt.xlim(10,100)
'''
