import numpy as np
from scipy.optimize import fsolve, minimize
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

def T_x(x):
    trans_x = sp.Matrix([[1,0,x],
                         [0,1,0],
                         [0,0,1]])
    return trans_x

def T_y(y):
    trans_y = sp.Matrix([[1,0,0],
                         [0,1,y],
                         [0,0,1]])
    return trans_y

def R_phi(phi):
    rot_phi = sp.Matrix([[sp.cos(phi),-sp.sin(phi),0],
                         [sp.sin(phi), sp.cos(phi),0],
                         [0          ,0           ,1]])
    return rot_phi

def position(x,y):
    r = sp.Matrix([x,y,1])
    return r

class SCHR:
    def __init__(self,phis,alfa,x):
        self.phis = phis
        self.alfa = alfa
        self.x = x
        self.U_muscles = []
        self.F_muscles = []
        self.U_muscles_np = []
        
    def geometry(self,angle,x,y):
        pos = R_phi(angle)*r(x,y)
        return pos
    
    def add_muscle(self,l0=1,koef=5,epsm0=0.5,F_iso=1,thorax_ins=None,scap_ins=None,humer_ins=None,muscle_group='ThorScap'):
        if muscle_group=='ThorScap':
            A = position(thorax_ins[0],thorax_ins[1])
            B = R_phi(self.phis)*position(scap_ins[0],scap_ins[1])
            length = sp.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
            F,U = self.thelen_03_muscle_pas_force(l0,koef,epsm0,F_iso)
            U_len = U.subs(self.x,length)
            F_len = F.subs(self.x,length)
            F_len_cond = sp.Piecewise((0,length<l0),
                                      (F_len,True))
            U_len_cond = sp.Piecewise((0,length<l0),
                                      (U_len,True))
            
        elif muscle_group=='ScapHum':
            A = R_phi(self.phis)*position(scap_ins[0],scap_ins[1])
            B = R_phi(self.alfa)*position(humer_ins[0],humer_ins[1])
            length = sp.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
            F,U = self.thelen_03_muscle_pas_force(l0,koef,epsm0,F_iso)
            U_len = U.subs(self.x,length)
            F_len = F.subs(self.x,length)
            F_len_cond = sp.Piecewise((0,length<l0),
                                      (F_len,True))
            U_len_cond = sp.Piecewise((0,length<l0),
                                      (U_len,True))
            
        self.F_muscles.append(F_len_cond)
        self.U_muscles.append(U_len_cond)
        
    def thelen_03_muscle_pas_force(self,l0,koef,epsm0,F_iso,):
        eps = self.x/l0
        F = ((sp.exp(koef*(eps-1)/epsm0)-1)/(sp.exp(koef)-1))*F_iso
        U = sp.integrate(F,(self.x,l0,self.x))
        
        return F,U
       
    def potential_energy(self):
        U_celk = sum(self.U_muscles)
        U_celk_np = sp.lambdify((self.phis,self.alfa),U_celk)
        
        return U_celk_np