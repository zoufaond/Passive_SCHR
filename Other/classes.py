import numpy as np
from scipy.optimize import fsolve, minimize
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

class SCHR:
    def __init__(self,var, a=1,b=1,c=1,d=1,phi0=0,koefs=2,koefh=3,l0s=0,l0h=0):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.phi0=phi0
        self.koefs = koefs
        self.koefh = koefh
        self.l0s = l0s
        self.l0h = l0h
        self.var = var
        
    def zkouska(self):
        return self.var
    
    
    def geometry_init(self,phis,alfa):
        phih = alfa-self.phi0-phis
        xs = sp.sqrt(self.a**2+self.b**2-2*self.a*self.b*sp.cos(phis))
        xh = sp.sqrt(self.c**2+self.d**2-2*self.c*self.d*sp.cos(phih))
        return xs, xh
    
    def nonlinear_spring_init(self,pomery,x):
        kh = 1
        ks = kh*pomery
        Fs = ks*(x-self.l0s)**(self.koefs)
        Fh = kh*(x-self.l0h)**(self.koefh)
        Us = sp.integrate(Fs,(x,self.l0s,x))
        Uh = sp.integrate(Fh,(x,self.l0h,x))
        return Fs,Fh, Us, Uh
    
    def potential_energy(self,Fs,Fh,xs,xh,x,dd):
        Us_sub = Fs.subs(x,xs)
        Uh_sub = Fh.subs(x,xh)
        U_celk = Us_sub+Uh_sub
        d_U_celk = sp.diff(U_celk,dd)
        return U_celk, d_U_celk
        
    def lambdified(self, U_c,d_U_c,phis,alfa,pomery):
        Uc_np = sp.lambdify([phis,alfa,pomery],U_c, 'numpy')
        d_Uc_np = sp.lambdify([phis,alfa,pomery],d_U_c, 'numpy')
        return Uc_np, d_Uc_np
        
    def graphs_scapula_position(self,U_C_np,dU_C_np):
        N = 100
        NG = 10
        alfa_start = 0.1*np.pi/180
        alfa_end = 140*np.pi/180
        alfavec = np.linspace(alfa_start,alfa_end,N)
        phisvec_root = np.zeros((N,NG))
        phisvec_minimize = np.zeros((N,NG))
        lsvec = np.zeros((N,NG))
        min_pomer = 0.5
        max_pomer = 10
        pomery = np.linspace(min_pomer,max_pomer,NG)

        root = 0.1
        bnds = (0, np.pi)
        for j in range(NG):
            root = 0.8
            for i in range(N):

                def fun_root(x):
                    return dU_C_np(x,alfavec[i],pomery[j])
                def fun_minimize(x):
                    return U_C_np(x,alfavec[i],pomery[j])

                root = fsolve(fun_root, [root])
                mnmz = minimize(fun_minimize,root,bounds=[bnds])
                phisvec_root[i,j] = root
                phisvec_minimize[i,j] = mnmz.x
                
        # forces_eq = np.zeros(N)
        # k = 6
        # for i in range(N):
        #     forces_eq[i] = forces(alfavec[i],phisvec_root[i,k],koefs,koefh,pomery[k],l0ss,l0hh)

        alfa_rig = 12
        pomer_rig = 0.5
        phisvec = np.linspace(0,np.pi/5,100)
        # Uh = Uh_np(phisvec,alfa_rig*np.pi/180,pomer_rig,l0ss,l0hh,0)
        # Us = Us_np(phisvec,alfa_rig*np.pi/180,pomer_rig,l0ss,l0hh,0)

        fig, axs = plt.subplots(2, figsize=(15, 15))
        fig.suptitle('vlevo - root, vpravo - minimize')  
        alfavec = alfavec*180/np.pi
        phisvec_root = phisvec_root*180/np.pi
        phisvec_minimize = phisvec_minimize*180/np.pi
        for i in range(NG):
            axs[0].plot(alfavec,phisvec_root[:,i],label='ks/kh = %s' % round(pomery[i],2))
            axs[1].plot(alfavec,phisvec_minimize[:,i],label='ks/kh = %s' % round(pomery[i],2))
        # axs[1,1].plot(phisvec*180/np.pi,Us)
        # axs[1,0].plot(phisvec*180/np.pi,Uh)
        plt.show()
