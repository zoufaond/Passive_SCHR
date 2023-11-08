import numpy as np
from scipy.optimize import fsolve, minimize
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

class SCHR:
    def __init__(self,phis,alfa,pomery,x,koefs,koefh,a=1,b=1,c=1,d=1,phi0=0,l0s=0,l0h=0):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.phi0=phi0
        self.koefs = koefs
        self.koefh = koefh
        self.l0s = l0s
        self.l0h = l0h
        self.phis = phis
        self.alfa = alfa
        self.pomery = pomery
        self.x = x
    
    def geometry_init(self):
        phih = self.alfa-self.phi0-self.phis
        xs = sp.sqrt(self.a**2+self.b**2-2*self.a*self.b*sp.cos(self.phis))
        xh = sp.sqrt(self.c**2+self.d**2-2*self.c*self.d*sp.cos(phih))
        return xs, xh
    
    def nonlinear_spring_init(self,akt_f = True,a_fun_coef=1000):
        if akt_f:
            a_funs = 1/(1+sp.exp(a_fun_coef*(self.l0s-self.x)))
            a_funh = 1/(1+sp.exp(a_fun_coef*(self.l0h-self.x)))
            kh = 1
            ks = kh*self.pomery
            Fs = ks*(self.x-self.l0s)**(self.koefs)
            Fh = kh*(self.x-self.l0h)**(self.koefh)
            Us_wo = sp.integrate(Fs,(self.x,self.l0s,self.x))
            Uh_wo = sp.integrate(Fh,(self.x,self.l0h,self.x))
            Us = a_funs*Us_wo
            Uh = a_funh*Uh_wo
        else:
            kh = 1
            ks = kh*self.pomery
            Fs = ks*(self.x-self.l0s)**(self.koefs)
            Fh = kh*(self.x-self.l0h)**(self.koefh)
            Us = sp.integrate(Fs,(self.x,self.l0s,self.x))
            Uh = sp.integrate(Fh,(self.x,self.l0h,self.x))
        return Fs,Fh, Us, Uh
    
    def potential_energy(self,Us,Uh,xs,xh,dd):
        Us_sub = Us.subs(self.x,xs)
        Uh_sub = Uh.subs(self.x,xh)
        U_celk = Us_sub+Uh_sub
        d_U_celk = sp.diff(U_celk,dd)
        return U_celk, d_U_celk
        
    def lambdified(self, U_c,d_U_c):
        Uc_np = sp.lambdify([self.phis,self.alfa,self.pomery,self.koefs,self.koefh],U_c, 'numpy')
        d_Uc_np = sp.lambdify([self.phis,self.alfa,self.pomery,self.koefs,self.koefh],d_U_c, 'numpy')
        return Uc_np, d_Uc_np
        
    def graphs_scapula_position(self,U_C_np,dU_C_np,alfa_start=0.1,alfa_end=140,koefs=2,koefh=3):
        N = 100
        NG = 10
        alfa_start = alfa_start*np.pi/180
        alfa_end = alfa_end*np.pi/180
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
                    return dU_C_np(x,alfavec[i],pomery[j],koefs,koefh)
                def fun_minimize(x):
                    return U_C_np(x,alfavec[i],pomery[j],koefs,koefh)

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
        fig.suptitle('Nahore - root, dole - minimize')  
        alfavec = alfavec*180/np.pi
        phisvec_root = phisvec_root*180/np.pi
        phisvec_minimize = phisvec_minimize*180/np.pi
        for i in range(NG):
            axs[0].plot(alfavec,phisvec_root[:,i],label='ks/kh = %s' % round(pomery[i],2))
            axs[1].plot(alfavec,phisvec_minimize[:,i],label='ks/kh = %s' % round(pomery[i],2))
        # axs[1,1].plot(phisvec*180/np.pi,Us)
        # axs[1,0].plot(phisvec*180/np.pi,Uh)
        axs[0].legend()
        axs[1].legend()
        plt.show()
        
    def graphs_potential_energy(self,U_C_np,alfa1=40,alfa2=70,alfa3=100,alfa4=130):
        alfa = np.array([alfa1,alfa2,alfa3,alfa4])*np.pi/180
        phis = np.linspace(0,alfa,100)
        koefs = np.array([1.5,2.5,3.5])
        koefh = np.array([1.5,2.5,3.5])
        N_koef = len(koefs)*len(koefh)
        N_alfa = len(alfa)
        NG = 10
        min_pomer = 0.1
        max_pomer = 5
        pomery = np.linspace(min_pomer,max_pomer,NG)
        l0s = 0
        l0h = 0

        fig, axs = plt.subplots(N_koef,N_alfa, figsize=(25, 30))
        # fig.suptitle('vlevo - root, vpravo - minimize')
        labels = ['1','2','3','4']
        # for i in range(len(pomery)):
        #     labels.append("ks/kh = %s " % round(pomery[i],2))
        for i in range(len(koefs)):
            for j in range(len(koefh)):
                for k in range(len(pomery)):
                    UC = U_C_np(phis,alfa,pomery[k],koefs[i],koefh[j])
                    for a in range(len(alfa)):
                        ## minimum UC
                        min_UC = UC[:,a].argmin()
                        axs[j+len(koefs)*i,a].plot(phis[min_UC,a]*180/np.pi,UC[min_UC,a],'*')
                        ## 

                        axs[j+len(koefs)*i,a].plot(phis[:,a]*180/np.pi,UC[:,a],label='ks/kh = %s' % round(pomery[k],2))
                        axs[j+len(koefs)*i,a].title.set_text('koefs = %s, koefh = %s ' % (koefs[i],koefh[j]))  

        # lines = []
        # labels = []
        # for axs in fig.axes:
        #     Line, Label = axs.get_legend_handles_labels()
        #     # print(Label)
        #     lines.extend(Line)
        #     labels.extend(Label)

        line, label = axs[0,0].get_legend_handles_labels()
        fig.legend(line, label, loc='upper left')
        fig.tight_layout()
        plt.show()