import numpy as np
from scipy.optimize import fsolve, minimize
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

class SCHR:
    def __init__(self,phis,alfa,x,a=1,b=1,c=1,d=1,phi0=0,l0s=0,l0h=0):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.phi0=phi0
        self.l0s = l0s
        self.l0h = l0h
        self.phis = phis
        self.alfa = alfa
        self.x = x
    
    def geometry_init(self):
        self.phih = self.alfa-self.phi0-self.phis
        xs = sp.sqrt(self.a**2+self.b**2-2*self.a*self.b*sp.cos(self.phis))+0.2
        xh = sp.sqrt(self.c**2+self.d**2-2*self.c*self.d*sp.cos(self.phih))+0.2
        xs_np = sp.lambdify([self.phis,self.alfa],xs, 'numpy')
        xh_np = sp.lambdify([self.phis,self.alfa],xh, 'numpy')
        return xs, xh, xs_np, xh_np
    
    def nonlinear_spring_init(self,pomery,koefs,koefh,akt_f = True,a_fun_coef=1000):
        self.pomery = pomery
        self.koefs = koefs
        self.koefh = koefh
        
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
    
    def thelen_03_tendon(self,pomery,koefs,koefh):
        self.pomery = pomery
        self.koefs = koefs
        self.koefh = koefh
        kh = 1
        ks = kh*self.pomery
        fttoe = 0.33
        kttoes=self.koefs
        kttoeh=self.koefh
        epst0=0.04
        epsttoe =  .99*epst0*np.e**3/(1.66*np.e**3 - .67)
        # Thorax-scapula
        epsts = (self.x-self.l0s)/self.l0s
        Fs = (fttoe/(sp.exp(kttoes)-1)*(sp.exp(kttoes*epsts/epsttoe)-1))*ks
        
        # Scapula-humerus
        epsth = (self.x-self.l0h)/self.l0h
        Fh = (fttoe/(sp.exp(kttoeh)-1)*(sp.exp(kttoeh*epsth/epsttoe)-1))*kh
        
        Us = sp.integrate(Fs,(self.x,self.l0s,self.x))
        Uh = sp.integrate(Fh,(self.x,self.l0h,self.x))
        
        return Fs,Fh,Us,Uh
    
    def thelen_03_muscle_pas_force(self,pomery,koefs,koefh,akt_f = True,a_fun_coef=1000):
        self.pomery = pomery
        self.koefs = koefs
        self.koefh = koefh
        kh = 1
        ks = kh*self.pomery
        epsm0=0.6
        
        if akt_f:
            a_funs = 1/(1+sp.exp(a_fun_coef*(self.l0s-self.x)))
            a_funh = 1/(1+sp.exp(a_fun_coef*(self.l0h-self.x)))
            lms = self.x/self.l0s
            Fs = ((sp.exp(self.koefs*(lms-1)/epsm0)-1)/(sp.exp(self.koefs)-1))*ks

            lmh = self.x/self.l0h
            Fh = ((sp.exp(self.koefh*(lmh-1)/epsm0)-1)/(sp.exp(self.koefh)-1))*kh

            Us_wo = sp.integrate(Fs,(self.x,self.l0s,self.x))
            Uh_wo = sp.integrate(Fh,(self.x,self.l0h,self.x))
            Us = a_funs*Us_wo
            Uh = a_funh*Uh_wo
        
        else:
            lms = self.x/self.l0s
            Fs = ((sp.exp(self.koefs*(lms-1)/epsm0)-1)/(sp.exp(self.koefs)-1))*ks

            lmh = self.x/self.l0h
            Fh = ((sp.exp(self.koefh*(lmh-1)/epsm0)-1)/(sp.exp(self.koefh)-1))*kh

            Us = sp.integrate(Fs,(self.x,self.l0s,self.x))
            Uh = sp.integrate(Fh,(self.x,self.l0h,self.x))
        
        return Fs,Fh,Us,Uh
        
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
        
    def graphs_scapula_position(self,U_C_np,dU_C_np,alfa_start=0.1,alfa_end=140,koefs=3,koefh=3,init_root=0.1):
        self.init_root = init_root
        
        N = 100
        NG = 10
        alfa_start = alfa_start*np.pi/180
        alfa_end = alfa_end*np.pi/180
        alfavec = np.linspace(alfa_start,alfa_end,N)
        phisvec_root = np.zeros((N,NG))
        phisvec_minimize = np.zeros((N,NG))
        lsvec = np.zeros((N,NG))
        min_pomer = 0.1
        max_pomer = 10
        pomery = np.linspace(min_pomer,max_pomer,NG)

        bnds = (0, np.pi)
        for j in range(NG):
            root = self.init_root
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
        
    def scapula_position_argmin(self,U_C_np,koefs,koefh,N=100):
        ## U_C_np = [x=phis, alfa, pomery, koefs, koefh]
        alfa_start = 0.1
        alfa_end = 140*np.pi/180
        alfavec = np.linspace(alfa_start,alfa_end,N)
        U_C_min = np.zeros(N)
        NG = 10
        min_pomer = 0.1
        max_pomer = 5
        pomery = np.linspace(min_pomer,max_pomer,NG)
        
        for j, pomer in enumerate(pomery):
            for i, alfa in enumerate(alfavec):
                phis = np.linspace(0,alfa,N)
                U_C_min[i] = phis[U_C_np(phis, alfa, pomer, koefs, koefh).argmin()]
                
            plt.plot(alfavec*180/np.pi,U_C_min*180/np.pi,label='ks/kh = %s' % round(pomer,2))
        plt.legend()
        plt.show()
        # plt.plot(alfavec*180/np.pi,U_C_min*180/np.pi)
            
        
        
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
                        axs[j+len(koefs)*i,a].set_yscale('log')

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