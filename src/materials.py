from dolfin import *
import numpy as np


def composition2property_IN718_INVAR(comp_list):
    k1 = 21.977058 # Thermal conductivity
    rho1 = 8110.33 # Density
    cp1 = 511.261005 # Specific heat
    E1 = 203947470000 # Young's modules
    yield1 = 885668975.9  # yield strength
    nu1 = .3 # poisson ratio
    alphaV1 = 1.42e-5 #CTE

    k2 = 22.14
    rho2 = 8149.9
    cp2 = 529.8
    E2 = 155000000000.
    yield2 = 715500000. 
    nu2 = .2892 
    alphaV2 = 1e-5
    
    kappa = comp_list*k1 + (1-comp_list)*k2
    rho = comp_list*rho1 + (1-comp_list)*rho2
    cp = comp_list*cp1 + (1-comp_list)*cp2
    E = comp_list*E1 + (1-comp_list)*E2
    sig0 = comp_list*yield1 + (1-comp_list)*yield2
    nu = comp_list*nu1 + (1-comp_list)*nu2
    alphaV = comp_list*alphaV1 + (1-comp_list)*alphaV2
    return rho,cp,kappa,E,sig0,nu,alphaV


def get_composition2property_from_csv(filename):
    csvdata = np.loadtxt(filename,delimiter=',',skiprows=1)
    def comp2prop(comp_list):
        comp_values = csvdata[:,0]

        E_values = csvdata[:,1]
        nu_values = csvdata[:,2]
        sig0_values = csvdata[:,3]
        alphaV_values = csvdata[:,4]
        kappa_values = csvdata[:,5]
        rho_values = csvdata[:,6]
        cp_values = csvdata[:,7]

        rho = np.interp(comp_list,comp_values,rho_values)
        kappa = np.interp(comp_list,comp_values,kappa_values)
        cp = np.interp(comp_list,comp_values,cp_values)
        E = np.interp(comp_list,comp_values,E_values)
        sig0 = np.interp(comp_list,comp_values,sig0_values)
        nu = np.interp(comp_list,comp_values,nu_values)
        alphaV = np.interp(comp_list,comp_values,alphaV_values)
        return rho,cp,kappa,E,sig0,nu,alphaV
    return comp2prop


def get_properties(mesh,comp_list,comp2prop=composition2property_IN718_INVAR):
    n_secs = len(comp_list)
    r1 = mesh.coordinates()[:,0].min()
    r2 = mesh.coordinates()[:,0].max()
    l = (r2-r1)/n_secs
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    for i in range(n_secs):
        class material(SubDomain):
            def inside (self,x, on_boundary):
                return (x[0]>=r1+i*l)
        material().mark(subdomains,i)
        
    kappa = Function(FunctionSpace(mesh, "DG", 0)) 
    rho = Function(FunctionSpace(mesh, "DG", 0)) 
    cp = Function(FunctionSpace(mesh, "DG", 0)) 
    E = Function(FunctionSpace(mesh, "DG", 0)) 
    sig0 = Function(FunctionSpace(mesh, "DG", 0)) 
    nu = Function(FunctionSpace(mesh, "DG", 0)) 
    alpha_V = Function(FunctionSpace(mesh, "DG", 0)) 
    
    temp = np.asarray(subdomains.array(), dtype=np.int32)

    rho_list,cp_list,kappa_list,E_list,sig0_list,nu_list,alphaV_list = comp2prop(comp_list)
    
#     kappa.vector()[:] = np.choose(temp, kappa_list)
#     rho.vector()[:] = np.choose(temp, rho_list)
#     cp.vector()[:] = np.choose(temp, cp_list)
#     E.vector()[:] = np.choose(temp, E_list)
#     sig0.vector()[:] = np.choose(temp, sig0_list)
#     nu.vector()[:] = np.choose(temp, nu_list)
#     alpha_V.vector()[:] = np.choose(temp, alphaV_list)
    
    kappa.vector()[:] = np.array(kappa_list[temp])
    rho.vector()[:] = np.array(rho_list[temp])
    cp.vector()[:] = np.array(cp_list[temp])
    E.vector()[:] = np.array(E_list[temp])
    sig0.vector()[:] = np.array(sig0_list[temp])
    nu.vector()[:] = np.array(nu_list[temp])
    alpha_V.vector()[:] = np.array(alphaV_list[temp])
    
    return rho,cp,kappa,E,sig0,nu,alpha_V
    
    
    
    
    