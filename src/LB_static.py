from dolfin import *
from utils import *
from materials import *
import numpy as np
import meshio
import pyvista as pv
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
import sys
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

## see: https://turtlefsi.readthedocs.io/en/latest/known_issues.html
PETScOptions.set("mat_mumps_icntl_4", 1)
PETScOptions.set("mat_mumps_icntl_14", 400)

def run(mesh,properties,T_max,omega,Displacement_BC = 'free-free'):
    
    # material properties
    rho,cp,kappa,E,sig0,nu,alpha_V = properties
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    Et = E/1e5  # tangent modulus - any better method for perfect plasticity?
    H = E*Et/(E-Et)  # hardening modulus
    
    # define function space
    U = VectorFunctionSpace(mesh, "CG", 1)
    We = VectorElement("Quadrature", mesh.ufl_cell(), degree=2, dim=4, quad_scheme='default')
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=2, quad_scheme='default')
    W0 = FunctionSpace(mesh, W0e)
    P = VectorFunctionSpace(mesh, "DG", 1)
    P0 = FunctionSpace(mesh, "DG", 0)

    U0 = FunctionSpace(mesh, "CG", 1)
    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    x = SpatialCoordinate(mesh)
    boundary_left,boundary_right,boundary_bot,_ = get_boundary(mesh)

    T_L = Constant(0.)
    T_R = Constant(0.)
    T_bc_L = DirichletBC(U0, T_L, boundary_left)
    T_bc_R = DirichletBC(U0, T_R, boundary_right)
    Thermal_BC = [T_bc_L,T_bc_R]

    U_bc_B = DirichletBC(U.sub(1), 0., boundary_bot)
    U_bc_L = DirichletBC(U.sub(0), 0., boundary_left)
    if Displacement_BC == 'fix-free':
        Mech_BC = [U_bc_B,U_bc_L]
    else:
        Mech_BC = [U_bc_B]
        
        
    def eps(v):
        return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                              [0, v[0]/x[0], 0],
                              [v[1].dx(0), 0, v[1].dx(1)]]))

    def local_project(v, V, u=None):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dxm
        b_proj = inner(v, v_)*dxm
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return
        
        
    def simulation_step_1(T_load):
        T_initial = Constant(0.)  # Initial temperature (e.g., room temperature)
        v = TestFunction(U0)
        T_pre = interpolate(T_initial, U0) # Temp. at last time step
        T_crt = TrialFunction(U0)
        F_thermal = kappa*dot(grad(T_crt),grad(v))*x[0]*dx
        a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

        def sigma(v, dT):
            return (lmbda*tr(eps(v))- alpha_V*(3*lmbda+2*mu)*dT)*Identity(3) + 2.0*mu*eps(v)

        dT = Function(U0)
        v_ = TrialFunction(U)
        u_ = TestFunction(U)
        Wint = inner(sigma(v_, dT), eps(u_))*x[0]*dx
        a_m, L_m = lhs(Wint),rhs(Wint)

        T_crt = Function(U0,name="Temperature")
        u = Function(U, name="Total displacement")

        T_R.assign(T_load)
        solve(a_thermal == L_thermal, T_crt, Thermal_BC)

        dT.assign(T_crt-interpolate(T_initial, U0))
        solve(a_m == L_m, u, Mech_BC)
        sig = sigma(u,dT)

        return sig


    def simulation_step_2(T_max,omega,T_tol = 5,tol = 1e-5):

        T_initial = Constant(0.)  # Initial temperature (e.g., room temperature)
        v = TestFunction(U0)
        T_old = interpolate(T_initial, U0) # ** Temp. at last mechanical step **
        T_crt = TrialFunction(U0)
        F_thermal = kappa*dot(grad(T_crt),grad(v))*x[0]*dx
        a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

        sig = Function(W)
        sig_old = Function(W)
        n_elas = Function(W)
        n_elas_old = Function(W)
        beta = Function(W0)
        beta_old = Function(W0) 

        p = Function(W0, name="Cumulative plastic strain")
        u = Function(U, name="Total displacement")
        du = Function(U, name="Iteration correction")
        Du = Function(U, name="Current increment")
        v_ = TrialFunction(U)
        u_ = TestFunction(U)

        omega = Constant(omega)

        def sigma(eps_el):
            return lmbda*tr(eps_el)*Identity(3) + 2*mu*eps_el

        def F_int(v):
            return rho*omega**2*x[0]*v[0]

        def thermal_strain(dT):
            return alpha_V*dT*Identity(3)

        ppos = lambda x: (x+abs(x))/2.
        def proj_sig(deps, dT, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            d_eps_T = thermal_strain(dT)
            sig_elas = sig_n + sigma(deps-d_eps_T)
            s = dev(sig_elas)
            sig_eq = sqrt(3/2.*inner(s, s))
            f_elas = sig_eq - sig0 - H*old_p
            dp = ppos(f_elas)/(3*mu+H)
            n_elas = s/sig_eq*ppos(f_elas)/f_elas
            beta = 3*mu*dp/sig_eq
            new_sig = sig_elas-beta*s
            return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 2]]), \
                   as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 2]]), \
                   beta, dp

        def update_sig_thermal(dT, old_sig):
            sig_n = as_3D_tensor(old_sig)
            new_sig = sig_n - alpha_V*(3*lmbda+2*mu)*dT*Identity(3)
            return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 2]])

        def sigma_tang(e):
            N_elas = as_3D_tensor(n_elas)
            return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e)

        a_Newton = inner(eps(v_), sigma_tang(eps(u_)))*x[0]*dxm
        res = -inner(eps(u_), as_3D_tensor(sig))*x[0]*dxm + F_int(u_)*x[0]*dxm

        current_time = 0

        T_crt = Function(U0,name="Temperature")
        u = Function(U, name="Total displacement")
        dT = Function(U0)
        p_avg = Function(P0, name="Plastic strain")

        T_tol = 5.
        T_list = np.linspace(0,T_max,int(T_max/T_tol)+1)
        sig_list = []

        for T_load in T_list:
            T_R.assign(T_load)
            solve(a_thermal == L_thermal, T_crt, Thermal_BC)


            dT.assign(T_crt-T_old)

            sig_ = update_sig_thermal(dT, sig_old)
            local_project(sig_, W, sig)

            A, Res = assemble_system(a_Newton, res, Mech_BC)
            Du.interpolate(Constant((0, 0)))

            nRes0 = Res.norm("l2")
            print("    Residual0:", nRes0)
            niter = 0

            nRes = nRes0
            Nitermax = 20

            while nRes/nRes0 > tol:
                solve(A, du.vector(), Res, "mumps")
                Du.assign(Du+du)
                deps = eps(Du)
                sig_, n_elas_, beta_, dp_ = proj_sig(deps, dT,  sig_old, p)
                local_project(sig_, W, sig)
                local_project(n_elas_, W, n_elas)
                local_project(beta_, W0, beta)
                A, Res = assemble_system(a_Newton, res, Mech_BC)
                nRes = Res.norm("l2")
                print("    Residual:", nRes)
                niter += 1
                if niter >= Nitermax:
                    sys.exit(print ("    Too many iterations"))

            u.assign(u+Du)
            p.assign(p+local_project(dp_, W0))
            p_avg.assign(project(p, P0))

            T_old.assign(T_crt)
            sig_old.assign(sig)
            sig_list.append(sig)

        return T_list,sig_list
        
        
    def shakedown_check(sig,T,sig_e,T_ref):
        s = as_3D_tensor(sig) - sig_e*T/T_ref
        s_ = dev(s)
        sig_eq_ = sqrt(3/2.*inner(s_, s_))
        sig_eq = local_project(sig_eq_,W0)
        return (project(sig_eq/sig0,P0)).vector().max()
        
    sig_e = simulation_step_1(T_max)
    T_list,sig_list = simulation_step_2(T_max,omega,T_tol = 5,tol = 1e-5)
    for i in range(len(sig_list)):
        T = T_list[i]
        sig = sig_list[i]
        sd_flag = shakedown_check(sig,T,sig_e,T_max)
        if sd_flag > 1.001:
            break
    T_sd = (T_list[i]+T_list[i-1])/2.
    return T_sd
        
        
        
if __name__ == "__main__":

    mesh_file = "../mesh/disk.inp" # .inp or .xdmf

    comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]

    Displacement_BC = 'fix-free' # fix-free or free-free

    mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)

    properties = get_properties(mesh,comp_list)

    T_max = 2000.
    omega = 1000.

    T_sd = run(mesh,properties,T_max,omega,Displacement_BC)

    print("omega = {}, T_SD = {}".format(omega,T_sd))