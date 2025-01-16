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

def run(mesh,properties,Displacement_BC = 'free-free'):
    # material properties
    rho,cp,kappa,E,sig0,nu,alpha_V = properties
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    Et = E/1e5  # tangent modulus - any better method for perfect plasticity?
    H = E*Et/(E-Et)  # hardening modulus
    
    
    # define function space
    x = SpatialCoordinate(mesh) # Coords
    U = VectorFunctionSpace(mesh, "CG", 1)
    We = VectorElement("Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme='default')
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme='default')
    W0 = FunctionSpace(mesh, W0e)
    P = VectorFunctionSpace(mesh, "DG", 1)
    P0 = FunctionSpace(mesh, "DG", 0)

    U0 = FunctionSpace(mesh, "CG", 1)
    metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)
    
    ## DEFINE BCs
    boundary_left,boundary_right,boundary_bot,_ = get_boundary(mesh)

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
        
    def collapse_check(sig,sig0):
        s_ = dev(as_3D_tensor(sig))
        sig_eq_ = sqrt(3/2.*inner(s_, s_))
        sig_eq = local_project(sig_eq_,W0)
        return project(sig_eq/sig0, P0).vector().min()>=1.


    def simulation_step_0(step0_iter_max = 1000,
                      omega_inc = 1000,
                      omega_tol = 1,
                      tol = 1e-6):
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

        def sigma(eps):
            return lmbda*tr(eps)*Identity(3) + 2*mu*eps

        omega = Constant(0.)
        def F_int(v,omega):
            return rho*omega**2*x[0]*v[0]

        ppos = lambda x: (x+abs(x))/2.
        def proj_sig(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
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

        def sigma_tang(e):
            N_elas = as_3D_tensor(n_elas)
            return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e)

        a_Newton = inner(eps(v_), sigma_tang(eps(u_)))*x[0]*dxm
        res = -inner(eps(u_), as_3D_tensor(sig))*x[0]*dxm + F_int(u_,omega)*x[0]*dxm

        step0_iter = 0
        omega_ctr = 0

        EP_Nitermax = 50

        while step0_iter < step0_iter_max:
            step0_iter += 1
            omega_ctr += omega_inc 
            omega.assign(Constant(omega_ctr))
            print("Trying Omega = ",omega_ctr)

            A, Res = assemble_system(a_Newton, res, Mech_BC)
            Du.interpolate(Constant((0, 0)))
            nRes0 = Res.norm("l2")
            nRes = nRes0
            print("Residual = ", nRes0)

            niter = 0
            while nRes/nRes0 > tol:
                solve(A, du.vector(), Res, "mumps")
                Du.assign(Du+du*1)
                deps = eps(Du)
                sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)
                local_project(sig_, W, sig)
                local_project(n_elas_, W, n_elas)
                local_project(beta_, W0, beta)
                A, Res = assemble_system(a_Newton, res, Mech_BC)
                nRes = Res.norm("l2")
                print("Residual = ", nRes)
                niter += 1
                if niter >= EP_Nitermax:
                    print ("Too many iterations, check the residual")
                    break

            if collapse_check(sig,sig0):
                sig.assign(sig_old)
                beta.assign(beta_old)
                n_elas.assign(n_elas_old)
                if omega_inc > omega_tol:
                    omega_ctr -= omega_inc
                    omega_inc *= 0.5
                else:
                    print("number of iters:", step0_iter)
                    print("max omega = ", omega_ctr)
                    return omega_ctr

            else:
                u.assign(u+Du)
                p.assign(p+local_project(dp_, W0))
                sig_old.assign(sig)
                beta_old.assign(beta)
                n_elas_old.assign(n_elas)

        return -1
    
    omega_max = simulation_step_0()
    return omega_max
    
if __name__ == "__main__":
    
    mesh_file = "../mesh/disk.inp" # .inp or .xdmf
    comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]
    
    Displacement_BC = 'fix-free' # fix-free or free-free
    
    mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)
    properties = get_properties(mesh,comp_list)
    
    omega_max = run(mesh,properties,Displacement_BC)