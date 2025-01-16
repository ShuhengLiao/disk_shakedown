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

    
def run(mesh,properties,T_load,omega,t_rise,t_heatdwell,t_fall,t_cooldwell,
        Displacement_BC = 'free-free',time_step=1.,time_step_coarse=10.):
    
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
    #     return project(sig_eq, P0).vector()[:].min()>=sig0.values()

    def shakedown_check(sig,sig0):
        s_ = dev(sig)
        sig_eq_ = sqrt(3/2.*inner(s_, s_))
        sig_eq = local_project(sig_eq_,W0)
        return (project(sig_eq/sig0,P0)).vector().max()
    
    def load(t):
        return  np.interp(t,[0,t_rise,t_rise+t_heatdwell,
                             t_rise+t_heatdwell+t_fall,
                             t_rise+t_heatdwell+t_fall+t_cooldwell],
                              [0.,1.,1.,0.,0.])
    
    ## DEFINE BCs
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

    def simulation_step_1(T_load,omega,
                          total_time = t_rise+t_heatdwell+t_fall+t_cooldwell,
                          step_change_time = t_rise+t_heatdwell+t_fall*2,
                          time_step = time_step,
                          time_step_coarse = time_step_coarse,
                          plastic_inverval = 1,
                          max_step = 100000,
                          newton_step = 1.,
                          tol = 1e-5):


        T_initial = Constant(0.)  # Initial temperature (e.g., room temperature)
        dt = Constant(time_step)
        v = TestFunction(U0)
        T_pre = interpolate(T_initial, U0) # Temp. at last time step
        T_old = interpolate(T_initial, U0) # ** Temp. at last mechanical step **
        T_crt = TrialFunction(U0)
        F_thermal = (rho*cp*(T_crt-T_pre)/dt)*v*x[0]*dx + kappa*dot(grad(T_crt),grad(v))*x[0]*dx
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

        for i in range(max_step):
            dt.assign(time_step)
            current_time += time_step
            print("Simulation time:",current_time)
            T_R.assign(T_load * load(current_time))
            solve(a_thermal == L_thermal, T_crt, Thermal_BC)
            T_pre.assign(T_crt)

            if (i+1)% plastic_inverval == 0:

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
                    Du.assign(Du+du*newton_step)
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

            if current_time > step_change_time - 1e-8:
                time_step = time_step_coarse

            if current_time > total_time - 1e-8:
                return sig


    def simulation_step_2(T_load, sig_res,
                          total_time = t_rise+t_heatdwell+t_fall*5,
                          time_step = time_step,
                          mech_inverval = 1,
                          max_step = 100000):

        T_initial = Constant(0.)  # Initial temperature (e.g., room temperature)
        dt = Constant(time_step)
        v = TestFunction(U0)
        T_pre = interpolate(T_initial, U0) # Temp. at last time step
        T_crt = TrialFunction(U0)
        F_thermal = (rho*cp*(T_crt-T_pre)/dt)*v*x[0]*dx + kappa*dot(grad(T_crt),grad(v))*x[0]*dx
        a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

        def sigma(v, dT):
            return (lmbda*tr(eps(v))- alpha_V*(3*lmbda+2*mu)*dT)*Identity(3) + 2.0*mu*eps(v)

        dT = Function(U0)
        v_ = TrialFunction(U)
        u_ = TestFunction(U)
        Wint = inner(sigma(v_, dT), eps(u_))*x[0]*dx
        a_m, L_m = lhs(Wint),rhs(Wint)

        current_time = 0

        T_crt = Function(U0,name="Temperature")
        u = Function(U, name="Total displacement")

        SD_flag = []
        for i in range(max_step):
            dt.assign(time_step)
            current_time += time_step
            print("Simulation time:",current_time)
            T_R.assign(T_load*load(current_time))
            solve(a_thermal == L_thermal, T_crt, Thermal_BC)
            T_pre.assign(T_crt)

            if (i+1)% mech_inverval == 0:
                dT.assign(T_crt-interpolate(T_initial, U0))
                solve(a_m == L_m, u, Mech_BC)
                sig = sigma(u,dT)
                SD_flag.append(shakedown_check(as_3D_tensor(sig_res)+sig,sig0))
                print(shakedown_check(as_3D_tensor(sig_res)+sig,sig0))

            if current_time > total_time - 1e-8:
                SD_flag = np.array(SD_flag)
                return SD_flag
            
    
    sig_res = simulation_step_1(T_load,omega)
    sd_flag = simulation_step_2(T_load, sig_res)
    return sd_flag



if __name__ == "__main__":
    
    mesh_file = "../mesh/disk.inp" # .inp or .xdmf

    comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]

    omega = 3800. # omega: rad/s
    T_load = 300. # T
    t_rise = 1. # time to heat to the max temp.
    t_heatdwell = 20. # heating dwell time
    t_fall = 3. # time to cool to the ambient temp
    t_cooldwell = 600. # cooling dwell time

    time_step = 1. # time step size
    time_step_coarse = 10. # use a coarser time step at during cooling dwell
    plastic_inverval = 1 # elasto-plastic simulation every N thermal time step

    Displacement_BC = 'fix-free' # fix-free or free-free
    
    mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)

    properties = get_properties(mesh,comp_list)
    
    sd_flag =run(mesh,properties,T_load,omega,t_rise,t_heatdwell,t_fall,t_cooldwell,
                 Displacement_BC,time_step,time_step_coarse)
    
    print(sd_flag)
    if sd_flag.max()>1.:
        print('Not Shakedown!')
    else:
        print('Shakedown!')
    
 