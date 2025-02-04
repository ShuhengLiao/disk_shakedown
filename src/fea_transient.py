from dolfin import *
import meshio
import pyvista as pv
import numpy as np
from utils import *
from materials import *
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
import sys
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

## see: https://turtlefsi.readthedocs.io/en/latest/known_issues.html
PETScOptions.set("mat_mumps_icntl_4", 1)
PETScOptions.set("mat_mumps_icntl_14", 400)

def run_simulation(mesh,properties,load,omega,t_list,
                   Displacement_BC = 'free-free',
                   plastic_inverval = 1,
                   output_inverval = 1,
                   saveVTK = False,
                   sol_folder = './sols'):
    
    # material properties
    rho,cp,kappa,E,sig0,nu,alpha_V = properties
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    Et = E/1e5  # tangent modulus - any better method for perfect plasticity?
    H = E*Et/(E-Et)  # hardening modulus
    
    # DEFINE THERMAL PROBLEM
    V = FunctionSpace(mesh, "P", 1)  # Temperature space
    v = TestFunction(V)
    x = SpatialCoordinate(mesh) # Coords
    T_initial = Constant(0.)
    T_pre = interpolate(T_initial, V) # Temp. at last time step
    T_old = interpolate(T_initial, V) # ** Temp. at last mechanical step **
    T_crt = TrialFunction(V)
    dt = Constant(1.)
    F_thermal = (rho*cp*(T_crt-T_pre)/dt)*v*x[0]*dx + kappa*dot(grad(T_crt),grad(v))*x[0]*dx
    a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)
    
    
    # DEFINE MECH PROBLEM
    U = VectorFunctionSpace(mesh, "CG", 1)
    We = VectorElement("Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme='default')
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme='default')
    W0 = FunctionSpace(mesh, W0e)

    metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    sig = Function(W)
    sig_old = Function(W)
    n_elas = Function(W)
    beta = Function(W0)
    p = Function(W0, name="Cumulative plastic strain")
    u = Function(U, name="Total displacement")
    du = Function(U, name="Iteration correction")
    Du = Function(U, name="Current increment")
    v_ = TrialFunction(U)
    u_ = TestFunction(U)

    def eps(v):
        return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                              [0, v[0]/x[0], 0],
                              [v[1].dx(0), 0, v[1].dx(1)]]))

    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha_V*(3*lmbda+2*mu)*dT)*Identity(3) + 2.0*mu*eps(v)


    def F_int(v):
        return rho*omega**2*x[0]*v[0]

    ppos = lambda x: (x+abs(x))/2.
    def proj_sig(v, dT, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(v,dT)
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

    def sigma_tang(v):
        N_elas = as_3D_tensor(n_elas)
        e = eps(v)
        return lmbda*tr(e)*Identity(3) + 2.0*mu*e- 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e) 


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

    a_Newton = inner(eps(v_), sigma_tang(u_))*x[0]*dxm
    res = -inner(eps(u_), as_3D_tensor(sig))*x[0]*dxm + F_int(u_)*x[0]*dxm
    
    
    ## DEFINE BCs
    boundary_left,boundary_right,boundary_bot,_ = get_boundary(mesh)

    T_L = Constant(0.)
    T_R = Constant(0.)
    T_bc_L = DirichletBC(V, T_L, boundary_left)
    T_bc_R = DirichletBC(V, T_R, boundary_right)
    Thermal_BC = [T_bc_L,T_bc_R]

    U_bc_B = DirichletBC(U.sub(1), 0., boundary_bot)
    U_bc_L = DirichletBC(U.sub(0), 0., boundary_left)
    if Displacement_BC == 'fix-free':
        Mech_BC = [U_bc_B,U_bc_L]
    else:
        Mech_BC = [U_bc_B]
    
    P0 = FunctionSpace(mesh, "DG", 0)
    p_avg = Function(P0,name="Plastic strain")
    T_crt = Function(V, name="Temperature")
    dT = Function(V)

    PEEQ = [p_avg.vector()]

    tol = 1e-5
    Nitermax = 50
    
    if saveVTK:
        T_vtk_file = File(sol_folder+'/T.pvd')
        u_vtk_file = File(sol_folder+'/u.pvd')
        p_vtk_file = File(sol_folder+'/p.pvd')

    for n in range(len(t_list)-1): 
        dt.assign(t_list[n+1]-t_list[n])
        T_R.assign(load(t_list[n+1]))
        solve(a_thermal == L_thermal, T_crt, Thermal_BC)
        T_pre.assign(T_crt)

        if (n+1)% plastic_inverval == 0:
            dT.assign(T_crt-T_old)

            sig_ = update_sig_thermal(dT, sig_old)
            local_project(sig_, W, sig)

            A, Res = assemble_system(a_Newton, res, Mech_BC)

            Du.interpolate(Constant((0, 0)))
            nRes0 = Res.norm("l2")
            print("Residual0:", nRes0)
            nRes = nRes0
            niter = 0
            while nRes/nRes0 > tol and niter < Nitermax:
                solve(A, du.vector(), Res, "mumps")
                Du.assign(Du+du*1)
                sig_, n_elas_, beta_, dp_ = proj_sig(Du, dT,  sig_old, p)
                local_project(sig_, W, sig)
                local_project(n_elas_, W, n_elas)
                local_project(beta_, W0, beta)
                A, Res = assemble_system(a_Newton, res, Mech_BC)
                nRes = Res.norm("l2")
                print("Residual:", nRes)
                niter += 1
                if niter >= Nitermax:
                    sys.exit(print ("Too many iterations"))

            u.assign(u+Du)
            p.assign(p+local_project(dp_, W0))
            p_avg.assign(project(p, P0))
            T_old.assign(T_crt)
            sig_old.assign(sig)

            PEEQ.append(p_avg.vector()) 
            
        if (n+1)% output_inverval == 0 and saveVTK:
            u_vtk_file << (u, t_list[n+1])
            p_vtk_file << (p_avg, t_list[n+1])
            T_vtk_file << (T_crt, t_list[n+1])
  
            
    return np.array(t_list),np.array(PEEQ)

def check_PEEQ(PEEQ,period,tol=2e-5):
    if (PEEQ[-1] - PEEQ[-(1+period)]).max() < tol:
        print ('max PEEQ the next to last cycle = {}, max PEEQ last cycle = {}, Shakedown!'.format(
            PEEQ[-(1+period)].max(),PEEQ[-1].max()))
        return 1
    else:
        ind = (PEEQ[-1] - PEEQ[-(1+period)]).argmax()
        print ('Critical Element: {}, PEEQ the next to last cycle = {}, PEEQ last cycle = {}, Not shakedown!'.format(ind,
            PEEQ[-(1+period),ind],PEEQ[-1,ind]))
        return 0


if __name__ == "__main__":
    
    mesh_file = "../mesh/disk.inp" # .inp or .xdmf

    comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]

    omega = 2800. # omega: rad/s
    T_load = 600. # T
    t_rise = 1. # time to heat to the max temp.
    t_heatdwell = 20. # heating dwell time
    t_fall = 3. # time to cool to the ambient temp
    t_cooldwell = 600. # cooling dwell time
    n_cyc = 30 # number of simulated cycles

    # #### define time intergration scheme: method 1
    # time_step = 1. # time step size
    # time_step_coarse = 20. # use a coarser time step at during cooling dwell
    # t_cycle = t_rise + t_heatdwell + t_fall + t_cooldwell
    # t_fine = t_rise + t_heatdwell + t_fall*2.
    # t_list = get_time_list(t_cycle,t_fine,time_step,time_step_coarse,n_cyc)


    ### define time intergration scheme: method 2
    time_intervals = [t_rise, t_heatdwell, t_fall, t_cooldwell]  # you can divided 1 cycle to multiple time intervals.
    # time_intervals = [t_rise*3, t_heatdwell-t_rise*2, t_fall, t_cooldwell]  # *** Prefered: instead of only using 1s for the rising time, I suggest to use also 1s for initial stage of heat dwell
    step_list = [1.,5.,1.,20.] # then you can define the time step at each time interval
    t_list = get_time_list_alt(time_intervals,step_list,n_cyc)


    plastic_interval = 1 # elasto-plastic simulation every N thermal time step
    saveVTK = True
    output_intervel = 1 # output vtk file                        

    Displacement_BC = 'fix-free' # fix-free or free-free
    
    mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)

    properties = get_properties(mesh,comp_list)
    load = get_loadfunc(T_load,t_rise,t_heatdwell,t_fall,t_cooldwell)


    
    t_list,PEEQ = run_simulation(mesh,properties,load,omega,t_list,Displacement_BC,plastic_interval,output_intervel,saveVTK)
    
    period = (len(PEEQ)-1)/n_cyc
    assert period-int(period) == 0
    SD_flag = check_PEEQ(PEEQ,int(period),tol=2e-5)