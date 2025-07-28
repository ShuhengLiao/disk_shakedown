from dolfin import *
import meshio
import pyvista as pv
import numpy as np

def read_mesh(mesh_file,get_pvmesh=False):
    if mesh_file[-4:] == '.inp':
        mesh = meshio.read(mesh_file)
        mesh_file = mesh_file.replace(".inp",".xdmf")
        mesh.write(mesh_file)
        
    mesh = Mesh()
    with XDMFFile(mesh_file) as  meshfile:
        meshfile.read(mesh)
        
    if get_pvmesh:
        pv_mesh = pv.wrap(meshio.read(mesh_file))
        return mesh,pv_mesh
    else:
        return mesh
    

def get_boundary(mesh):
    r1 = mesh.coordinates()[:,0].min()
    r2 = mesh.coordinates()[:,0].max()
    h1 = mesh.coordinates()[:,1].min()
    h2 = mesh.coordinates()[:,1].max()

    # BCs
    def bound_left(x):
        return near(x[0],r1)

    def bound_right(x):
        return near(x[0],r2)

    def bound_bot(x):
        return near(x[1],h1)
    
    def bound_top(x):
        return near(x[1],h2)
    
    return bound_left,bound_right,bound_bot,bound_right


def get_loadfunc(T_load,t_rise,t_heatdwell,t_fall,t_cooldwell,virtual_step=True):
    cycle_time = t_rise + t_heatdwell + t_fall + t_cooldwell
    if virtual_step:
        cycle_time = cycle_time + 1
    def load(t):
        t = t % cycle_time
        return  T_load * np.interp(t,[0,t_rise,t_rise+t_heatdwell,
                                      t_rise+t_heatdwell+t_fall,
                                      t_rise+t_heatdwell+t_fall+t_cooldwell],[0.,1.,1.,0.,0.])
    return load


def get_time_list(t_cycle,t_fine,time_step,time_step_coarse,n_cyc):
    t_list_fine = np.linspace(0,t_fine,int(t_fine/time_step)+1)
    t_list_coarse = np.linspace(t_fine,t_cycle,np.ceil((t_cycle-t_fine)/time_step_coarse).astype(int)+1)
    t_list_cycle = np.concatenate((t_list_fine,t_list_coarse[1:]))
    t_list_cycle = np.concatenate((t_list_cycle,t_list_cycle[-1:]+1)) # add 1s virtual time step
    t_cycle = t_list_cycle[-1]
    t_list = t_list_cycle
    cyc_list = np.ones_like(t_list)
    cyc_list[0] = 0
    for n in range(1,n_cyc):
        t_list = np.concatenate((t_list,t_list_cycle[1:]+n*t_cycle))
        cyc_list = np.concatenate((cyc_list,np.ones_like(t_list_cycle[1:])*(n+1)))
    return np.vstack((t_list,cyc_list))

def get_time_list_alt(time_intervals,step_list,n_cyc):
    t_start = 0.
    t_cycle = (np.array(time_intervals)).sum()
    t_list_cycle = np.array([0.])
    for t, step in zip(time_intervals,step_list):
        t_list_local = np.linspace(t_start,t_start+t,np.ceil(t/step).astype(int)+1)
        t_list_cycle = np.concatenate((t_list_cycle,t_list_local[1:]))
        t_start = t_start + t
    t_list_cycle = np.concatenate((t_list_cycle,t_list_cycle[-1:]+1)) # add 1s virtual time step
    t_cycle = t_list_cycle[-1]
    t_list = t_list_cycle
    cyc_list = np.ones_like(t_list)
    cyc_list[0] = 0
    for n in range(1,n_cyc):
        t_list = np.concatenate((t_list,t_list_cycle[1:]+n*t_cycle))
        cyc_list = np.concatenate((cyc_list,np.ones_like(t_list_cycle[1:])*(n+1)))
    return np.vstack((t_list,cyc_list))


    
def as_3D_tensor(X):
    return as_tensor([[X[0], 0, X[3]],
                      [0, X[1], 0],
                      [X[3], 0, X[2]]])

def cal_principal_strain(eps):
    eps_r  = eps[:,:,0]
    eps_th  = eps[:,:,1]
    eps_z  = eps[:,:,2]
    eps_rz  = eps[:,:,3]
    eps1 = (eps_r + eps_z)/2 + np.sqrt((eps_r-eps_z)**2/4.+eps_rz**2)
    eps2 = eps_th
    eps3 = (eps_r + eps_z)/2 - np.sqrt((eps_r-eps_z)**2/4.+eps_rz**2)
    eps_PI = np.concatenate((eps1[:,:,None],eps2[:,:,None],eps3[:,:,None]),axis=-1)
    return eps_PI

def max_principal(eps_PI):
    eps1 = eps_PI.max(axis=-1)
    return eps1

def max_shear(eps_PI):
    eps1 = eps_PI.max(axis=-1)
    eps3 = eps_PI.min(axis=-1)
    gamma = eps1-eps3
    return gamma

def Brown_Miller(eps_PI):
    eps1 = eps_PI.max(axis=-1)
    eps3 = eps_PI.min(axis=-1)
    gamma = eps1-eps3
    epsN = (eps1+eps3)/2.
    return gamma + epsN

def von_Mises(eps_PI):
    eps_VM = np.sqrt(2/9.*((eps_PI[:,:,0]-eps_PI[:,:,1])**2 + 
                           (eps_PI[:,:,1]-eps_PI[:,:,2])**2 +
                           (eps_PI[:,:,2]-eps_PI[:,:,0])**2))
    return eps_VM

def cal_strain_range(strain,period,n_cyc):
    strain_max = [strain[i*period:(i+1)*period+1].max(axis=0) for i in range(n_cyc)]
    strain_min = [strain[i*period:(i+1)*period+1].min(axis=0) for i in range(n_cyc)]
    strain_range = (np.array(strain_max)-np.array(strain_min))/2.
    return strain_range

def fatigue_criter_range(eps,n_cyc,criter='all'):
    assert int((eps.shape[0] - 1)/n_cyc) == (eps.shape[0] - 1)/n_cyc
    period = int((eps.shape[0] - 1)/n_cyc)
    eps_PI = cal_principal_strain(eps)
    if criter == 'max_princicpal':
        strain = max_principal(eps_PI)
        return cal_strain_range(strain,period,n_cyc)
    if criter == 'max_shear':
        strain = max_shear(eps_PI)
        return cal_strain_range(strain,period,n_cyc)
    if criter == 'Brown_Miller':
        strain = Brown_Miller(eps_PI)
        return cal_strain_range(strain,period,n_cyc)
    if criter == 'von_Mises':
        strain = von_Mises(eps_PI)
        return cal_strain_range(strain,period,n_cyc)
    if criter == 'all':
        strain1 = max_principal(eps_PI)
        strain2 = max_shear(eps_PI)
        strain3 = Brown_Miller(eps_PI)
        strain4 = von_Mises(eps_PI)
        return (cal_strain_range(strain1,period,n_cyc), 
                cal_strain_range(strain2,period,n_cyc), 
                cal_strain_range(strain3,period,n_cyc),
                cal_strain_range(strain4,period,n_cyc))
