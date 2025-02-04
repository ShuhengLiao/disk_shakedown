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


def get_loadfunc(T_load,t_rise,t_heatdwell,t_fall,t_cooldwell):
    cycle_time = t_rise + t_heatdwell + t_fall + t_cooldwell
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
    t_list = t_list_cycle
    for n in range(1,n_cyc):
        t_list = np.concatenate((t_list,t_list_cycle[1:]+n*t_cycle))
    return t_list

def get_time_list_alt(time_intervals,step_list,n_cyc):
    t_start = 0.
    t_cycle = (np.array(time_intervals)).sum()
    t_list_cycle = np.array([0.])
    for t, step in zip(time_intervals,step_list):
        t_list_local = np.linspace(t_start,t_start+t,np.ceil(t/step).astype(int)+1)
        t_list_cycle = np.concatenate((t_list_cycle,t_list_local[1:]))
        t_start = t_start + t
    t_list = t_list_cycle
    for n in range(1,n_cyc):
        t_list = np.concatenate((t_list,t_list_cycle[1:]+n*t_cycle))
    return t_list


    
def as_3D_tensor(X):
    return as_tensor([[X[0], 0, X[3]],
                      [0, X[1], 0],
                      [X[3], 0, X[2]]])
