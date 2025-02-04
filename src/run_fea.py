from fea_transient import run_simulation
from utils import *
from materials import *
import numpy as np

mesh_file = "../mesh/disk_fine.inp" # .inp or .xdmf
mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)

comp_list = np.arange(100)
comp2prop = get_composition2property_from_csv('../material/disk_33_properties.csv')
properties = get_properties(mesh,comp_list,comp2prop)

# comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]
# properties = get_properties(mesh,comp_list)

Displacement_BC = 'fix-free' # fix-free or free-free

time_step = 1. # time step size
time_step_coarse = 20. # use a coarser time step at during cooling dwell
plastic_interval = 1 # elasto-plastic simulation every N thermal time step
saveVTK = True
output_intervel = 1 # output vtk file 


t_rise = 1. # time to heat to the max temp.
t_heatdwell = 20. # heating dwell time
t_fall = 3. # time to cool to the ambient temp
t_cooldwell = 600. # cooling dwell time
n_cyc = 30 # number of simulated cycles
t_cycle = t_rise + t_heatdwell + t_fall + t_cooldwell
t_fine = t_rise + t_heatdwell + t_fall*2.
t_list = get_time_list(t_cycle,t_fine,time_step,time_step_coarse,n_cyc)


                       
for omega in [2100,2700]:
	for T_load in [300,350]:
		sol_folder = './benchmark_fine/disk33_grid_1s_20s/sols_{}_{}'.format(omega,T_load)
		load = get_loadfunc(T_load,t_rise,t_heatdwell,t_fall,t_cooldwell)
		t_list,PEEQ = run_simulation(mesh,properties,load,omega,t_list,Displacement_BC,plastic_interval,output_intervel,saveVTK,sol_folder)