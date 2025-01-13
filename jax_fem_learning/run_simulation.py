# Import some useful modules.
import jax
import jax.numpy as np
import os
import glob

# Import JAX-FEM specific modules.
from jax_fem.generate_mesh import box_mesh_gmsh, Mesh, get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

# If you have multiple GPUs, set the one to use.
# device = ' cpu'

# if device == 'gpu':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define some useful directory paths.
crt_file_path = os.path.dirname(__file__)
# crt_file_path = os.getcwd()
data_dir = os.path.join(crt_file_path, 'data/test_simulation')
vtk_dir = os.path.join(data_dir, 'vtk')

# Define the thermal problem. 
# We solve the following equation (weak form of FEM):
# (rho*Cp/dt*(T_crt-T_old), Q) * dx + (k*T_crt_grad, Q_grad) * dx - (heat_flux, Q) * ds = 0
# where T_crt is the trial function, and Q is the test function.
class Thermal(Problem):
    # The function 'get_tensor_map' is responsible for the term (k*T_crt_grad, Q_grad) * dx. 
    # The function 'get_mass_map' is responsible for the term (rho*Cp/dt*(T_crt-T_old), Q) * dx. 
    # The function 'set_params' makes sure that the Neumann boundary conditions use the most 
    # updated T_old and laser information, i.e., positional information like laser_center (x_l, y_l) and 
    # 'switch' controlling ON/OFF of the laser.
    def get_tensor_map(self):
        def fn(u_grad, T_old):
            return k * u_grad
        return fn
 
    def get_mass_map(self):
        def T_map(T, x, T_old):
            return rho * Cp * (T - T_old) / dt
        return T_map

    def get_surface_maps(self):
        # Neumann BC values for thermal problem
        def thermal_neumann_top(u, point, old_T, laser_center, switch):
            # q is the heat flux into the domain
            d2 = (point[0] - laser_center[0]) ** 2 + (point[1] - laser_center[1]) ** 2
            q_laser = 2 * eta * P / (np.pi * rb ** 2) * np.exp(-2 * d2 / rb ** 2) * switch
            q_conv = h * (T0 - old_T[0])
            q_rad = SB_constant * emissivity * (T0 ** 4 - old_T[0] ** 4)
            q = q_conv + q_rad + q_laser
            return -np.array([q])
 
        def thermal_neumann_walls(u, point, old_T):
            # q is the heat flux into the domain.
            q_conv = h * (T0 - old_T[0])
            q_rad = SB_constant * emissivity * (T0 ** 4 - old_T[0] ** 4)
            q = q_conv + q_rad
            return -np.array([q])

        return [thermal_neumann_top, thermal_neumann_walls]

    def set_params(self, params):
        # Override base class method.
        sol_T_old, laser_center, switch = params

        # ?????
        sol_T_old_top = self.fes[0].convert_from_dof_to_face_quad(sol_T_old, self.boundary_inds_list[0])
        sol_T_old_walls = self.fes[0].convert_from_dof_to_face_quad(sol_T_old, self.boundary_inds_list[1])

        # (num_selected_faces, num_face_quads, dim)
        laser_center_quad = laser_center[None, None, :] * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))[:, :, None]
        # (num_selected_faces, num_face_quads)
        switch_quad = switch * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))

        self.internal_vars_surfaces = [[sol_T_old_top, laser_center_quad, switch_quad], [sol_T_old_walls]]
        self.internal_vars = [self.fes[0].convert_from_dof_to_quad(sol_T_old)]

# Define material properties. 
# We generally assume Inconel 625 material is used. 
# SI units are used throughout this example.
Cp = 670. # heat capacity (J/kg·K)
rho = 8440. # material density (kg/m^3)
k = 25.2 # thermal conductivity (W/m·K)
Tl = 1623 # liquidus temperature (K)
h = 100. # heat convection coefficien (W/m^2·K)
eta = 0.25 # absorption rate
SB_constant = 5.67e-8 # Stefan-Boltzmann constant (kg·s^-3·K^-4)
emissivity = 0.3 # emissivity
T0 = 293. # ambient temperature (K)
POWDER = 0 # powder flag
LIQUID = 1 # liquid flag
SOLID = 2 # solid flag

# Define laser properties.
vel = 0.875 # laser scanning velocity (m/s)
rb = 0.1e-3 # laser beam size (m)
P = 169. # laser power (W)

# Specify mesh-related information. 
# We use first-order hexahedron element for both T_crt and u_crt.
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Nx, Ny, Nz = 100, 100, 50  # 10 um per element
Lx, Ly, Lz = 2e-3, 2e-3, 1e-3 # domain size (m)
meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Define boundary locations.
def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def walls(point):
    left = np.isclose(point[0], 0., atol=1e-5)
    right = np.isclose(point[0], Lx, atol=1e-5)
    front = np.isclose(point[1], 0., atol=1e-5)
    back = np.isclose(point[1], Ly, atol=1e-5)
    return left | right | front | back

# Specify boundary conditions and problem definitions.
# Dirichlet BC values for thermal problem
def thermal_dirichlet_bottom(point):
    return T0

# Define thermal problem
dirichlet_bc_info_T = [[bottom], [0], [thermal_dirichlet_bottom]]
location_fns = [top, walls]

sol_T_old = T0 * np.ones((len(mesh.points), 1))
sol_T_old_for_u = np.array(sol_T_old)
problem_T = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info_T, location_fns=location_fns)

# Do some cleaning work.
files = glob.glob(os.path.join(vtk_dir, f'*'))
for f in files:
    os.remove(f)

# Save initial solution to local folder.
vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
save_sol(problem_T.fes[0], sol_T_old, vtk_path)

# Start the major loop of time iteration.
dt = 5 * 1e-6  # time increment ???
laser_on_t = 0.5 * Lx / vel  # time duration for scanning over half the length
simulation_t = 2 * laser_on_t  # total simulation time
ts = np.arange(0., simulation_t, dt)  # time steps

for i in range(len(ts[1:])):
    laser_center = np.array([Lx * 0.25 + vel * ts[i + 1], Ly / 2., Lz])  # laser center in the middle of the powder
    switch = np.where(ts[i + 1] < laser_on_t, 1., 0.) # Turn off the laser after some time
    print(f"\nStep {i + 1}, total step = {len(ts[1:])}, laser_x = {laser_center[0]}, Lx = {Lx}, laser ON = {ts[i + 1] < laser_on_t}")

    # Set parameter and solve for T
    problem_T.set_params([sol_T_old, laser_center, switch])
    sol_T_new_list = solver(problem_T)
    sol_T_new = sol_T_new_list[0]
    
    vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
    save_sol(problem_T.fes[0], sol_T_new, vtk_path)
    
    # Update T solution
    sol_T_old = sol_T_new
