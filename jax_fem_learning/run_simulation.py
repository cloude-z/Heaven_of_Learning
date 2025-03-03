# This script is directly fechted from the demos of the open-source JAX-FEM
# library and modified to fit the current project. For more information, please
# visit https://github.com/deepmodeling/jax-fem/tree/main/demos/thermal_mechanical.

# Import some useful modules.
import jax
import jax.numpy as np
import csv
import os
import glob
import time

# Import JAX-FEM specific modules.
from jax_fem.generate_mesh import box_mesh_gmsh, Mesh, get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

class Timer(object):
    def __init__(self):
        self.elapsed_time = 0.

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed_time += time.time() - self.start_time

    def reset(self):
        self.elapsed_time = 0.

    def get_time(self):
        return self.elapsed_time
    
    def print_time(self):
        print(f"Elapsed time: {self.elapsed_time/60:.2f} minutes")

## NEED TO SAVE ALL SETTINGS TO A CSV FILE, INCLDUING TIME ELAPSED
def write_to_csv(csv_file, header, data):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# # Start to record the time
# timer = Timer()
# timer.start()

# # If you have multiple GPUs, set the one to use.
# device = 'gpu'

# if device == 'gpu':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Define some useful directory paths.
# sim_dir = "/scratch/yzhang11/outputs/lpbf_simulations" ## CAN BE A FUNCTION ARGUMENT
# out_dir = os.path.join(sim_dir, '01_data')  ## HOW TO NAME THE SIMULATION FOLDER??
# vtk_dir = os.path.join(out_dir, 'vtk')
# csv_file = os.path.join(out_dir, 'settings.csv')

# # Do some cleaning work.
# files = glob.glob(os.path.join(vtk_dir, f'*'))

# for f in files:
#     os.remove(f)

# Define the thermal problem. 
# We solve the following equation (weak form of FEM):
# (rho * Cp / dt * (T_crt - T_old), Q) * dx + (k * T_crt_grad, Q_grad) * dx - (heat_flux, Q) * ds = 0
# where T_crt is the trial function, and Q is the test function.
class Thermal(Problem):
    # The function 'get_tensor_map' is responsible for the term (k * T_crt_grad, Q_grad) * dx. 
    # The function 'get_mass_map' is responsible for the term (rho * Cp / dt * (T_crt - T_old), Q) * dx. 
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

        sol_T_old_top = self.fes[0].convert_from_dof_to_face_quad(sol_T_old, self.boundary_inds_list[0])
        sol_T_old_walls = self.fes[0].convert_from_dof_to_face_quad(sol_T_old, self.boundary_inds_list[1])

        # (num_selected_faces, num_face_quads, dim)
        laser_center_quad = laser_center[None, None, :] * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))[:, :, None]
        # (num_selected_faces, num_face_quads)
        switch_quad = switch * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))

        self.internal_vars_surfaces = [[sol_T_old_top, laser_center_quad, switch_quad], [sol_T_old_walls]]
        self.internal_vars = [self.fes[0].convert_from_dof_to_quad(sol_T_old)]

# Define the mechanics problem. 
# Generally, JAX-FEM handles ((f(u_grad, alpha_1, alpha_2, ..., alpha_N)), v_grad) * dx 
# in the weak form. Here, we have f(u_grad, alpha_1, alpha_2, ..., alpha_N) = sigma_crt(u_crt_grad, epsilon_old, sigma_old, dT_crt, zeta_crt),
# where zeta_crt being the phase state variable. This is reflected by the function 'stress_return_map'.
class Plasticity(Problem):
    # We solve the following equation (weak form of FEM):
    # (sigma(u_grad), v_grad) * dx = 0
    # where u is the trial function, and v is the test function.
    def custom_init(self):
        # Initializing total strain, stress, temperature increment, and material phase.
        sigmas_old = np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, self.fes[0].vec, self.dim))
        epsilons_old = np.zeros_like(sigmas_old)
        dT = np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1))
        phase = np.ones_like(dT, dtype=np.int32) * POWDER
        self.internal_vars = [sigmas_old, epsilons_old, dT, phase]
    
    def get_tensor_map(self):
        _, stress_return_map, _ = self.get_maps()
        return stress_return_map

    def get_maps(self):
        def safe_sqrt(x):  
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x / y)

        def strain(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            return epsilon

        def stress(epsilon, E):
            # nu = 0.3  # Poisson ratio
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma

        def stress_return_maps(u_grad, sigma_old, epsilon_old, dT, phase):
            """
            The attribute self.internal_vars will be fed in this function as the
            four last arguments while performing the simulation.
            """
            # E0 = 70.e9  # Young's modulus ## REPLACE WITH THE RIGHT VALUE
            # sig0 = 250.e6  # yield stress  ## REPLACE WITH THE RIGHT VALUE
            # alpha_V0 = 1e-5  # thermal expansion coefficient ## REPLACE WITH THE RIGHT VALUE
            alpha_V = np.where(phase == SOLID, alpha_V0, 0.)  # alpha_V if solid, 0 if liquid
            E = np.where(phase == SOLID, E0, 1e-2 * E0)  # E_0 if solid, 1% * E_0 if liquid
            epsilon_inc_T = alpha_V * dT * np.eye(self.dim)
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc - epsilon_inc_T, E) + sigma_old
            s_dev = sigma_trial - 1. / self.dim * np.trace(sigma_trial) * np.eye(self.dim)
            s_norm = safe_sqrt(3. / 2. * np.sum(s_dev * s_dev))
            f_yield = s_norm - sig0  # perfect plasticity
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus * s_dev, s_norm)
            return sigma, (f_yield_plus, sigma[0, 0])

        stress_return_map = lambda *args: stress_return_maps(*args)[0]
        yield_val_fn = lambda *args: stress_return_maps(*args)[1]
        return strain, stress_return_map, yield_val_fn

    def vmap_stress_strain_fns(self):
        strain, stress_return_map, yield_val_fn = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        vmap_yield_val_fn = jax.vmap(jax.vmap(yield_val_fn))
        return vmap_strain, vmap_stress_return_map, vmap_yield_val_fn

    def update_stress_strain(self, sol, params):
        # Update sigmas and epsilons
        # Keep dT and phase unchanged
        # Output plastic_info for debugging purpose: we want to know if plastic deformation occurs, and the x-x direction stress
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        vmap_strain, vmap_stress_rm, vmap_yield_val_fn = self.vmap_stress_strain_fns()
        sigmas_old, epsilons_old, dT, phase = params
        sigmas_update = vmap_stress_rm(u_grads, sigmas_old, epsilons_old, dT, phase)
        epsilons_update = vmap_strain(u_grads)
        plastic_info = vmap_yield_val_fn(u_grads, sigmas_old, epsilons_old, dT, phase)
        return [sigmas_update, epsilons_update, dT, phase], plastic_info

    def update_dT_and_phase(self, dT, T, params):
        # Update dT and phase
        # Keep sigmas and epsilons unchanged
        sigmas, epsilons, _, phase = params
        dT_quad = self.fes[0].convert_from_dof_to_quad(dT)
        T_quad = self.fes[0].convert_from_dof_to_quad(T)
        powder_to_liquid = (phase == POWDER) & (T_quad > Tl)
        liquid_to_solid = (phase == LIQUID) & (T_quad < Tl)
        phase = phase.at[powder_to_liquid].set(LIQUID)
        phase = phase.at[liquid_to_solid].set(SOLID)
        return sigmas, epsilons, dT_quad, phase

    def set_params(self, params):
        # Override base class method.
        self.internal_vars = params

# ################################################################################
# # NEED TO REPLACE WITH THE RIGHT VALUES AND ADD DISTRIBUTIONS 
# ################################################################################
# # Define thermal properties in SI units.
# Cp = 670.  # specific heat capacity (J/kg·K)
# rho = 8440.  # material density (kg/m^3)
# k = 25.2  # thermal conductivity (W/m·K)
# Tl = 1623  # liquidus temperature (K)
# h = 100.  # heat convection coefficien (W/m^2·K)
# eta = 0.25  # absorption rate 0.37
# SB_constant = 5.67e-8  # Stefan-Boltzmann constant (kg·s^-3·K^-4)
# emissivity = 0.3  # emissivity
# T0 = 293.  # ambient temperature (K)
# POWDER = 0  # powder flag
# LIQUID = 1  # liquid flag
# SOLID = 2  # solid flag

# # Define mechanical properties in SI units.
# nu = 0.3  # Poisson ratio
# E0 = 70.e9  # Young's modulus (Pa)
# sig0 = 250.e6  # yield stress (Pa)
# alpha_V0 = 1e-5  # thermal expansion coefficient (m/m·K)

# # Define laser properties.
# vel = 0.875 # laser scanning velocity (m/s)
# rb = 0.1e-3 # laser beam size (m)
# P = 169. # laser power (W)

# # Specify mesh-related information. 
# # We use first-order hexahedron element for both T_crt and u_crt.
# ele_type = 'HEX8'
# cell_type = get_meshio_cell_type(ele_type)
# Nx, Ny, Nz = 400, 128, 64  # 12.5 μm in x-direction and 6.25 μm in y and z-directions
# Lx, Ly, Lz = 5.0e-3, 0.8e-3, 0.4e-3 # domain size (m)
# meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, out_dir)
# mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

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

# Dirichlet BC values for mechanical problem
def displacement_dirichlet_bottom(point):
    return 0.

# # Define thermal problem
# dirichlet_bc_info_T = [[bottom], [0], [thermal_dirichlet_bottom]]
# location_fns = [top, walls]

# sol_T_old = T0 * np.ones((len(mesh.points), 1))  # initial temperature field
# sol_T_old_for_u = np.array(sol_T_old)
# problem_T = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info_T, location_fns=location_fns)

# # Define mechanical problem
# dirichlet_bc_info_u = [[bottom] * 3, [0, 1, 2], [displacement_dirichlet_bottom] * 3]
# problem_u = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info_u)
# params_u = problem_u.internal_vars
# sol_u_list = [np.zeros((problem_u.fes[0].num_total_nodes, problem_u.fes[0].vec))]  # initial displacement field

# # Save initial solution to local folder.
# vtk_file_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
# save_sol(problem_T.fes[0], sol_T_old, vtk_file_path, point_infos=[('u', np.zeros((len(sol_T_old), 3)))], 
#                                                      cell_infos=[('f_plus', np.zeros(len(mesh.cells))),
#                                                                  ('strain_xx', np.zeros(len(mesh.cells))),
#                                                                  ('stress_xx', np.zeros(len(mesh.cells))),
#                                                                  ('phase', np.mean(params_u[-1][:, :, 0], axis=1))])

# ############################################################################################
# # NEED TO REVISE THE TIME ITERATION LOOP TO FIT THE CURRENT PROJECT
# ############################################################################################
# # Start the major loop of time iteration.
# dt = 1e-6  # time increment
# laser_on_t = 0.5 * Lx / vel  # time duration of laser scanning
# simulation_t = 3 * laser_on_t  # total simulation time until completely cooling down
# ts = np.arange(0., simulation_t, dt)  # time steps

# for i in range(len(ts[1:])):
#     laser_center = np.array([Lx * 0.25 + vel * ts[i + 1], Ly / 2., Lz])  # laser scanning starts at the point (0.25*Lx, 0.5*Ly, Lz)
#     switch = np.where(ts[i + 1] < laser_on_t, 1., 0.)  # turn off the laser after some time
#     print(f"\nStep {i + 1}, total step = {len(ts[1:])}, laser_x = {laser_center[0]}, Lx = {Lx}, laser ON = {ts[i + 1] < laser_on_t}")

#     # Set parameter and solve for T
#     problem_T.set_params([sol_T_old, laser_center, switch])
#     sol_T_new_list = solver(problem_T)
#     sol_T_new = sol_T_new_list[0]

#     # Since mechanics problem is more expensive to solve, we skip some steps of the thermal problem.
#     if (i + 1) % 10 == 0:
#         params_u = problem_u.update_dT_and_phase(sol_T_new - sol_T_old_for_u, sol_T_new, params_u)

#         # Set parameter and solve for u
#         problem_u.set_params(params_u)
#         sol_u_list = solver(problem_u, solver_options={'initial_guess': sol_u_list})

#         params_u, plastic_info = problem_u.update_stress_strain(sol_u_list[0], params_u) 

#         # Check if plastic deformation occurs (with f_yield_vals > 0.)
#         print(f"max f_plus = {np.max(plastic_info[0])}, max stress_xx = {np.max(plastic_info[1])}")

#         # Update T solution for u
#         sol_T_old_for_u = sol_T_new
#         vtk_file_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
#         save_sol(problem_T.fes[0], sol_T_new, vtk_file_path, point_infos=[('u', sol_u_list[0])], 
#                                                              cell_infos=[('f_plus', np.mean(plastic_info[0], axis=1)),
#                                                                          ('strain_xx', np.mean(params_u[1][:, :, 0, 0], axis=1)),
#                                                                          ('stress_xx', np.mean(plastic_info[1], axis=1)),
#                                                                          ('phase', np.max(params_u[-1][:, :, 0], axis=1))])
#     # Update T solution
#     sol_T_old = sol_T_new
#     ## JUST TRY 10 LOOPs FIRST TO MAKE SURE IT WORKS
#     if i == 10:
#         break

# timer.stop()
# runtime = timer.get_time()
# timer.print_time()

# # Save simulation settings, process variables, and runtime to a csv file.
# header = ['Lx', 'Ly', 'Lz', 'Nx', 'Ny', 'Nz', 'ele_type', 'dt', 't_laser',
#           't_sim', 'runtime', 'C_p', 'rho', 'k', 'h', 'eta', 'emissivity',
#           'T_liquid', 'T_0', 'P', 'vel', 'r_b', 'E_0', 'sig_0', 'alpha_V0', 'nu']
# settings = [Lx, Ly, Lz, Nx, Ny, Nz, ele_type, dt, laser_on_t, simulation_t,
#             runtime, Cp, rho, k, h, eta, emissivity, Tl, T0, P, vel, rb, E0,
#             sig0, alpha_V0, nu]
# write_to_csv(csv_file, header, [settings])

## MAIN FUNCTION
if __name__ == '__main__':
    # Start the timer
    timer = Timer()
    timer.start()

    # If you have multiple GPUs, set the one to use.
    device = 'gpu'

    if device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Define some useful directory paths.
    sim_dir = "/scratch/yzhang11/outputs/lpbf_simulations" ## CAN BE A FUNCTION ARGUMENT
    out_dir = os.path.join(sim_dir, '01_data')  ## HOW TO NAME THE SIMULATION FOLDER??
    vtk_dir = os.path.join(out_dir, 'vtk')
    csv_file = os.path.join(out_dir, 'settings.csv')

    # Do some cleaning work.
    files = glob.glob(os.path.join(vtk_dir, f'*'))

    for f in files:
        os.remove(f)
    
    ################################################################################
    # NEED TO REPLACE WITH THE RIGHT VALUES AND ADD DISTRIBUTIONS 
    ################################################################################
    # Define thermal properties in SI units.
    Cp = 670.  # specific heat capacity (J/kg·K)
    rho = 8440.  # material density (kg/m^3)
    k = 25.2  # thermal conductivity (W/m·K)
    Tl = 1623  # liquidus temperature (K)
    h = 100.  # heat convection coefficien (W/m^2·K)
    eta = 0.25  # absorption rate 0.37
    SB_constant = 5.67e-8  # Stefan-Boltzmann constant (kg·s^-3·K^-4)
    emissivity = 0.3  # emissivity
    T0 = 293.  # ambient temperature (K)
    POWDER = 0  # powder flag
    LIQUID = 1  # liquid flag
    SOLID = 2  # solid flag

    # Define mechanical properties in SI units.
    nu = 0.3  # Poisson ratio
    E0 = 70.e9  # Young's modulus (Pa)
    sig0 = 250.e6  # yield stress (Pa)
    alpha_V0 = 1e-5  # thermal expansion coefficient (m/m·K)

    # Define laser properties.
    vel = 0.875 # laser scanning velocity (m/s)
    rb = 0.1e-3 # laser beam size (m)
    P = 169. # laser power (W)
    
    # Specify mesh-related information. 
    # We use first-order hexahedron element for both T_crt and u_crt.
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Nx, Ny, Nz = 400, 128, 64  # 12.5 μm in x-direction and 6.25 μm in y and z-directions
    Lx, Ly, Lz = 5.0e-3, 0.8e-3, 0.4e-3 # domain size (m)
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, out_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    
    # Define thermal problem
    dirichlet_bc_info_T = [[bottom], [0], [thermal_dirichlet_bottom]]
    location_fns = [top, walls]

    sol_T_old = T0 * np.ones((len(mesh.points), 1))  # initial temperature field
    sol_T_old_for_u = np.array(sol_T_old)
    problem_T = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info_T, location_fns=location_fns)

    # Define mechanical problem
    dirichlet_bc_info_u = [[bottom] * 3, [0, 1, 2], [displacement_dirichlet_bottom] * 3]
    problem_u = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info_u)
    params_u = problem_u.internal_vars
    sol_u_list = [np.zeros((problem_u.fes[0].num_total_nodes, problem_u.fes[0].vec))]  # initial displacement field
    
    # Save initial solution to local folder.
    vtk_file_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
    save_sol(problem_T.fes[0], sol_T_old, vtk_file_path, point_infos=[('u', np.zeros((len(sol_T_old), 3)))], 
                                                        cell_infos=[('f_plus', np.zeros(len(mesh.cells))),
                                                                    ('strain_xx', np.zeros(len(mesh.cells))),
                                                                    ('stress_xx', np.zeros(len(mesh.cells))),
                                                                    ('phase', np.mean(params_u[-1][:, :, 0], axis=1))])

    ############################################################################################
    # NEED TO REVISE THE TIME ITERATION LOOP TO FIT THE CURRENT PROJECT
    ############################################################################################
    # Start the major loop of time iteration.
    dt = 1e-6  # time increment
    laser_on_t = 0.5 * Lx / vel  # time duration of laser scanning
    simulation_t = 3 * laser_on_t  # total simulation time until completely cooling down
    ts = np.arange(0., simulation_t, dt)  # time steps

    for i in range(len(ts[1:])):
        laser_center = np.array([Lx * 0.25 + vel * ts[i + 1], Ly / 2., Lz])  # laser scanning starts at the point (0.25*Lx, 0.5*Ly, Lz)
        switch = np.where(ts[i + 1] < laser_on_t, 1., 0.)  # turn off the laser after some time
        print(f"\nStep {i + 1}, total step = {len(ts[1:])}, laser_x = {laser_center[0]}, Lx = {Lx}, laser ON = {ts[i + 1] < laser_on_t}")

        # Set parameter and solve for T
        problem_T.set_params([sol_T_old, laser_center, switch])
        sol_T_new_list = solver(problem_T)
        sol_T_new = sol_T_new_list[0]

        # Since mechanics problem is more expensive to solve, we skip some steps of the thermal problem.
        if (i + 1) % 10 == 0:
            params_u = problem_u.update_dT_and_phase(sol_T_new - sol_T_old_for_u, sol_T_new, params_u)

            # Set parameter and solve for u
            problem_u.set_params(params_u)
            sol_u_list = solver(problem_u, solver_options={'initial_guess': sol_u_list})

            params_u, plastic_info = problem_u.update_stress_strain(sol_u_list[0], params_u) 

            # Check if plastic deformation occurs (with f_yield_vals > 0.)
            print(f"max f_plus = {np.max(plastic_info[0])}, max stress_xx = {np.max(plastic_info[1])}")

            # Update T solution for u
            sol_T_old_for_u = sol_T_new
            vtk_file_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
            save_sol(problem_T.fes[0], sol_T_new, vtk_file_path, point_infos=[('u', sol_u_list[0])], 
                                                                cell_infos=[('f_plus', np.mean(plastic_info[0], axis=1)),
                                                                            ('strain_xx', np.mean(params_u[1][:, :, 0, 0], axis=1)),
                                                                            ('stress_xx', np.mean(plastic_info[1], axis=1)),
                                                                            ('phase', np.max(params_u[-1][:, :, 0], axis=1))])
        # Update T solution
        sol_T_old = sol_T_new
        ## JUST TRY 10 LOOPs FIRST TO MAKE SURE IT WORKS
        if i == 10:
            break

    # Stop the timer
    timer.stop()
    runtime = timer.get_time()
    timer.print_time()

    # Save simulation settings, process variables, and runtime to a csv file.
    header = ['Lx', 'Ly', 'Lz', 'Nx', 'Ny', 'Nz', 'ele_type', 'dt', 't_laser',
              't_sim', 'runtime', 'C_p', 'rho', 'k', 'h', 'eta', 'emissivity',
              'T_liquid', 'T_0', 'P', 'vel', 'r_b', 'nu', 'E_0', 'sig_0', 'alpha_V0']
    settings = [Lx, Ly, Lz, Nx, Ny, Nz, ele_type, dt, laser_on_t, simulation_t,
                runtime, Cp, rho, k, h, eta, emissivity, Tl, T0, P, vel, rb, nu,
                E0, sig0, alpha_V0]
    write_to_csv(csv_file, header, [settings])
