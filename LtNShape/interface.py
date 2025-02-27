import time

import nlfem
import numpy as np

import mesh_data
import helper
from schwarz import Schwarz

def local_to_nonlocal_coupling(max_iterations, mesh_dict, confs, kernels, loads, boundary_function, forcing_term=None,
                               coupling="neumann", initial_solution=None, residual_tol=1e-10, gmres_tol=1E-12,
                               print_time=False):
    starting_time = time.time()
    boundary_map, mesh = mesh_data.get_artificial_node_meshes(mesh_dict)

    if initial_solution is not None:
        initial_solution = helper.convert_cg_to_splitted_function(initial_solution, boundary_map)

    if forcing_term is None:
        load_1 = nlfem.loadVector(mesh, loads[0], confs[0])
        load_2 = nlfem.loadVector(mesh, loads[1], confs[1])

    else:
        number_new_vertices = len(boundary_map) + 1
        load_1 = np.concatenate((forcing_term[0], np.zeros(number_new_vertices)), axis=None)
        load_2 = helper.rearrange_forcing_term(forcing_term[1], boundary_map)

    if coupling == "dirichlet":
        for index in range(len(boundary_map)):
            boundary_vertex = boundary_map[index, 1]
            load_2[boundary_vertex] = 0.0

    # dictionary = helper.create_configuration_dict(mesh_dict, kernels_data, confs_data, loads_data)
    # helper.save_dict(dictionary, results_folder + "configuration")


    subproblem_1 = dict(mesh=mesh, kernel=kernels[0], conf=confs[0], load_vector=load_1)
    subproblem_2 = dict(mesh=mesh, kernel=kernels[1], conf=confs[1], load_vector=load_2)

    problem_data = [subproblem_1, subproblem_2]
    problem = Schwarz(problem_data, cholesky=0, operator="diffusion", local_domain=2.0, boundary_map=boundary_map,
                      coupling=coupling)
    intermediate_time = time.time()

    solutions = problem.block_iterative_method(max_iterations, 0, method="multiplicative",
                                               boundary_function=boundary_function, print_error=0,
                                               residual_tol=residual_tol, initial_solution=initial_solution,
                                               gmres_tol=gmres_tol)
    if print_time:
        print('Assembly needed ' + str(intermediate_time - starting_time) + 's')
        print('Iterative algorithm needed ' + str(time.time() - intermediate_time) + 's')

    trimmed_solutions = helper.trim_artificial_node_from_solution_series(solutions)
    # dg_function = helper.convert_to_dg_function(trimmed_solutions[-1][-1], first_mesh["elements"])
    cg_function = helper.convert_to_cg_function(trimmed_solutions[-1][-1], boundary_map)
    # additional_information = {"load_1":load_1, "load_2": load_2, "mesh_dict": mesh}
    return cg_function
