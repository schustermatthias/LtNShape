import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg
import scipy.sparse as sp

import nlfem

import helper


def identity(mesh, weights, points, kappa=None, subdomainLabel=None):
    vertices = mesh["vertices"]
    number_vertices = len(vertices)
    L = sp.lil_matrix((number_vertices, number_vertices), dtype=float)
    if subdomainLabel is None:
        elements = mesh["elements"]
    else:
        elements = mesh["elements"][mesh["elementLabels"] == subdomainLabel]
    points_transposed = points.transpose()
    basis_function_values = helper.basis_function_value(points_transposed)

    for i in range(len(elements)):
        triangle_i = elements[i]
        triangle_i_vertices = np.array([vertices[triangle_i[0]], vertices[triangle_i[1]], vertices[triangle_i[2]]])
        matrix_i = np.array([triangle_i_vertices[1] - triangle_i_vertices[0],
                             triangle_i_vertices[2] - triangle_i_vertices[0]]).transpose()
        det_matrix_i = matrix_i[0, 0] * matrix_i[1, 1] - matrix_i[1, 0] * matrix_i[0, 1]
        if kappa is not None:
            points_transformed = np.array([triangle_i_vertices[0] + matrix_i @ points[i] for i in range(len(points))])
            kappa_values = kappa(points_transformed.transpose())
        else:
            kappa_values = np.ones(len(points))
        for a in range(3):
            for b in range(3):
                integral_value = (kappa_values * basis_function_values[a] * basis_function_values[b] * weights).sum()
                L[triangle_i[a], triangle_i[b]] += abs(det_matrix_i) * integral_value

    return L.tocsr()


def laplace(mesh, subdomainLabel=None, boundary_map=None, coupling="neumann"):
    """Works currently only for two domains, where the local domain has label 2"""
    if coupling == "dirichlet" and boundary_map is not None:
        boundary_vertices = boundary_map[:, 1]
    else:
        boundary_vertices = []
    # deform = kwargs.get('deform', 0)
    vertices = mesh["vertices"] # + deform
    number_vertices = len(vertices)
    L = sp.lil_matrix((number_vertices, number_vertices), dtype=float)
    if subdomainLabel is None:
        elements = mesh["elements"]
    else:
        elements = mesh["elements"][mesh["elementLabels"] == subdomainLabel]
    gradient = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]
    for i in range(len(elements)):
        triangle_i = elements[i]
        triangle_i_vertices = np.array([vertices[triangle_i[0]], vertices[triangle_i[1]], vertices[triangle_i[2]]])
        matrix_i = np.array([triangle_i_vertices[1] - triangle_i_vertices[0],
                             triangle_i_vertices[2] - triangle_i_vertices[0]]).transpose()
        det_matrix_i = matrix_i[0, 0] * matrix_i[1, 1] - matrix_i[1, 0] * matrix_i[0, 1]
        inverse_matrix_i = (1/det_matrix_i) * np.array([[matrix_i[1, 1], -matrix_i[0, 1]],
                                                        [-matrix_i[1, 0], matrix_i[0, 0]]])
        for a in range(3):
            if triangle_i[a] not in boundary_vertices:
                for b in range(3):
                    L[triangle_i[a], triangle_i[b]] += 0.5 * abs(det_matrix_i) * gradient[a].dot(inverse_matrix_i.dot(inverse_matrix_i.transpose().dot(gradient[b])))
                    #np.dot(inverse_matrix_i @ gradient[a], inverse_matrix_i @ gradient[b])

    if boundary_map is not None and coupling=="dirichlet":
        for a in range(len(boundary_map)):
            L[boundary_map[a, 1], boundary_map[a, 1]] = 1.0
            L[boundary_map[a, 1], boundary_map[a, 0]] = -1.0
    return L.tocsr()


class Schwarz:
    def __init__(self, problem_data, cholesky=0, one_mesh=1, operator="convection-diffusion", local_domain=None,
                 coupling="neumann", boundary_map=None, nonlocal_neumann_problem=None, points=None, weights=None,
                 subdomain_labels=None, neumann_boundary_map=None, kappa_list=None):
        self.problem_data = problem_data
        self.cholesky = cholesky
        self.one_mesh = one_mesh
        self.cholesky_matrices = []
        self.vertexLabels = []
        self.vertices = []
        self.stiffness_matrices = []
        self.load_vectors = []
        self.ansatz = problem_data[0]["conf"]["ansatz"]
        self.residual = []
        self.energy_error = []
        self.solution_error = []

        self.initial_matrices = []

        if subdomain_labels is not None:
            self.number_subdomains = len(subdomain_labels)
            self.subdomain_labels = subdomain_labels
        else:
            self.number_subdomains = len(problem_data)
            self.subdomain_labels = range(1, self.number_subdomains + 1)

        if self.ansatz not in {"CG"}:
            raise ValueError("Wrong argument for the variable 'method'. Use 'CG' instead. 'DG' is not implemented at "
                             "the moment.")

        if operator not in {"convection-diffusion", "diffusion"}:
            raise ValueError("Wrong argument for the variable 'operator'. Use 'convection-diffusion' or 'diffusion' "
                             "instead.")

        initial_matrices = []
        for subdomain in range(self.number_subdomains):
            subdomain_label = self.subdomain_labels[subdomain]
            mesh = problem_data[subdomain]["mesh"]
            elements = mesh["elements"]
            elementLabels = mesh["elementLabels"]

            vertices = mesh["vertices"]
            kernel = problem_data[subdomain]["kernel"]
            conf = problem_data[subdomain]["conf"]

            if local_domain is not None:
                if local_domain == subdomain_label:
                    A = laplace(mesh, subdomain_label, boundary_map, coupling)
                    vertexLabels = nlfem.get_vertexLabel(elements, elementLabels, vertices)
                    if coupling == "neumann":
                        empty_dict = {}
                        conf['is_ShapeDerivative'] = 0
                        mesh_nl, A_neumann = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel,
                                                                             conf, empty_dict)
                        A = A + A_neumann
                else:
                    empty_dict = {}
                    conf['is_ShapeDerivative'] = 0
                    mesh_nl, A = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel, conf,
                                                                 empty_dict)
                    vertexLabels = mesh_nl["vertexLabels"]
            else:
                if nonlocal_neumann_problem is None:
                    empty_dict = {}
                    conf['is_ShapeDerivative'] = 0
                    mesh_nl, A = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel, conf,
                                                                 empty_dict)
                    vertexLabels = mesh_nl["vertexLabels"]
                else:
                    if points is None or weights is None:
                        p = conf["quadrature"]["outer"]["points"]
                        w = conf["quadrature"]["outer"]["weights"]
                    else:
                        p = points
                        w = weights


                    if kappa_list is not None:
                        kappa = kappa_list[subdomain]
                    else:
                        kappa = None
                    A = identity(mesh, w, p, subdomainLabel=subdomain_label, kappa=kappa)
                    empty_dict = {}
                    conf['is_ShapeDerivative'] = 0
                    mesh_nl, A_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel, conf,
                                                                    empty_dict)
                    A = A + A_nl
                    vertexLabels = mesh_nl["vertexLabels"]
                    # Change boundary label to corresponding domain label
                    for pair in neumann_boundary_map:
                        boundary_label = pair[0]
                        domain_label = pair[1]
                        vertexLabels[np.where(vertexLabels == boundary_label)] = domain_label

            self.vertexLabels.append(vertexLabels)
            self.initial_matrices.append(A)
            self.vertices.append(vertices)

        for subdomain in range(self.number_subdomains):
            subdomainLabel = self.subdomain_labels[subdomain]  # label for elements in this subdomain

            A = self.initial_matrices[subdomain]
            vertexLabels = self.vertexLabels[subdomain]

            if not self.one_mesh:
                mesh = problem_data[subdomain]["mesh"]
                elements = mesh["elements"]
                elementLabels = mesh["elementLabels"]
                vertices = self.vertices[subdomain]
                empty_dict = {}

            # Set up matrices for every subproblem
            matrices = []
            for domain in range(self.number_subdomains):
                domainLabel = self.subdomain_labels[domain]
                if subdomain == domain or operator=="diffusion" or (coupling=="dirichlet" and
                                                                    local_domain == subdomainLabel):
                    A_O = A[vertexLabels == subdomainLabel][:, vertexLabels == domainLabel]
                else:
                    if one_mesh:
                        A_domain = self.initial_matrices[domain]
                    else:
                        kernel_domain = problem_data[domain]["kernel"]
                        conf_domain = problem_data[domain]["conf"]
                        mesh_nl, A_domain = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices,
                                                                            kernel_domain, conf_domain, empty_dict)
                    A_O = A_domain[vertexLabels == subdomainLabel][:, vertexLabels == domainLabel]

                matrices.append(A_O)
            A_D = A[vertexLabels == subdomainLabel][:, vertexLabels < 0]
            matrices.append(A_D)
            f = problem_data[subdomain]["load_vector"][vertexLabels == subdomainLabel]

            if self.cholesky:
                l_upper, l_lower = scipy.linalg.cho_factor(matrices[subdomain].toarray())
                self.cholesky_matrices.append([l_upper, l_lower])

            self.stiffness_matrices.append(matrices)
            self.load_vectors.append(f)

        # number of vertices in every subdomain
        self.number_inner_vertices = [len(self.vertices[subdomain][self.vertexLabels[subdomain] == subdomain + 1])
                                      for subdomain in range(self.number_subdomains)]

    def boundary_function_default(self, x):
        return 0.0

    def block_iterative_method(self, max_iterations, save_history, method="multiplicative", energy_tol=-1.0,
                               residual_tol=-1.0, print_error=0, boundary_function=None, initial_solution=None,
                               gmres_tol=1E-12, exact_solution=None):
        if method not in {"additive", "multiplicative"}:
            raise ValueError("Wrong argument for the variable 'method'. Use 'additive' or 'multiplicative' instead.")

        boundary_function = self.get_boundary_function(boundary_function)
        solutions = self.get_starting_solutions(boundary_function, initial_solution=initial_solution)

        k = 1
        residual = residual_tol + 1
        energy_error = energy_tol + 1
        while k <= max_iterations and energy_error > energy_tol and residual > residual_tol:
            new_solution = []
            for subdomain in range(self.number_subdomains):
                subdomainLabel = self.subdomain_labels[subdomain]
                f = self.load_vectors[subdomain].copy()
                matrices = self.stiffness_matrices[subdomain]
                subdomain_new_solution = np.random.rand(len(solutions[-1][subdomain]))
                # assemble right-hand side
                for domain in range(self.number_subdomains):
                    if domain != subdomain:
                        domainLabel = self.subdomain_labels[domain]
                        if method == "additive" or domain > subdomain:
                            theta = solutions[-1][domain][self.vertexLabels[domain] == domainLabel]
                        else:
                            theta = new_solution[domain][self.vertexLabels[domain] == domainLabel]
                        if not self.one_mesh:
                            theta = helper.linear_interpolation(
                                self.vertices[domain][self.vertexLabels[domain] == domainLabel], theta,
                                self.vertices[subdomain][self.vertexLabels[subdomain] == domainLabel])
                        if len(theta) > 0:
                            f -= matrices[domain] @ theta
                            subdomain_new_solution[self.vertexLabels[subdomain] == domainLabel] = theta

                dirichlet_data = solutions[-1][subdomain][self.vertexLabels[subdomain] < 0.0]
                f -= matrices[-1] @ dirichlet_data
                subdomain_new_solution[self.vertexLabels[subdomain] < 0.0] = dirichlet_data
                subdomain_old_solution = solutions[-1][subdomain][self.vertexLabels[subdomain] == subdomainLabel]
                #Solve subproblem
                if self.cholesky:
                    l_upper = self.cholesky_matrices[subdomain][0]
                    l_lower = self.cholesky_matrices[subdomain][1]
                    solution = scipy.linalg.cho_solve((l_upper, l_lower), f)
                else:
                    solution, info = spla.lgmres(matrices[subdomain], f, x0=subdomain_old_solution, tol=gmres_tol)
                    # print(info)

                subdomain_new_solution[self.vertexLabels[subdomain] == subdomainLabel] = solution

                new_solution.append(subdomain_new_solution)

            energy_error = self.get_energy_error(new_solution, solutions[-1])
            residual = self.get_residual(new_solution)
            if print_error:
                print("Iteration " + str(k) + " energy error: " + str(energy_error)
                      + " residual error " + str(residual) + "\n")
            k += 1

            if save_history:
                solutions.append(new_solution)
                self.energy_error.append(energy_error)
                self.residual.append(residual)
                if exact_solution is not None:
                    solution_error = self.get_solution_error(exact_solution, new_solution)
                    self.solution_error.append(solution_error)
            else:
                solutions[0] = new_solution
        return solutions

    def get_boundary_function(self, function=None):
        boundary_function = self.boundary_function_default
        if function != None:
            boundary_function = function
        return boundary_function

    def get_starting_solutions(self, boundary_function, initial_solution=None):
        solutions = []
        start_solution = []
        if initial_solution is None:
            for subdomain in range(self.number_subdomains):
                number_dofs = len(self.vertexLabels[subdomain])
                solution = np.zeros(number_dofs)  # np.ones(number_vertices)
                # maybe add the following lines also for the case where the initial solution is given.
                dirichlet_nodes = self.vertices[subdomain][self.vertexLabels[subdomain] < 0]
                if len(dirichlet_nodes) > 0:
                    solution[self.vertexLabels[subdomain] < 0.0] = np.apply_along_axis(boundary_function,
                                                                                       1, dirichlet_nodes)
                start_solution.append(solution)
        else:
            for subdomain in range(self.number_subdomains):
                start_solution.append(initial_solution)
        solutions.append(start_solution)
        return solutions

    def get_residual(self, solution):
        """Computes the norm of the residual, i.e., ||Au^k - f||"""
        residual_list = []
        for subdomain in range(self.number_subdomains):
            matrices = self.stiffness_matrices[subdomain]
            subdomainLabel = self.subdomain_labels[subdomain]
            number_dofs = solution[subdomain][self.vertexLabels[subdomain] == subdomainLabel].shape[0]
            result_subdomain = np.zeros(number_dofs)
            for domain in range(self.number_subdomains):
                domainLabel = self.subdomain_labels[domain]
                solution_domain = solution[domain][self.vertexLabels[domain] == domainLabel]
                if not self.one_mesh:
                    solution_domain = helper.linear_interpolation(
                        self.vertices[domain][self.vertexLabels[domain] == domainLabel], solution_domain,
                        self.vertices[subdomain][self.vertexLabels[subdomain] == domainLabel])
                result_subdomain += matrices[domain] @ solution_domain
            boundary_data = solution[subdomain][self.vertexLabels[subdomain] < 0.0]
            result_subdomain += matrices[-1] @ boundary_data
            residual_list.append(self.load_vectors[subdomain] - result_subdomain)
        residual_vector = np.concatenate(residual_list)
        residual = np.linalg.norm(residual_vector)
        return residual

    def get_solution_error(self, exact_solution, current_solution):
        difference_list = []
        for subdomain in range(self.number_subdomains):
            subdomainLabel = self.subdomain_labels[subdomain]
            domain_vertices = self.vertices[subdomain][self.vertexLabels[subdomain] == subdomainLabel]
            exact_solution_vec = np.apply_along_axis(exact_solution, axis=1, arr=domain_vertices)
            solution_domain = current_solution[subdomain][self.vertexLabels[subdomain] == subdomainLabel]
            difference_list.append(exact_solution_vec - solution_domain)
            difference_vec = np.concatenate(difference_list)
        return np.linalg.norm(difference_vec)

    def get_energy_error(self, new_solution, old_solution):
        """Computes the energy norm of the error u^{k+1} - u^k, i.e., ||u^{k+1} - u^k||_A"""
        result = 0.0
        for subdomain in range(self.number_subdomains):
            matrices = self.stiffness_matrices[subdomain]
            subdomainLabel = self.subdomain_labels[subdomain]
            new_solution_subdomain = new_solution[subdomain][self.vertexLabels[subdomain] == subdomainLabel]
            old_solution_sudomain = old_solution[subdomain][self.vertexLabels[subdomain] == subdomainLabel]
            error_vector_subdomain = new_solution_subdomain - old_solution_sudomain
            for domain in range(self.number_subdomains):
                domainLabel = self.subdomain_labels[domain]
                new_solution_domain = new_solution[domain][self.vertexLabels[domain] == domainLabel]
                old_solution_domain = old_solution[domain][self.vertexLabels[domain] == domainLabel]
                error_vector_domain = new_solution_domain - old_solution_domain
                if not self.one_mesh:
                    error_vector_domain = helper.linear_interpolation(
                        self.vertices[domain][self.vertexLabels[domain] == domainLabel], error_vector_domain,
                        self.vertices[subdomain][self.vertexLabels[subdomain] == domainLabel])
                result += np.dot(error_vector_subdomain, matrices[domain] @ error_vector_domain)

        return np.sqrt(result)

    def preconditioned_gmres(self, method=None, mesh=None):

        counter = helper.gmres_counter()

        # build linear operator to solve the whole problem
        number_vertices = np.sum(self.number_inner_vertices)

        complete_operator = spla.LinearOperator(shape=(number_vertices, number_vertices),
                                                matvec=self.matrix_vector_product)
        # build preconditioner as a linear operator
        if method == "jacobi":
            M = spla.LinearOperator(shape=(number_vertices, number_vertices), matvec=self.block_jacobi_preconditioner)
        elif method == "gauss-seidel":
            M = spla.LinearOperator(shape=(number_vertices, number_vertices),
                                    matvec=self.block_gauss_seidel_preconditioner)
        f = np.concatenate(self.load_vectors)

        # start gmres
        if method not in {"jacobi", "gauss-seidel"}:
            gmres_solution, info = spla.gmres(complete_operator, f, tol=1E-10, x0=np.zeros(number_vertices),
                                              callback=counter)
        else:
            gmres_solution, info = spla.gmres(complete_operator, f, M=M, tol=1E-10, x0=np.zeros(number_vertices),
                                              callback=counter)

        divided_solution = self.get_divided_vector(gmres_solution)

        if mesh != None:
            # Interpolate solution on given mesh.
            new_vertices = mesh["vertices"]
            new_vertexLabels = mesh["vertexLabels"]
            solution = self.interpolate_divided_solution(divided_solution, new_vertices, new_vertexLabels)

        elif self.one_mesh:
            solution = np.zeros(len(self.vertices[0]))
            for domain in range(self.number_subdomains):
                domainLabel = domain + 1
                solution[self.vertexLabels[0] == domainLabel] = divided_solution[domain]
        else:
            solution = self.interpolate_divided_solution(divided_solution, self.vertices[0], self.vertexLabels[0])
        return solution

    def interpolate_divided_solution(self, divided_solution, vertices, vertexLabels):
        solution = np.zeros(len(vertices))
        for domain in range(self.number_subdomains):
            domainLabel = domain + 1
            next_solution_interpolated = helper.linear_interpolation(
                                                        self.vertices[domain][self.vertexLabels[domain] == domainLabel],
                                                        divided_solution[domain], vertices[vertexLabels == domainLabel])
            solution[vertexLabels == domainLabel] = next_solution_interpolated
        return solution

    def matrix_vector_product(self, x):
        x_divided = self.get_divided_vector(x)
        solutions = []
        for subdomain in range(self.number_subdomains):
            solution = 0.0
            matrices = self.stiffness_matrices[subdomain]
            for domain in range(self.number_subdomains):
                x_domain = x_divided[domain]
                domainLabel = domain + 1
                if domain != subdomain:
                    if not self.one_mesh:
                        x_domain = helper.linear_interpolation(
                            self.vertices[domain][self.vertexLabels[domain] == domainLabel], x_domain,
                            self.vertices[subdomain][self.vertexLabels[subdomain] == domainLabel])
                    solution += matrices[domain] @ x_domain
                else:
                    solution += matrices[subdomain] @ x_domain
            solutions.append(solution)
        return np.concatenate(solutions)

    def block_jacobi_preconditioner(self, x):
        x_divided = self.get_divided_vector(x)
        solutions = []
        for subdomain in range(self.number_subdomains):
            if self.cholesky:
                l_upper = self.cholesky_matrices[subdomain][0]
                l_lower = self.cholesky_matrices[subdomain][1]
                solution = scipy.linalg.cho_solve((l_upper, l_lower), x_divided)
            else:
                solution = scipy.linalg.solve(self.stiffness_matrices[subdomain][subdomain].toarray(),
                                              x_divided[subdomain])
            solutions.append(solution)
        return np.concatenate(solutions)

    def block_gauss_seidel_preconditioner(self, x):
        x_divided = self.get_divided_vector(x)
        solutions = []
        for subdomain in range(self.number_subdomains):
            matrices = self.stiffness_matrices[subdomain]
            x_subdomain = x_divided[subdomain]
            for domain in range(subdomain):
                domainLabel = domain + 1
                solution_domain = solutions[domain]
                if not self.one_mesh:
                    solution_domain = helper.linear_interpolation(
                        self.vertices[domain][self.vertexLabels[domain] == domainLabel], solution_domain,
                        self.vertices[subdomain][self.vertexLabels[subdomain] == domainLabel])
                x_subdomain -= matrices[domain] @ solution_domain

            if self.cholesky:
                l_upper = self.cholesky_matrices[subdomain][0]
                l_lower = self.cholesky_matrices[subdomain][1]
                solution_subdomain = scipy.linalg.cho_solve((l_upper, l_lower), x_subdomain)
            else:
                solution_subdomain = scipy.linalg.solve(matrices[subdomain].toarray(), x_subdomain)
            solutions.append(solution_subdomain)

        return np.concatenate(solutions)

    def get_divided_vector(self, x):
        end_index = self.number_inner_vertices[0]
        x_divided = [x[range(end_index)]]
        for k in range(1, self.number_subdomains):
            start_index = end_index
            end_index += self.number_inner_vertices[k]
            x_divided.append(x[range(start_index, end_index)])
        return x_divided

    def get_condition_number(self, method=None):
        vertexLabels = self.vertexLabels[0]
        dimension = len(vertexLabels)
        A = np.zeros((dimension, dimension)) # sp.csr_matrix((dimension, dimension))
        #print(vertexLabels)
        for subdomain in range(self.number_subdomains):
            list_rows = list(np.where(vertexLabels == subdomain + 1))
            for domain in range(self.number_subdomains):
                list_columns = list(np.where(vertexLabels == domain + 1))
                submatrix = self.stiffness_matrices[subdomain][domain]
                i = 0
                for row in list_rows[0]:
                    A[row, list_columns[0]] += submatrix[i].toarray()[0]
                    i = i + 1

        A_O = A[vertexLabels > 0][:, vertexLabels > 0]
       # A_O = A_O.toarray()
        if method == 'additive':
            P = np.zeros((dimension, dimension))
            for subdomain in range(self.number_subdomains):
                list_rows = list(np.where(vertexLabels == subdomain + 1))
                submatrix = self.stiffness_matrices[subdomain][subdomain]
                i = 0
                for row in list_rows[0]:
                    P[row, list_rows[0]] += submatrix[i].toarray()[0]
                    i = i + 1
            P = P[vertexLabels > 0][:, vertexLabels > 0]
            P_inverse = scipy.linalg.inv(P)
            A_O = P_inverse @ A_O
        elif method == 'multiplicative':
            P = np.zeros((dimension, dimension))
            for subdomain in range(self.number_subdomains):
                list_rows = list(np.where(vertexLabels == subdomain + 1))
                for domain in range(subdomain + 1):
                    list_columns = list(np.where(vertexLabels == domain + 1))
                    submatrix = self.stiffness_matrices[subdomain][domain]
                    i = 0
                    for row in list_rows[0]:
                        P[row, list_columns[0]] += submatrix[i].toarray()[0]
                        i = i + 1
                # P[list_rows[0]][:, list_columns[0]] = self.stiffness_matrices[subdomain][subdomain]
            P = P[vertexLabels > 0][:, vertexLabels > 0]
            P_inverse = scipy.linalg.inv(P)
            A_O = P_inverse @ A_O

        return np.linalg.cond(A_O)
