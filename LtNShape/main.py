from conf import configuration_ex1, configuration_ex2, configuration_ex3, configuration_ex4
import ltn_shape_problem
import warnings
warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive. "
                                          "lil_matrix is more efficient.")

if __name__ == '__main__':
    conf = configuration_ex1
    interface_problem = ltn_shape_problem.NonlocalShapeProblem(conf)
    interface_problem.solve_shape_problem(conf["number_iterations"], conf["shape_gradient_tol"])
