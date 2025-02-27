import numpy as np
import os

import fenics
import meshio
import nlfem


def read_gmsh(filename, method='CG', problem='dirichlet'):
    if problem not in {"dirichlet", "neumann"}:
        raise ValueError("Wrong argument for the variable 'problem'. Use 'dirichlet' or 'neumann' instead.")
    meshio_mesh = meshio.read(filename)
    elements = np.array(meshio_mesh.cells_dict["triangle"], dtype=np.long)
    vertices = np.array(meshio_mesh.points[:, :2])
    elementLabels = np.array(meshio_mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.long)

    try:
        lines = np.array(meshio_mesh.cells_dict["line"], dtype=int)
        # The following line may be needed in the future, if we have multiple boundaries (maybe including the boundary
        # of Omega)
        lineLabels = np.array(meshio_mesh.cell_data_dict["gmsh:physical"]["line"], dtype=np.long)
    except KeyError:
        lines = None
        lineLabels = None
    number_elements = len(elements)
    if method == "CG":
        vertexLabels = np.zeros(len(vertices), dtype=np.long)
        for i in range(number_elements):
            label = elementLabels[i]
            for k in elements[i]:
                vertexLabels[k] = np.max((vertexLabels[k], label))
    elif method == "DG":
        vertexLabels = np.zeros(3 * len(elements), dtype=np.int)
        for i in range(number_elements):
            vertexLabels[3 * i:3 * i + 2] = elementLabels[i]
    else:
        raise ValueError("Wrong argument for the variable 'method'. Use 'DG' or 'CG' instead.")

    if problem == "dirichlet":
        boundary_label = np.max(elementLabels)
        elementLabels[elementLabels == boundary_label] = -1.0
        vertexLabels[vertexLabels == boundary_label] = -1.0

    return elements, vertices, elementLabels, vertexLabels, lines, lineLabels


def get_artificial_node_meshes(mesh_object, method='CG', problem="dirichlet"):
    """In this method the mesh-dictionary is set up. Furthermore, the interface-nodes are doubled and an artificial node
    is added. Elements are changed such that every element, that contains an interface node, now addresses the version
    of this node that corresponds to the label of this element. The two interfaces are now connected through cells with
    label 0 that additionally contain the artificial node."""
    if type(mesh_object) == str:
        elements, vertices, elementLabels, vertexLabels, lines, lineLabels = read_gmsh(mesh_object + ".msh", method,
                                                                                       problem=problem)
    elif type(mesh_object) == dict:
        elements, elementLabels = mesh_object["elements"], mesh_object["elementLabels"]
        vertices, vertexLabels = mesh_object["vertices"], mesh_object["vertexLabels"]
        lines, lineLabels = mesh_object["lines"], mesh_object["lineLabels"]
    else:
        raise ValueError("The variable mesh_object either has to be the name of an .msh file(without msh) or a "
                         "dictionary with the required information.")
    artificial_node = [0.0, 0.0]

    boundary_indices = np.unique(lines)

    # Substract boundary nodes that are on the Dirichlet boundary, extract Dirichlet boundary from lines
    inner_boundary_vertices = np.where([vertexLabels[boundary_indices[i]] > 0.0 for i in range(len(boundary_indices))])
    inner_boundary_vertices = np.array(inner_boundary_vertices[0])
    if len(inner_boundary_vertices) < len(boundary_indices):
        # TODO > or >=?
        boundary_line_indices = np.where([vertexLabels[lines[i][0]] > 0.0 or vertexLabels[lines[i][1]] > 0.0
                                          for i in range(len(lines))])
        lines = lines[boundary_line_indices]
        lineLabels = lineLabels[boundary_line_indices]

        boundary_indices = boundary_indices[inner_boundary_vertices]

    number_vertices = len(vertices)
    number_boundary_nodes = len(boundary_indices)

    # get indices of the new boundary nodes
    new_boundary_indices = np.arange(number_vertices, number_vertices + number_boundary_nodes)
    # create map with the information, which boundary nodes correspond to each other
    boundary_map = np.concatenate((boundary_indices.reshape((number_boundary_nodes, 1)),
                                   new_boundary_indices.reshape((number_boundary_nodes, 1))), axis=1, dtype=np.int)

    # add new vertices
    boundary_vertices = vertices[boundary_indices]
    boundary_vertices = np.append(boundary_vertices, [artificial_node], axis=0)
    vertices = np.append(vertices, boundary_vertices, axis=0)
    artificial_node_index = len(vertices) - 1

    # change interface nodes of elements with label 2 to the new interface nodes
    unique_lineLabels = np.unique(lineLabels)
    for label in unique_lineLabels:
        current_line_vertices = np.unique(lines[lineLabels == label])
        first_line = lines[lineLabels == label][0]

        # returns list of elements, which include the vertices of the first line
        elements_first_line = np.where([first_line[0] in elements[i] and first_line[1] in elements[i]
                                        for i in range(len(elementLabels))])
        max_elementLabel = np.max(elementLabels[elements_first_line])

        element_list = np.where(elementLabels == max_elementLabel)[0]
        for index in element_list:
            element = elements[index]
            for i in range(3):
                # both checks are necessary since we only want to double the current line without the dirichlet nodes
                # (dirichlet nodes can be in current_line_vertices but not in boundary_indices)
                if element[i] in current_line_vertices and element[i] in boundary_indices:
                    index_boundary_list = np.where(boundary_indices == element[i])
                    elements[index][i] = boundary_map[index_boundary_list[0], 1]

    # add new elements with label 0 to connect the two boundaries that are the same
    number_new_elements = len(lines)
    new_elements = []
    for index in range(number_new_elements):
        vertex_1_index = np.where(boundary_indices == lines[index][0])[0]
        if len(vertex_1_index) == 0:
            # vertex_1 is on the Dirichlet boundary and therefore is not doubled
            vertex_1 = lines[index][0]
        else:
            vertex_1 = boundary_map[vertex_1_index[0], 1]

        vertex_2_index = np.where(boundary_indices == lines[index][1])[0]
        if len(vertex_2_index) == 0:
            # vertex_2 is on the Dirichlet boundary and therefore is not doubled
            vertex_2 = lines[index][0]
        else:
            vertex_2 = boundary_map[vertex_2_index[0], 1]

        new_elements.append([vertex_1, vertex_2])
    new_elements = np.array(new_elements, dtype=np.int)
    new_elements = np.concatenate((lines, new_elements), axis=0, dtype=np.int)
    artificial_node_array = np.ones((2*number_new_elements, 1), dtype=np.int) * artificial_node_index
    new_elements = np.concatenate((new_elements, artificial_node_array), axis=1)
    elements = np.concatenate((elements, new_elements), axis=0, dtype=np.int)
    new_elementLabels = np.zeros(len(new_elements), dtype=np.int)
    elementLabels = np.concatenate((elementLabels, new_elementLabels), dtype=np.int)
    vertexLabels = nlfem.get_vertexLabel(elements, elementLabels, vertices)

    mesh = {"elements": elements,
            "vertices": vertices,
            "elementLabels": elementLabels,
            "vertexLabels": vertexLabels,
            "lines": lines,
            "lineLabels": lineLabels,
            "outdim": 1}
    return boundary_map, mesh


def get_mesh(name):
    mesh = fenics.Mesh()
    with fenics.XDMFFile("mesh/" + name + ".xdmf") as infile:
        infile.read(mesh)
    return mesh


def get_mesh_and_mesh_data(name, interface_label):
    mesh = fenics.Mesh()
    with fenics.XDMFFile("mesh/" + name + ".xdmf") as infile:
        infile.read(mesh)

    subdomain_data = fenics.MeshValueCollection("size_t", mesh, 2)
    with fenics.XDMFFile("mesh/" + name + "_subdomains.xdmf") as infile:
        infile.read(subdomain_data)
    subdomain_function = fenics.cpp.mesh.MeshFunctionSizet(mesh, subdomain_data)

    boundary_data = fenics.MeshValueCollection("size_t", mesh, 1)
    with fenics.XDMFFile("mesh/" + name + "_boundaries.xdmf") as infile:
        infile.read(boundary_data)
    interface_function = fenics.cpp.mesh.MeshFunctionSizet(mesh, boundary_data)
    interface_array = interface_function.array()
    for l in range(interface_array.size):
        if interface_array[l] != interface_label:
            interface_function.set_value(l, 0)

    return mesh, subdomain_function, interface_function


def get_interface_indices(mesh, interface, interface_label):
    # Find facets on interior boundary
    indices_interface_facets = []
    for facet in range(len(interface)):
        if interface[facet] == interface_label:
            indices_interface_facets.append(facet)

    # Find vertices on interior boundary
    interface_vertices = []
    interface_elements = []
    for cell in fenics.cells(mesh):
        for facet in fenics.facets(cell):
            if facet.index() in indices_interface_facets:
                interface_elements.append(cell.index())
                for vertex in fenics.vertices(facet):
                    interface_vertices.append(vertex.index())

    return list(set(interface_vertices)), list(set(interface_elements))


def convert_mesh(mesh, subdomains):
    elements = np.array(mesh.cells(), dtype=int)
    vertices = mesh.coordinates()
    num_elements = elements.shape[0]
    elementLabels = np.array(subdomains.array(), dtype='long')
    for triangle in range(num_elements):
        if elementLabels[triangle] == 3:
            elementLabels[triangle] = -1.0
    return elements, vertices, elementLabels


def get_indices_boundary_vertices(mesh, subdomains, boundary_label):
    elements = np.array(mesh.cells(), dtype=int)
    num_elements = elements.shape[0]
    elementLabels = np.array(subdomains.array(), dtype='long')
    indices_boundary_vertices = []
    for triangle in range(num_elements):
        if elementLabels[triangle] == boundary_label:
            for vertex in elements[triangle]:
                indices_boundary_vertices.append(vertex)
    return list(set(indices_boundary_vertices))


def convert_msh_to_xdmf(name, boundary_label):
    msh = meshio.read("mesh/" + name + ".msh")
    for cell in msh.cells:
        if cell.type == "triangle":
            elements = cell.data
        elif cell.type == "line":
            lines = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            elementLabels = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "line":
            lineLabels = msh.cell_data_dict["gmsh:physical"][key]

    mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": elements})
    mesh_function_subdomains = meshio.Mesh(points=msh.points[:, :2],
                                cells=[("triangle", elements)],
                                cell_data={"name_to_read": [elementLabels]})
    mesh_function_boundaries = meshio.Mesh(points=msh.points[:, :2],
                                           cells=[("line", lines)],
                                           cell_data={"name_to_read": [lineLabels]})
    meshio.write("mesh/" + name + ".xdmf", mesh)
    meshio.write("mesh/" + name + "_subdomains.xdmf", mesh_function_subdomains)
    meshio.write("mesh/" + name + "_boundaries.xdmf", mesh_function_boundaries)

    number_elements = len(elements)
    for triangle in range(number_elements):
        if elementLabels[triangle] == boundary_label:
            elementLabels[triangle] = -1.0
    vertices = msh.points
    vertexLabels = nlfem.get_vertexLabel(elements, elementLabels, vertices)

    return elements, elementLabels, lines, lineLabels, vertexLabels


def remesh(mesh, interface_vertices, name, num, element_size):
    new_name = name + "_remeshed_" + str(num)
    create_geo_file(mesh, interface_vertices, new_name)
    os.system('gmsh mesh/' + new_name + '.geo -2 -clscale ' + str(element_size)
              + ' -format msh22 -o mesh/' + new_name + '.msh')
    return new_name


def create_geo_file(mesh, interface_vertices, name):
    dummy_file = open("mesh/dummy_shape.geo", "r")
    data = dummy_file.readlines()

    new_file = open("mesh/" + name + ".geo", "w")

    for line in data[0:10]:
        new_file.write(line)

    # Prepare text, that is added to the .geo-file.
    vertices = mesh.coordinates()
    counter = 0
    interface = "Spline(10) = {"
    for index in interface_vertices:
        coordinates = vertices[index]
        new_file.write("Point(" + str(10 + counter) + ") = {" + str(coordinates[0]) + ", " + str(coordinates[1])
                        + ", 0, lc};\n")
        interface += str(10 + counter) + ", "
        counter += 1
    interface += "10};"

    for line in data[10:40]:
        new_file.write(line)

    new_file.write(interface)

    for line in data[40:]:
        new_file.write(line)

    dummy_file.close()
    new_file.close()


class MeshData:

    def __init__(self, file_name, interface_label, boundary_label):
        self.elements, self.elementLabels, self.lines, self.lineLabels, self.vertexLabels \
            = convert_msh_to_xdmf(file_name, boundary_label)
        self.mesh, self.subdomains, self.interface = get_mesh_and_mesh_data(file_name, interface_label)
        self.interface_vertices, self.interface_elements = get_interface_indices(self.mesh, self.interface,
                                                                                 interface_label)
        self.boundary_vertices = get_indices_boundary_vertices(self.mesh, self.subdomains, boundary_label)
        vertices = np.array([i for i in range(self.mesh.num_vertices())])
        self.indices_nodes_not_on_interface = list(set(vertices) - set(self.interface_vertices))
