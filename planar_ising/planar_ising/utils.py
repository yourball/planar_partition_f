import numpy as np
from .planar_ising_model import PlanarIsingModel
from ..planar_graph import PlanarGraph, Triangulator
from mpmath import mpf, matrix

def get_inverse_sub_mapping(sub_mapping, sub_elements_count):

    sub_element_exists_mask = (sub_mapping != -1)

    inverse_mapping = -np.ones(sub_elements_count, dtype=int)
    inverse_mapping[sub_mapping[sub_element_exists_mask]] = np.where(sub_element_exists_mask)[0]

    return inverse_mapping

def triangulate_ising_model(ising_model):

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    new_edge_indices_mapping, new_graph = Triangulator.triangulate(graph)

    new_interaction_values = np.zeros(new_graph.edges_count)
    new_interaction_values[new_edge_indices_mapping] = interaction_values

    # print('interaction_values', interaction_values)
    # print('new_edge_indices_mapping', new_edge_indices_mapping)
    # print('len(new_interaction_values)', len(new_interaction_values))

    new_interaction_values = matrix([[mpf('0') for i in range(new_graph.edges_count)]])

    for i, indx in enumerate(new_edge_indices_mapping):
        new_interaction_values[int(indx)] = interaction_values[int(i)]

    # print('new_interaction_values', new_interaction_values)
    return new_edge_indices_mapping, PlanarIsingModel(new_graph, new_interaction_values)
