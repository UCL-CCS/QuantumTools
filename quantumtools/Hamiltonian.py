from .symplectic_form import PauliwordOp, symplectic_to_string
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt


class QubitHamiltonian(PauliwordOp):
    # TODO
    def __init__(self,
            operator: Union[List[str], Dict[str, float], np.array],
            coeff_list=None):
        super().__init__(operator, coeff_list)

    def Get_ground_state(self):
        # TODO
        pass


class HamiltonianGraph(QubitHamiltonian):
    # TODO
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_list: Union[List[complex], np.array] = None):
        super().__init__(operator, coeff_list)

    def build_graph(self, edge_relation='C', weighted=None):

        if edge_relation =='AC':
            # commuting edges
            adjacency_mat = np.bitwise_not(self.adjacency_matrix)

            # removes self adjacency for graph
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='C':
            # anticommuting edges
            adjacency_mat = self.adjacency_matrix
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='QWC':
            adjacency_mat = np.zeros((self.n_terms, self.n_terms))
            for i in range(self.n_terms):
                for j in range(i+1, self.n_terms):
                    Pword_i = self.symp_matrix[i]
                    Pword_j = self.symp_matrix[j]

                    self_I = np.bitwise_or(Pword_i[:self.n_qubits], Pword_i[self.n_qubits:]).astype(bool)
                    Pword_I = np.bitwise_or(Pword_j[:self.n_qubits], Pword_j[self.n_qubits:]).astype(bool)

                    # Get the positions where neither self nor Pword have I acting on them
                    unique_non_I_locations = np.bitwise_and(self_I, Pword_I)

                    # check non I operators are the same!
                    same_Xs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[:self.n_qubits][unique_non_I_locations],
                                       Pword_j[:self.n_qubits][unique_non_I_locations]).astype(
                            bool))
                    same_Zs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[self.n_qubits:][unique_non_I_locations],
                                       Pword_j[self.n_qubits:][unique_non_I_locations]).astype(
                            bool))

                    if np.all(same_Xs) and np.all(same_Zs):
                        adjacency_mat[i,j] = adjacency_mat[j,i] = 1
                    else:
                        continue
        else:
            raise ValueError(f'unknown edge relation: {edge_relation}')

        graph = nx.from_numpy_matrix(adjacency_mat)

        return graph

    def clique_cover(self, clique_relation, colouring_strategy, colour_interchange=False):

        if clique_relation == 'C':
            graph = self.build_graph(edge_relation='C')
        elif clique_relation == 'AC':
            graph = self.build_graph(edge_relation='AC')
        elif clique_relation == 'QWC':
            graph = self.build_graph(edge_relation='QWC')
            graph = nx.complement(graph)
        else:
            raise ValueError(f'unknown clique relation: {clique_relation}')

        # keys give symplectic row index and value gives colour of clique
        greedy_colouring_output_dic = nx.greedy_color(graph,
                                                      strategy=colouring_strategy,
                                                      interchange=colour_interchange)

        unique_colours = set(greedy_colouring_output_dic.values())

        clique_dict = {}
        for Clique_ind in unique_colours:
            clique_Pword_symp = []
            clique_coeff_symp = []
            for sym_row_ind, clique_id in greedy_colouring_output_dic.items():
                if clique_id == Clique_ind:
                    clique_Pword_symp.append(self.symp_matrix[sym_row_ind,:])
                    clique_coeff_symp.append(self.coeff_vec[sym_row_ind])

            clique = PauliwordOp(np.array(clique_Pword_symp, dtype=int),
                                 clique_coeff_symp)

            clique_dict[Clique_ind] = clique

        return clique_dict

    def draw_graph(self, graph_input, with_node_label=False, node_sizes=True, ):

        if node_sizes:
            node_sizes = 200 * np.abs(np.round(self.coeff_vec)) + 1

        options = {
            'node_size': node_sizes,
            'node_color': 'b'
                 }

        plt.figure()
        pos = nx.circular_layout(graph_input)

        nx.draw_networkx_nodes(graph_input,
                               pos,
                               nodelist=list(graph_input.nodes),
                               **options)
        nx.draw_networkx_edges(graph_input, pos,
                               width=1.0,
                               # alpha=0.5
                               nodelist=list(graph_input.nodes),
                               )

        if with_node_label:
            labels = {row_ind: symplectic_to_string(self.symp_matrix[row_ind]) for row_ind in graph_input.nodes}
            nx.draw_networkx_labels(graph_input,
                                    pos,
                                    labels,
                                    font_size=18)

        # plt.savefig('G_raw', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
        plt.show()
        return None
