import numpy as np
import graphviz
from typing import List


# Define transition matrix and initial state
P = [
    [0.4, 0.25, 0.20, 0.10, 0.05],
    [0, 0.45, 0.25, 0.20, 0.10],
    [0, 0, 0.3, 0.45, 0.25],
    [0, 0, 0, 0.35, 0.65],
    [0, 0, 0, 0, 1]
]

p = [1, 0, 0, 0, 0]

# Create the graph
dot = graphviz.Digraph()

# Add nodes
for i in range(len(P)):
    dot.node(str(i), label=f"State {i}")

# Add edges
for i in range(len(P)):
    for j in range(len(P[i])):
        if P[i][j] > 0:
            dot.edge(str(i), str(j), label=f"{P[i][j]:.2f}")

# Render and display the graph
dot.render('markov_chain_graph', format='png', cleanup=True)


class MarkovChain:
    def __init__(self, transition_matrix: List[List[float]], initial_state: List[float]) -> None:

        self.P = np.array(transition_matrix)
        self.p_initial = np.array(initial_state)

    def run_tests(self, num_tests: int = 5) -> None:
        p_current = self.p_initial
        for i in range(num_tests):
            p_current = np.dot(p_current, self.P)
            print(f'Після {i + 1}-го тесту:')
            for j, p in enumerate(p_current):
                print(f'p{j + 1}({i + 1}) = {p:.4f}', end=' ')
            print('\n')

    def calculate_state_after_iterations(self, num: int = 5) -> np.array:
        return np.dot(self.p_initial, np.linalg.matrix_power(self.P, num))


p = [1, 0, 0, 0, 0]

mc = MarkovChain(P, p)
mc.run_tests()

print('Перевірка p(5) = p(0) * P⁵:')
for i, p in enumerate(mc.calculate_state_after_iterations()):
    print(f'p({i + 1}) = {p:.4f}', end=' ')