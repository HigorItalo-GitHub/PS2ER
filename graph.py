import state as st
# import yaml
import itertools

class Graph:
    '''
    This class represents a graph by a list of states (its nodes) and a list of
    the letters (the alphabet) from which it is constructed. As the states
    contain the information about the edges, a list of them fully describe a
    graph.
    There are methods to read a list of states from a file and to save it in the
    same format.
    '''

    def __init__(self, states=[], alphabet=[], path=''):
        if path:
            self.open_graph_file(path)
        else:
            self.states = states  # Graph's list of states
            self.alphabet = alphabet  # List of letters representing the alphabet

        self.label_names = [''.join(i) for i in itertools.product(alphabet, repeat=1)]
        self.index_labels = {}

        i = 0
        for w in self.label_names:
            self.index_labels[w] = i
            i += 1

    def __str__(self):
        print("\nGraphs's states:")
        for s in self.states:
            print(f'{s.name}: {s.outedges}')
        r = '****************************************\n'
        r += 'Number of states: ' + str(len(self.states)) + '\n'
        # states_number = len(self.states)
        return r

    def print_state_named(self, n):
        s = self.state_named(n)
        print(s)
