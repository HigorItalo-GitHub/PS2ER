import graph
import Probabilistic_State as pst
from scipy import stats
from random import random as rd

'''
Probabilistic version of a graph. It uses probabilistic states instead of 
common states. This allows for a method that computes statistical tests of two
states' morphs, create randomly generated sequences and use statistical criteria
to complete the last level of the graph.
'''


class ProbabilisticGraph(graph.Graph):
    def __init__(self, states=[], alphabet=[], path=''):
        p_states = []
        for s in states:
            if s:
                s = pst.ProbabilisticState(s.name, s.outedges, s.state_prob)
                p_states.append(s)
        graph.Graph.__init__(self, p_states, alphabet, path)
