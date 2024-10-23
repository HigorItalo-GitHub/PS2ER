class State:
    '''
    This class represents a state or node in a graph. This representation is
    done using a name (or a label) to refer to the state and a list of its
    outgoing edges. The outgoing edges list is a list of tuples where the first
    element is the letter from the automata/shift space alphabet labeling the
    edge and the second element is the name of the state to where that edge
    leads.
    There are methods to retrieve the destination of an outgoing edge based on
    the letter of the alphabet labeling that edge and, respectively, to retrieve
    a state name from the letter labeling an edge.
    '''

    def __init__(self, name, outedges = [], state_prob = 0.0):
        self.name = name            #The state's name/label
        self.outedges = outedges    #List of outgoing edges
        self.state_prob = state_prob # Probability of the state in graph
        # An outgoing edge is a tuple composed of:
        #    (label, destination state, transition probability)
