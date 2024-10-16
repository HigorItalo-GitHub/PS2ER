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
        #An outgoing edge is a 2-tuple composed of:
        #(label, destination state)

    '''
    Input: letter from the graph's alphabet
    Output: Destination from the edge containing the input letter as label.
    '''
    def next_state_from_edge(self, label):
        #Finds and returns a state name from outedges based on a letter.
        match = [x[1] for x in self.outedges if x[0] == label]
        if match:
            return match[0]
        else:
            return None

    '''
    Input: Destination state's name.
    Output: Label from the edge that goes to the desired state.
    '''
    def edge_leads_to_state(self, state_name):
        #Finds and returns an edge label from outedges based on a state name.
        match = [x[0] for x in self.outedges if x[1].name == state_name][0]
        return match
    '''
    Input:
    Output: Returns all the destinations from all outgoing edges.
    '''
    def obtain_children(self):
        children = [x[1] for x in self.outedges if x[1]]
        return children

    '''
    Input:
    Output: Returns the length of the state's name. It is redefined in order to
    return length 0 for the empty string.
    '''
    def name_length(self):
        if self.name == 'e':
            return 0
        else:
            return len(self.name)

    # def __str__(self):
    #     r = '*************\n'
    #     r += 'State name: ' + self.name + '\n'
    #     r += 'Outedges:\n'
    #     for edge in self.outedges:
    #         r += "-----\n"
    #         i = 0
    #         for e in edge:
    #             if i == 0:
    #                 r += "Edge with label: " + e + '\n'
    #             elif i == 1:
    #                 if e:
    #                     r += "To: " + e.name + '\n'
    #                 else:
    #                     r += "To nowhere \n"
    #             else:
    #                 r += str(e) + '\n'
    #             i += 1
    #     r += "\n\n"
    #     return r

    def serialize(self):
        serial_edges = []
        for edge in self.outedges:
            i = 0
            s_edge = []
            for e in edge:
                if i == 1:
                    if e:
                        s_edge.append(e.name)
                    else:
                        s_edge.append('')
                else:
                    s_edge.append(e)
                i += 1
            serial_edges.append(s_edge)
        serial_state = [self.name, serial_edges]
        return serial_state
