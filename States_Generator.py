from Probabilistic_State import ProbabilisticState

"""Function used to manage the routine for creating data structures 
corresponding to the states of the PFSA model to be generated."""
def states_generator(states_data, probs_states_data):

    States = [] # list that stores each state with an object of the ProbalilisticState class

    print("\n ( ) ( ) ( ) RUNNING PFSA'S STATE CREATION ROUTINE ( ) ( ) ( )")

    for key,value in states_data.items():
        E = ProbabilisticState(key, value, probs_states_data[key])
        States.append(E)

    print("\n(=) (=) (=) PFSA'S STATE CREATION ROUTINE COMPLETED! (=) (=) (=)")
    return States
