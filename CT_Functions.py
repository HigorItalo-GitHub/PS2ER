from States_Data import States_data_generator
from States_Generator import states_generator
from Probabilistic_Graph import ProbabilisticGraph
from Sequence_Generator import generate_sequence
import Sequences_Analyser as sa


def entropy_analyser(sequence, L_ent):
    """ Function that receives a sequence of characters and
    returns a list with the entropy values of subsequences of size up to L_ent"""

    # print(f"\n---PERFOMRMING CONDITIONAL ENTROPY CALCULATION FOR SEQUENCIES OF LENGHT L={L_ent}---")

    # Calculating probabilities of sequences of length 'L_ent' associated with the 'sequence'
    p, a = sa.calc_probs(sequence, L_ent)

    # Calculating conditional probabilities of sequences of length 'L_ent' associated with the 'sequence'
    p_cond = sa.calc_cond_probs(p, a, L_ent)

    # Using probabilistic data to calculate the vector containing entropies up to size 'L_ent'
    condit_ent = sa.calc_cond_entropy(p, p_cond, L_ent)

    trunc_values = []
    for h_a in condit_ent:
        trunc_values.append(round(h_a, 4))

    # print("\n---CONDITIONAL ENTROPIES CALCULATED---")

    return trunc_values


def occup_vector_analyser(machine, tam):
    """ Function that calculates the state occupancy vector originating from
    a state machine for a given sequence size 'tam' """

    print("\n-*-*-*-*-*-* STATE OCCUPATION ANALYSIS ROUTINE BEING EXECUTED -*-*-*-*-*-*-*-*-*")

    occupation_vector = sa.calc_occup_vector_V2(machine, tam)

    occup_ctxs = []

    print("\n##### Current occupancy vector: #####")

    for k, v in occupation_vector.items():
        print(f"'{k}': {v},")

def new_sequence_generator_v2(LEAFS, alph, probs, probs_conds, seq_lenght):

    sts_prob = {}

    data_for_states_generation = States_data_generator(LEAFS, alph, probs_conds)

    for k in data_for_states_generation.keys():
        probs_data = probs[len(k)-1]
        if k in probs_data:
            sts_prob[k]=probs_data[k]
        else:
            print("\nState does not have calculated probability. Adopting null value")
            sts_prob[k] = 0.0

    states = states_generator(data_for_states_generation, sts_prob)  # list of objects of the 'States' class, each with a name and limits

    generated_machine = ProbabilisticGraph(states, alph)

    number_of_states = len(generated_machine.states)

    print(f"\nNumber of PFSA states: {number_of_states}")

    # print(f"\n------------------PROBABILITY OF SEQUENCES ASSOCIATED WITH MACHINE STATE LABELS-----------------------------")
    #
    # for i in generated_machine.states:
    #     print(f"{i.name}: {round(sts_prob[i.name], 6)}")

    if seq_lenght:

        print("\nGenerating experimental sequence from created PFSA...")

        generated_sequence = generate_sequence(generated_machine, seq_lenght)

        print(f"\nExperimental sequence generated:\n{generated_sequence[0:100]}...")

        return states, generated_machine, number_of_states, generated_sequence

    else:

        return states, generated_machine, number_of_states
