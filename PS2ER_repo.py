import Sequences_Analyser
import numpy as np
import CT_Functions
import Adjust_Contexts


"""The 'calc_H' function implements the calculation of the 
Entropy function H, based on the probability data associated with the 
sequences corresponding to the sequencies in 'leafs_set'"""
def calc_H(alphabet, leafs_set, probs, probs_conds):
    H = 0

    for q in leafs_set:
        for sigma in alphabet:
            if f'{sigma}|{q}' in probs_conds[len(q)] and not probs_conds[len(q)][f'{sigma}|{q}'] == 0:
                # print("ok")
                H -= probs[len(q) - 1][q] * probs_conds[len(q)][sigma + '|' + q] * np.log2(
                    probs_conds[len(q)][sigma + '|' + q])

    return H

"""The 'split_H_depthed' function is responsible for 
governing the extension of nodes, until max lenght 'depth', in the context tree, 
referring to Step 1 of the PS2ER algorithm."""
def split_H_depthed(Q_set, depth, ALPHABET, PROBS, COND_PROBS):

    ent_concs = {}
    break_H, registred_h = None, None

    for estado in Q_set:

        actual_leafs = Q_set.copy()

        for e in actual_leafs:

            if e == estado and len(e) < depth:

                for symb in ALPHABET:
                    actual_leafs.append(symb + estado)

                actual_leafs.remove(e)

        entropy_H = calc_H(ALPHABET, actual_leafs, PROBS, COND_PROBS)

        ent_concs[entropy_H] = actual_leafs

    for h in ent_concs.keys():
        if h == min(ent_concs.keys()):
            registred_h = h
            break_H = ent_concs[h]

    return break_H, registred_h


if __name__ == "__main__":

    """Reading of the symbolic sequence MafaulDa, 
    previously symbolic via maximum entropy reduction over 
    binary symbolic alphabet, stored in .txt file"""
    system_name = "MaFaulDa_C3_MaxEnt"  # string to output file
    with open('MaFaulDa_Normal_C3_12288_BIN.txt', 'r') as f:
        for line in f:
            sequence = line
    sequence_lenght = len(sequence)

    print(f"\nOriginal symbolic sequence:\n{sequence[0:100]}...")

    """Call to modules that return probabilities of 
    symbolic sequences obtained from the original MaFaulDa symbolic series."""
    L = 35  # Limit variable for calculating symbolic sequences of length L
    probabilities, seq_alf = Sequences_Analyser.calc_probs(sequence, L)
    conditional_probabilities = Sequences_Analyser.calc_cond_probs(probabilities, seq_alf, L)

    Sigma = sorted(seq_alf)  # orders the alphabet of symbols associated with the analyzed sequence

    """Defining the maximum number of leafs (N_max) to be generated for the leafs set (Gamma)"""
    Gamma = ['']    # Initialization of the Gamma leaf set with the empty word (root of the context tree)
    N_leafs = [18, 25, 42, 78, 122, 142, 178, 202, 228, 250]    # 'N_max' values used do generate multiple PS2ER models.
    l_max = 9  # tree depth threshold - corresponds to the maximum length allowed for leaves

    save_maq = False  # Variable that enables storing the generated PFSA in a pickle file (.pkl)

    param = 10  # Parameter adopted do quantifiers performance calculation
    original_seq_entropy = None # variable for storing the conditional entropy reference value calculated for the analyzed system
    States = []     # stores the labels of the states of the PFSA to be generated
    Ents = []       # stores conditional entropies calculated for a PFSA obtained for the number of leaves defined in the variable 'Folhas'
    DivergsKL = []  # stores Kullback-Leibler divergences calculated for a PFSA obtained for the number of leaves defined in the variable 'Folhas'

    model_name = "PS2ER"    # label used do name some archive created

    for N_max in N_leafs:
        while len(Gamma) < N_max:   # Loop executed until the maximum number of leaves determined, 'N_max'

            """The call to the 'split_H_depthed' function at this point performs 
            the extension of nodes of the generated context-tree until a max depth 'l_max', 
            corresponding to the action performed in Step 1 of the PS2ER algorithm."""
            Gamma, h = split_H_depthed(Gamma, l_max, Sigma, probabilities, conditional_probabilities)

            """The call to the 'leafs_complementer' function of 'Adjust_Context' auxiliar module at this point performs 
            the suffix identification action for each of the leaves in the current Gamma set, 
            corresponding to the action performed in Step 2 of the PS2ER algorithm"""
            Gamma = Adjust_Contexts.leafs_complementer(Gamma, Gamma, Sigma, conditional_probabilities)

        """Once the set of adjusted leaves has been defined (with the completion of the previous loop), 
        the set of leaves is used at this point to construct the states in the PFSA model, 
        corresponding to that performed in Step 3 of the PS2ER algorithm."""
        numbr_of_model_states, generated_PFSA, n_states, generated_sequence = \
            CT_Functions.new_sequence_generator_v2(
                Gamma, Sigma, probabilities, conditional_probabilities, sequence_lenght)

        new_sequence = generated_sequence

        if save_maq:    # allow save the PFSA model in a .pkl archive, if 'save_maq' is setted as 'True'
            import pickle
            with open(f'{system_name}_MACHINE_{model_name}_N{n_states}.pkl', 'wb') as output_maq:
                pickle.dump(generated_PFSA, output_maq, pickle.HIGHEST_PROTOCOL)

        """vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv QUANTIFIERS PERFORMANCE CALCULATION vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"""

        print(f"\n---PERFOMRMING CONDITIONAL ENTROPY CALCULATION FROM GENERATED SEQUENCE FOR WORDS OF LENGHT L={param}---")
        generated_sequence_entropy = CT_Functions.entropy_analyser(new_sequence, param)

        print(f"\n---PERFOMRMING KULLBACK-LEIBLER DIVERGENCE CALCULATION FROM GENERATED SEQUENCE FOR WORDS OF LENGHT L={param}---")
        generated_sequence_probabilities, alf_seq_new_sequence = Sequences_Analyser.calc_probs(new_sequence, param)
        generated_sequence_divergKL = Sequences_Analyser.calc_kldivergence(generated_sequence_probabilities, probabilities, param)

        print(f"\n---PERFOMRMING CONDITIONAL ENTROPY CALCULATION FROM ORIGINAL SEQUENCE FOR WORDS OF LENGHT L={param}---")
        original_seq_entropy = CT_Functions.entropy_analyser(sequence, param)

        # print(f"Coditional Entropy h{param} of system:\n", original_seq_entropy[-1])

        # print("\nTotal of states in generated PFSA:\n", len(used_states_vI))
        # print(f"Entropy h{param} of generated model:\n", generated_sequence_entropy[-1])

        # print(f"Kullback-Leibler Divergence D{param} of generated model:\n{generated_sequence_divergKL}")

        print("\n========================== MODEL ANALYSES CONCLUDED! ===========================\n")

        """ storing generated model data """
        States.append(len(numbr_of_model_states))          # store number os states of the generated PFSA
        Ents.append(generated_sequence_entropy[-1]) # store conditional entropy for given 'param' of the generated PFSA
        DivergsKL.append(generated_sequence_divergKL)   # store Kullback-Leibler divergence for given 'param' of the generated PFSA

        Gamma = ['']

    """The results presented below can be used to 
    plot graphs of the quantifiers adopted versus the number of states of the PS2ER model generated."""

    print("\n==========================================================\n"
          "========================= RESULTS ========================\n"
          "==========================================================\n")

    print(f"Entropy h{param} from original system (baseline):\n", original_seq_entropy[-1])
    print("Number of states:\n", States)
    print(f"Entropy h{param}:\n", Ents)
    print(f"Kullback-Leibler D{param}:\n{DivergsKL}")
