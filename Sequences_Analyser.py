import numpy as np
import random

'''
Name: calc_probs
Inputs:
 * X: sequence to be analyzed
 * L: maximum length of the subsequence to be analyzed.
Outputs:
 * probabilities: a list of dictionaries. Each dictionary contains 
 keys that represent sequences of the same length. The value associated with 
 a key is the probability that the subsequence appears in the original sequence.
 * alphabet: the unique symbols that appear in the sequence.
Description:  
Checks the number of occurrences of subsequences with lengths from 1 to L.  
Divides the number of occurrences by the length of the sequence to obtain relative frequencies.  
Creates a dictionary for subsequences of each length. When checking subsequences of length 1, 
the method records each individual symbol that appears and stores it as the alphabet of the sequence.  
'''
def calc_probs(X, L):

    print("\n--- Performing probability calculations ---")

    probabilities = []
    alphabet = set([c for c in X])
    max_probs = {}
    for i in range(0, len(X) - (L + 1)):
        curr_word = ''.join(map(str, X[i:(i + (L + 1))]))
        if not curr_word in max_probs.keys():
            max_probs[curr_word] = 1
        else:
            max_probs[curr_word] += 1
    for key in max_probs.keys():
        # max_probs[key] /= float(len(X))
        max_probs[key] /= float(len(X) - (L))
    probabilities.insert(0, max_probs)

    for l in range(L + 1, 1, -1):
        # print(f'Calculando probabilidades para palavras com comprimento {l} ...')
        node_prob = {}
        aux_probs = max_probs.copy()

        for key in max_probs.keys():
            sub_word = key[0:-1]
            sub_prob = aux_probs.pop(key, 0)
            # print(f'word:{key}; curr_prob:{sub_prob}')
            for c in alphabet:
                comp_word = sub_word + str(c)
                sub_prob += aux_probs.pop(comp_word, 0)
                # print(f'\tword:{comp_word}; curr_prob:{sub_prob}')
            # print(f'\tsub_word:{sub_word}')
            if not sub_word in node_prob.keys():
                node_prob[sub_word] = sub_prob
        # print(node_prob)
        probabilities.insert(0, node_prob)
        max_probs = node_prob.copy()
    # print(f'took {timer()-init_time} secs')

    # print("calculated probabilities!")
    return [probabilities, alphabet]

'''
Name: calc_cond_probs
Inputs:
 * probabilities: list of dictionaries containing all the probabilities of subsequences in a sequence;
 * alphabet: set of symbols that appear in subsequences of probabilities;
 * L: maximum length of the subsequence to be analyzed;
Output:
 * conditional_probabilities: a list of dictionaries. Each dictionary contains keys in the form:  
                                        symbol|subsequence  
  representing the probability of "symbol" occurring after that subsequence.  
  There is a dictionary for each subsequence length.
Description:  
Calculates the probability of each symbol in the alphabet occurring in each subsequence in probabilities and  
creates a similar dictionary for these conditional probabilities.
'''


def calc_cond_probs(probabilities, alphabet, L):

    print("\n --- Performing conditional probability calculations ---")

    # Saida inicializada como uma lista vazia:
    conditional_probabilities = []
    # print("\nCalculating conditional subsequence probabilities...")
    # print("\nMaximum length of the subsequence to be analyzed: L =", str(L))
    print("")
    if probabilities:
        # The first element, that is, the probabilities of each symbol,
        # given the empty string, are simply the probabilities of the occurrence
        # of these symbols, which is the first element of the probabilities list.
        conditional_probabilities = [probabilities[0]]
        # This loop calculates the conditional probabilities of subsequences
        # longer than 0 given each symbol in the alphabet:
        for l in range(0, L):
            # print("Calculating conditional probabilities of length subsequences: " + str(l + 1))
            d = {}
            l1 = probabilities[l]
            l2 = probabilities[l + 1]
            for s in l1:
                for a in alphabet:
                    cond = a + "|" + s
                    t = s + a
                    if t in l2.keys():
                        d[cond] = l2[t] / l1[s]
                    else:
                        d[cond] = 0.0
            conditional_probabilities.append(d)
    else:
        print("Uncalculated probabilities.")
        print("Please, certify that calc_probs function be executed first before running this one.")
    # print("*****************")
    # print("conditional probability calculated!")
    # print("*****************")
    return conditional_probabilities

'''
Name: calc_cond_entropy
Input:
    * probabilities: list of dictionaries containing probabilities for subsequences in a sequence
    * conditional_probabilities: list of dictionaries containing conditional probabilities for subsequences in a sequence
    * L: maximum length of subsequences to be analyzed
     
Output:
    * cond_entropy: list of conditional entropy values for subsequences of each length from 1 to L
     
Description:
    * Calculates the conditional entropy for subsequences of different lengths up to L.
    For each subsequence length, computes the entropy based on the probabilities and conditional probabilities.
    Uses the formula H(X|Y) = - Σ p(x) log2(p(x|y)) for conditional entropy.
    If L = 1, the entropy is calculated using the unconditional probabilities of the symbols.
    Stores and returns the entropy values for each subsequence length.
'''


def calc_cond_entropy(probabilities, conditional_probabilities, L):
    cond_entropy = []
    # print("Calculating conditional entropy for sequences up to: ")
    # print("L = " + str(L))
    if probabilities:
        if conditional_probabilities:
            for l in range(0, L):
                # l corresponds to the number of conditional bits. Thus, for a certain fixed l, we can calculate h_{l+1}.
                # print("Sequence: ")
                # print("Calculating conditional entropy of length: " + str(l+1))
                acc = 0
                p = probabilities[l]
                pcond = conditional_probabilities[l]
                for x in p.keys():
                    # if len(x) == L-1:
                    #     print('x=' + x)
                    if l == 0:
                        acc -= p[x] * np.log2(p[x])
                    else:
                        y_given_x = x[-1] + '|' + x[0:-1]
                        if not pcond[y_given_x] == 0:
                            # if x in list(p.keys())[-1]:
                            #     print(f'Computando P({x})*P({x[-1]}|{x[0:-1]})')
                            #     print(f'{p[x]} * {np.log2(pcond[y_given_x])}')
                            acc -= p[x] * np.log2(pcond[y_given_x])
                        else:
                            print(f'P({x[-1]}|{x[0:-1]}) nula')
                cond_entropy.append(acc)
                # print('\n')
        else:
            print("Uncalculated conditional probabilities.")
            print("Please certify that calc_cond_probs function be executed first before running this one.")
    else:
        print("Uncalculated probabilities.")
        print("Please certify that calc_probs function be executed first before running this one.")
    print("*****************")
    print("Conditional entropy calculated!")
    print("*****************")
    return cond_entropy


'''
Name: calc_kldivergence
Input:
    *base_probs: A list of probability dictionaries to which the
        probabilities contained in this class will be compared.
    *K: The length/level of probabilities from each that will be compared.
Output:
    *kldivergence: The Kullback-Leibler Divergence between the probability
        distributions of sequences of length K from base_probs and
        probabilities.
Description:
    Calculates the KL Divergence of prob distributions of K-length seqs.
'''


def calc_kldivergence(seq_probs, base_probs, K):
    kldivergence = 0
    # print(f"\nCalculating Kullback-Leibler divergence for sequences of lenght L={K}...\n")
    # print("K = " + str(K))
    if seq_probs:
        # Probabilities of subsequences of length K are stored in probabilities[K-1]
        for key in base_probs[K - 1].keys():
            p = base_probs[K - 1][key]
            if key in seq_probs[K - 1].keys():
                q = seq_probs[K - 1][key]

                if not q:
                    q = 1e-15
            else:
                q = 1e-15  # Default non-zero really small value

            kldivergence += p * np.log2(p / q)
    else:
        print("[error] Probabilities not computed.")
        print("Run calc_probs function before this one.")
    print("*****************")
    print("Kullback-Leibler divergence calculated!")
    print("*****************")
    return kldivergence

'''
Here is the translation into English:

---
*Name: calc_occup_vector_v2  
Inputs: 
* machine**: group of states to be analyzed   
* N: length of the subsequence from the sequence used in the analysis  

Output:  
* occup_vector**: empirically obtained occupation vector of the machine  

*Description:  
Calculates the machine's Occupation Vector by performing state transitions  
based on states outedges. The number of visits per state is stored,  
and then the occupation of each state is calculated.  
'''
def calc_occup_vector_V2(machine, N):

    states = machine.states # receives each of the machine's states

    curr_state = states[0] # stores the first element of the set of machine states

    idx = dict((s.name, states.index(s)) for s in states) # dictionary associating the name of the state with its index in the set of states

    st_counter = np.zeros(len(states)) # creates a list of zeros in the same number of existing states

    for i in range(int(N)): # for an iteration of N times

        # Set data parameters
        labels = [outedge[0] for outedge in curr_state.outedges] # stores the letters of the alphabet, contained in the outedges

        probabilities = [outedge[-1] for outedge in curr_state.outedges] # store the probabilities of each label
        
        probabilities = [int(p * 10e16) for p in probabilities] # Weight formatting: Multiplies the probabilities at the analyzed outedge by 10e16

        label = random.choices(labels, probabilities)[0] # Chooses next state: randomly stores a label from the set of labels

        next_state_name = curr_state.name + label # Goes to next state: stores the next state from the concatenation of the selected random label with the name of the current state

        # Search for the destination state with a label corresponding to the concatenation to define it as the current state in the occupation process
        while len(next_state_name) >= 1:
            if next_state_name in idx:
                curr_state = states[idx[next_state_name]]
                break
            else:
                next_state_name = next_state_name[1:]

        st_counter[idx[curr_state.name]] += 1    # Counts the visitation of the current state observed

    occupation_vector = st_counter/st_counter.sum()    # Calculates the occupancy of each state of the given machine, arranged in a vector

    occup_vect = {}

    for k1,v1 in idx.items():

        occup_vect[k1] = occupation_vector[v1]    # Dictionary containing the key (state label) and value (its occupancy calculation)

    return occup_vect
