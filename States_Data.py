from States_Analyser import states_analyser


def States_data_generator(actual_leafs, alf, probs_conds):

    def leafs_complementer(Actual_Leafs, Leafs, Alphabet, probs_conds):

        # creates a dictionary with elements in the form
        # {context: [(symbopl of the alphabet,
        #             subsequent context,
        #             conditional prob. of the context associated with the letter)]}

        print("\n-*-*- STATE'S DATA CREATION ROUTINE RUNNING -*-*-")

        # states=[]

        # dic = {}  # stores the final dictionary that contains the data for each generated edge

        # outs = []

        new_ctx = []

        for f in Actual_Leafs:  # for each leaf in the currently analyzed leaf set...

            # pbs = cop[f]

            # print(F"\nANALYSING q = {f}:")

            for symbol in Alphabet:  # ...take each of the symbols in the alphabet in the sequence...

                next = f + symbol  # ... and contaminate the leaf with the symbol on the left, creating the 'next' context.

                cont = 0  # variable to account for non-identification of leafs as a suffix for 'next'

                for context in Leafs:  # For each of the contexts (leaves) in the original set of leaves analyzed...

                    if next.endswith(context):    # ...if any leaf is identified as suffix of the 'next' concatenation...

                        pass  # just register identification.

                        # print(f"EXTENSION {next} --> q = {context} IDENTIFIED AS TRANSITION TO {next}")

                    else:  # If no leaf is identified as a suffix for 'next'...

                        cont += 1  # ...iterate the counter.

                if cont == len(Leafs):  # If the value in 'cont' in the round is equal to the total number of original leafs...

                    # print(f"\nEXTENSION {next} --> No leaf was identified among the current ones as a suffix for this context. "
                    #       f"Stored for addition to the current sheet set.")

                    new_ctx.append(next)  # store 'next' in 'new_ctx' to be added to the original sheets

            # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        if new_ctx:  # If there is any context stored in 'new_ctx', perform a new analysis of the leaves...

            Leafs += new_ctx  # ...add the element contained in 'new_ctx' to the current sheets...

            # print(f"\nAdding {new_ctx} context(s) to current sheets...")

            print(f"\nNew set of leafs to analyze: {Leafs}")

            leafs_complementer(new_ctx, Leafs, Alphabet, probs_conds)  # ...and re-execute this function.

        return Leafs  # RETURN THE LEAFS PROPERLY COMPLETED

    def leafs_probs_data(Leafs, alphabet, conditional_probs):

        lpd = {}  # Dictionary to store the conditional probabilities of analyzed contexts

        for f in Leafs:

            adendo = {}  # Auxiliary variable to store the conditional probabilities taken in the current cycle

            # In the list of probab dictionaries. condition. corresponding to the context (sheet) 'f' currently analyzed...
            for k1,v1 in conditional_probs[len(f)].items():

                if k1.endswith(f): #...if the key in the dic. ends with context 'f'...

                    adendo[k1] = v1  # ...store this prob. conditional in 'addendum'.

            if not adendo:  # if 'addendum' did not receive any data (i.e. 'f' does not have calculated prob. cond.)...

                for l in alphabet:

                    adendo[l + "|" + f] = 0.0  # ... add 'f' in addendum with prob. cond. equal to zero.

                lpd[f] = adendo  # 'lpd' receives 'addendum', thus storing the prob data. cond. from context 'f'

            else:  # the data that has already been added in 'addendum'...

                lpd[f] = adendo  # ...are passed to 'lpd'.

        return lpd  # RETURN THE DICTIONARY THAT CONTAINS EACH LEAF (CONTEXT) WITH ITS RESPECTIVE CONDITIONAL PROBABILITY

    def generate_states_data(Leaves, alphabet, COP):

        states_data = {}  # dictionary that will store the useful states for building a state machine

        outs = []  # auxiliary variable for storing tuples that correspond to the outedges of a state

        for l in Leaves:  # For each element 'l' in the set of contexts (complemented leaves)...

            pbs = COP[l]  # ... we take the context and its outedges data present in the 'COP' data dictionary...

            for symb in alphabet:  # ...and for each symbol of the alphabet in the sequence...

                next = l + symb  # ...generate 'next' by connecting 'l' with 'letter' on the left...

                sufix = []

                for element in Leaves:

                    if next.endswith(element):    # if any of the leaves is a suffix of the concatenation currently analyzed...

                        # print(f"LEAF {element} IDENTIFIED AS SUFFIX TO {next}")

                        sufix.append(element)  #...store such leaf in 'suffix'

                #print("---")

                '''In this section of the code, 
                the longest context that is the suffix of the concatenation currently analyzed is selected.'''

                if len(sufix) > 1:  # if there is only one item in 'sufix' (only 'element' is suffix for 'next'...

                    lens = [len(s) for s in sufix]  #...stores context lengths in 'suffix'...

                    max_ctx = sufix[lens.index(max(lens))] #  ...and take the element with the longest length.

                    out_prob = pbs[symb+ "|" + l]  # take the conditional prob. of the context analyzed in this cycle...

                    if out_prob != 0:  #... and if not null...
                        outs.append((symb, max_ctx, out_prob))  # ...it is used to compose the outedge tuple

                else:  # if there is only one item in 'suffix', its data is taken to compose the tuple of its outedge.
                    out_prob = pbs[symb + "|" + l]
                    if out_prob != 0:
                        outs.append((symb, sufix[0], out_prob))


            states_data[l] = outs  # 'states_data' is then populated with state 'l' and its 'outs' outedge tuples

            outs=[]  # 'outs' is reset to execute a new analysis cycle.

        return states_data  # RETURN THE DICTIONARY COMPOSED BY THE SET OF INITIAL STATES GENERATED WITH THE LEAVES

    leafs_comp = leafs_complementer(actual_leafs, actual_leafs, alf, probs_conds)  # Complements the initial pages

    l_p_d = leafs_probs_data(leafs_comp, alf, probs_conds)  # get the conditional prob. of the complemented leaves

    sts_dt = generate_states_data(leafs_comp, alf, l_p_d)  # organizes context data in the form of machine states

    usefull_states = states_analyser(sts_dt)  # refines the states, removing strandeds

    print("\n*-*-* STATE'S DATA CREATION ROUTINE COMPLETED! *-*-*")

    return usefull_states  # RETURN USEFUL DATA TO STATE MACHINE GENERATION
