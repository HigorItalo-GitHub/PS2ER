import Sequences_Analyser
import numpy as np
import CT_Functions
import Sequence_Generator
import Adjust_Contexts
from Eigenvector import occup_vector


def calc_H(alf, leafs_set, probs, probs_conds):
    H = 0

    for q in leafs_set:
        for sigma in alf:
            if f'{sigma}|{q}' in probs_conds[len(q)] and not probs_conds[len(q)][f'{sigma}|{q}'] == 0:
                # print("ok")
                H -= probs[len(q) - 1][q] * probs_conds[len(q)][sigma + '|' + q] * np.log2(
                    probs_conds[len(q)][sigma + '|' + q])

    return H


def split_H(Q_set, ALFABETO, PROBS, COND_PROBS):
    # print("\n*** SPLIT VIA REDUÇÃO DE ENTROPIA ***")
    # print("Conjunto de folhas de entrada:\n", Q_set)

    ent_concs = {}
    break_H, registred_h = None, None

    for estado in Q_set:

        actual_leafs = Q_set.copy()

        for e in actual_leafs:

            if e == estado:

                for symb in ALFABETO:
                    actual_leafs.append(symb + estado)

                actual_leafs.remove(e)

        # print("Conjunto de possives folhas a analisar:\n", actual_leafs)

        # P_est = PROBS[len(estado) - 1][estado]
        # print(f"P({estado}) = {P_est}")

        entropy_H = calc_H(ALFABETO, actual_leafs, PROBS, COND_PROBS)
        # print(f"H = {entropy_H}")
        ent_concs[entropy_H] = actual_leafs

        # print("dicionario criado:", ent_concs)
    # print("Conjuntos possiveis de folhas a analisar:")
    # for set in ent_concs.values():
    #     print(set)

    for h in ent_concs.keys():
        if h == min(ent_concs.keys()):
            # print(f"Conjunto de folhas que gera menor H ({h}):\n", ent_concs[h])
            registred_h = h
            break_H = ent_concs[h]

    # print("dicionario criado:", ent_concs)

    # print("menor entropia H calculada: ", registred_h)

    # print("\n=> Conjunto de folhas reajustado:\n", break_H)
    # return break_H
    return break_H, registred_h


def split_H_depthed(Q_set, depth, ALFABETO, PROBS, COND_PROBS):
    # print("\n*** SPLIT VIA REDUÇÃO DE ENTROPIA ***")
    # print("Conjunto de folhas de entrada:\n", Q_set)

    ent_concs = {}
    break_H, registred_h = None, None

    for estado in Q_set:

        actual_leafs = Q_set.copy()

        for e in actual_leafs:

            if e == estado and len(e) < depth:

                for symb in ALFABETO:
                    actual_leafs.append(symb + estado)

                actual_leafs.remove(e)

        # print("Conjunto de possives folhas a analisar:\n", actual_leafs)

        # P_est = PROBS[len(estado) - 1][estado]
        # print(f"P({estado}) = {P_est}")

        entropy_H = calc_H(ALFABETO, actual_leafs, PROBS, COND_PROBS)
        # print(f"Split do estado {estado}: H = {entropy_H}")
        # print(f"H = {entropy_H}")
        # if entropy_H in ent_concs:
        #     print(f"==> Mais de uma sequencia apresenta H = {entropy_H}")
        ent_concs[entropy_H] = actual_leafs

        # print("{Entropia : sequencias consideradas} -> ", ent_concs)

    # print("Conjuntos possiveis de folhas a analisar:")
    # for set_h, set in sorted(ent_concs.items()):
    #     print(f"{set_h}: {set}")
    # input("continue")

    for h in ent_concs.keys():
        if h == min(ent_concs.keys()):
            # print(f"Conjunto de folhas que gera menor H ({h}):\n", ent_concs[h])
            registred_h = h
            break_H = ent_concs[h]

    # print("dicionario criado:", ent_concs)

    # print("menor entropia H calculada: ", registred_h)

    # print("\n=> Conjunto de folhas gerados por split_H:\n", break_H)

    return break_H, registred_h


def dist_euclidiana(vector):
    import itertools
    import math

    x = list(itertools.combinations(vector, 2))
    dists = []
    for h in x:
        v1, v2 = np.array(h[0]), np.array(h[1])
        # print(v1, v2)
        diff = v1 - v2
        # print(diff)
        quad_dist = np.dot(diff, diff)
        # print(quad_dist)
        dists.append(round((math.sqrt(quad_dist)), 4))
        # print(f'    Distancia euclidiana entre {v1} e {v2}: %.3f' % math.sqrt(quad_dist))

    return dists[0]


if __name__ == "__main__":

    """Reading of the symbolic sequence MafaulDa, 
    previously symbolic via maximum entropy reduction over 
    binary symbolic alphabet, stored in .txt file"""
    system_name = "MaFaulDa_C3_MaxEnt"  # string to output file
    with open('MaFaulDa_Normal_C3_12288_BIN.txt', 'r') as f:
        for line in f:
            sequence = line
    sequence_lenght = len(sequence)

    """Call to modules that return probabilities of 
    symbolic sequences obtained from the original MaFaulDa symbolic series."""
    L = 35  # Limit variable for calculating symbolic sequences of length L
    probabilities, seq_alf = Sequences_Analyser.calc_probs(sequence, L)
    conditional_probabilities = Sequences_Analyser.calc_cond_probs(probabilities, seq_alf, L)

    Sigma = sorted(seq_alf)  # orders the alphabet of symbols associated with the analyzed sequence

    """Defining the maximum number of leafs (N_max) to be generated for the leafs set (Gamma)"""
    Gamma = ['']
    N_leafs = [18, 25, 42, 78, 122, 142, 178, 202, 228, 250]
    l_max = 9  # tree depth threshold - corresponds to the maximum length allowed for leaves

    save_maq = False  # Variable that enables storing the generated PFSA in a pickle file (.pkl)

    States = []     # stores the labels of the states of the PFSA to be generated
    Ents = []   # stores conditional entropies calculated for a PFSA obtained for the number of leaves defined in the variable 'Folhas'
    DivergsKL = []  # stores Kullback-Leibler divergences calculated for a PFSA obtained for the number of leaves defined in the variable 'Folhas'

    for N_max in N_leafs:
        while len(Gamma) < N_max:  #
            # folhas, h = split_H(folhas, alfabeto_da_sequencia, probabilidades, probabilidades_condicionais); model_name = "S2ER"
            Gamma, h = split_H_depthed(Gamma, l_max, Sigma, probabilities, conditional_probabilities)
            model_name = "PS2ER"

            # folhas = split_compair(folhas, alfabeto_da_sequencia, probabilidades_condicionais, 'euclid')

            Gamma = Adjust_Contexts.leafs_complementer(Gamma, Gamma, Sigma,
                                                       conditional_probabilities);
            model_name = "PS2ER"

            # print("\n==> Folhas geradas após quebras por redução de entropia + quebras anti-strands:\n", folhas)

            # if len(folhas) == len(alfabeto_da_sequencia) ** prof:
            #     print(f"\n!_!_!_!_Número máximo de folhas possivel para alfabeto de ordem {len(alfabeto_da_sequencia)} e profundidade"
            #           f" de arvore = {prof} atingido _!_!_!_!")
            #     break

        used_states_vI, actual_machine_vI, n_states_vI, generated_sequence = \
            CT_Functions.new_sequence_generator_v2(
                Gamma, Sigma, probabilities, conditional_probabilities, sequence_lenght)
        # expanded_Q_Splittings, alfabeto_da_sequencia, probabilidades, probabilidades_condicionais, sequence_lenght)

        new_sequence = generated_sequence
        # new_sequence = Sequence_Generator.generate_sequence(actual_machine_vI, sequence_lenght)

        # with open(f'MACHINE_MaFaulDa_Splitting_{len(Q_splitings)}.pkl', 'wb') as output:
        #     pickle.dump(actual_machine_vI, output, pickle.HIGHEST_PROTOCOL)

        # from Comparing_Morphs import calc_morphs_dists
        # registers = {}
        # lim = 0.0605
        # for state in actual_machine_vI.states:
        #     # print(state.name, state.outedges)
        #     for morph in state.outedges:
        #         if state.name + morph[0] == morph[1]:
        #             pass
        #         else:
        #             d = calc_morphs_dists([state.name + morph[0],morph[1]],alfabeto_da_sequencia,probabilidades_condicionais)
        #             if d >= lim:
        #                 if state.name not in registers:
        #                     registers[state.name] = [f"{morph} --> dist[{state.name + morph[0]},{morph[1]}] = {d}"]
        #                 else:
        #                     registers[state.name].append(f"{morph} --> dist[{state.name + morph[0]},{morph[1]}] = {d}")
        #
        # for k,v in registers.items():
        #     print(k,v)
        # print(f"\n===Quantidade de estados que apresentam morphs com distancia entre transições acima do limiar {lim}: {len(registers)}===")

        # input("continue")

        machine = actual_machine_vI
        counting_ins = {state.name: 0 for state in machine.states}
        for state in machine.states:
            for outedge in state.outedges:
                counting_ins[outedge[1]] += 1
        print("\n==> Nº de in-edges para cada um dos estados do PFSA gerado:")
        for k, v in counting_ins.items():
            print(f"{k}: {v}")

        """=============================================================================================================="""
        trans_mtx = np.zeros((len(actual_machine_vI.states), len(actual_machine_vI.states)))
        # print(trans_mtx)
        num_mach_sts = {st: num for num, st in enumerate(actual_machine_vI.states)}
        # for a,b in num_mach_sts.items():
        #     print(a.name,b)

        for stt, idx in num_mach_sts.items():
            i_row = idx
            i_col = None
            for oe in stt.outedges:
                # print(oe)
                stt_col = oe[1]
                for s_k, s_v in num_mach_sts.items():
                    if s_k.name == stt_col:
                        i_col = s_v
                # print("dest: ", i_col)
                # print("---")
                #     print(i_col)
                trans_mtx[i_row][i_col] = 1

        # for line in trans_mtx:
        #     print(line)
        # input("cont.")

        # trans_mtx = np.array([[0,0,1,1],[1,1,0,0],[0,0,1,1],[0,0,1,0]]) # teste

        verif_mtx = np.linalg.matrix_power(trans_mtx, len(actual_machine_vI.states))

        # for lin in verif_mtx:
        #     print(lin)
        # input("cont.")

        print("\nIndices de elementos iguais a 0 no array 'verif_mtx':")
        # res = np.where(verif_mtx == 0)[0]
        res = np.argwhere(verif_mtx == 0)
        print(res)
        # input("cont.")

        if save_maq:
            import pickle

            with open(f'{system_name}_MACHINE_{model_name}_N{n_states_vI}.pkl', 'wb') as output_maq:
                pickle.dump(actual_machine_vI, output_maq, pickle.HIGHEST_PROTOCOL)

        print("Vetor de ocupação téorico da máquina gerada:")
        for occ in occup_vector(actual_machine_vI):
            print(round(occ, 6))
        # input("CONT-->")

        """=============================================================================================================="""

        param = 10

        entropia_new_sequence = CT_Functions.entropy_analyser(new_sequence, param)

        probabilidades_new_sequence, alf_seq_new_sequence = Sequences_Analyser.calc_probs(new_sequence, param)

        divergKL_new_sequence = Sequences_Analyser.calc_kldivergence(probabilidades_new_sequence, probabilities, param)

        print("========================== RESULTADOS ===========================\n")
        entropia_original_seq = CT_Functions.entropy_analyser(sequence, param)
        print(f"Entropia h{param} sistema original:\n", entropia_original_seq[-1])

        print("\nTotal de estados do PFSA gerado:\n", len(used_states_vI))
        print(f"Entropia h{param} do modelo gerado:\n", entropia_new_sequence[-1])

        print(f"Divergencia Kullback-Leibler D{param} do modelo gerado:\n{divergKL_new_sequence}")

        States.append(len(used_states_vI))
        Ents.append(entropia_new_sequence[-1])
        DivergsKL.append(divergKL_new_sequence)

        Gamma = ['']

    print(f"Entropia h{param} sistema original:\n", entropia_original_seq[-1])

    print("\nEstados dos PFSAs gerados:\n", States)
    print(f"Entropias h{param} dos modelos gerados:\n", Ents)
    print(f"Divergencias Kullback-Leibler D{param} dos modelos gerados:\n{DivergsKL}")

    # num_folhas = []
    # regists_H = []
    #
    # reg_H = 1.0
    # folhas = ['']
    # num_f = 23
    #
    # while reg_H > 0.1231:
    # # while len(folhas) <= num_f:
    #
    #     folhas, reg_H = split_H(folhas, alfabeto_da_sequencia, probabilidades, probabilidades_condicionais)
    #     # if len(folhas) == num_f:
    #     print(len(folhas), folhas)
    #     num_folhas.append(len(folhas))
    #     regists_H.append(reg_H)
    #
    #     # folhas = split_compair(folhas, alfabeto_da_sequencia, probabilidades_condicionais, 'euclid')
    #
    # print(num_folhas)
    # print(regists_H)
