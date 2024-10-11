import numpy as np
# import pandas as pd
from Sequence_Generator import logistic_map
import random
# debug
from timeit import default_timer as timer

'''
Nome: calc_probs
Entradas:
 * X: sequência a ser analisada
 * L: comprimento máximo da subseqüência a ser analisado.
Saídas:
 * probabilities: uma lista de dicionários. Cada dicionário contém chaves que 
    são sequências do mesmo tamanho. O valor associado a uma chave é uma 
    probabilidade de que a subsequência apareça na sequência original
 * alphabet: os símbolos únicos que aparecem na sequência.
Descrição:
Verifica o número de ocorrências de subseqüências de comprimentos de 1 a L.
Divide o número de ocorrências pelo comprimento da sequência para 
    obter freqüências relativas. Cria um dicionário para subseqüências de 
    cada tamanho. Ao verificar subseqüências de comprimento 1, o método registra cada 
    símbolo individual que aparece e o armazena como o alfabeto da seqüência.
'''


def calc_probs(X, L):

    print("\n--- Executando cálculos de probabilidades ---\n")

    probabilities = []
    alphabet = set([c for c in X])  # computa todos os diferentes caracteres na sequencia
    max_probs = {}
    init_time = timer()
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

    print("Probabilidades calculadas!")
    return [probabilities, alphabet]

'''
Nome: calc_cond_probs
Entrada:
 * probabilities: lista de dicionários contendo todas as probabilidades de subseqüências em uma seqüência;
 * alphabet: conjunto de símbolos que aparecem em sub-seqüências de probabilidades;
 * L: comprimento máximo da subseqüência a ser analisado;
Saída:
 * conditional_probabilities: uma lista de dicionários. Cada dicionário contém chaves que são da forma:
                                        símbolo|subsequencia
 significando a probabilidade de "símbolo" ocorrer após essa subsequência.
 Existe um dicionário para cada tamanho de subsequência.
Descrição:
 Calcula a probabilidade de cada símbolo no alfabeto ocorrer em cada subseqüência em probabilidades e 
     cria um dicionário semelhante para essas probabilidades condicionais.
'''


def calc_cond_probs(probabilities, alphabet, L):

    print("\n --- Executando cálculos de probabilidades condicionais ---")

    # Saida inicializada como uma lista vazia:
    conditional_probabilities = []
    print("\nCalculando probabilidades condicionais de subsequência...")
    print("\nComprimento máximo da subsequência a ser analisado: L =", str(L))
    print("")
    if probabilities:
        # O primeiro elemento, isto é, as probabilidades de cada símbolo,
        # dada a cadeia vazia, são apenas as probabilidades da ocorrência
        # desses símbolos, isto é, o primeiro elemento da lista de probabilidades.
        conditional_probabilities = [probabilities[0]]
        # Este loop calcula as probabilidades condicionais de subsequancias de
        # comprimento maior que 0 dado cada simbolo no alfabeto:
        for l in range(0, L):
            # print("Calculando probabilidades condicioanis de subsequencias de comprimento: " + str(l + 1))
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
        print("Probabilidades não calculadas.")
        print("Execute a função calc_probs function primeiro antes de executar esta.")
    print("*****************")
    print("Probabilidades condicionais calculadas!")
    print("*****************")
    return conditional_probabilities


def calc_cond_entropy(probabilities, conditional_probabilities, L):
    cond_entropy = []
    print("Calculando entropia condicional para sequencias até: ")
    print("L = " + str(L))
    if probabilities:
        if conditional_probabilities:
            for l in range(0, L):
                # l corresponde ao numero de bits condicionais. Assim, para um certo l fixo, podemos calcular h_{l+1}.
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
            print("Probabilidades condicionais não calculadas.")
            print("Execute a função calc_cond_probs antes dessa.")
    else:
        print("Probabilidades não calculadas.")
        print("Execute a função calc_probs function antes dessa.")
    print("*****************")
    #print("Sequencia: ")
    print("Entropia condicional calculada!")
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
    print("Calculating Kullback-Leibler divergence for sequence at: ")
    print("K = " + str(K))
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
    print("Sequence test: ")
    print("Kullback-Leibler divergence calculated!")
    print("*****************")
    return kldivergence


def calc_kldivergence_vector(vec1, vec2):
    kldivergence_pq = 0
    kldivergence_qp = 0
    print("Calculating Kullback-Leibler divergence")
    if len(vec1) and len(vec1) == len(vec2):
        #Probabilities of subsequences of length K are stored in probabilities[K-1]
        for i in range(len(vec1)):
            p = vec1[i] or 1e-15
            q = vec2[i] or 1e-15
            # print(f'p={p}, q={q}')
            kldivergence_pq += p*np.log2(p/q)
            kldivergence_qp += q*np.log2(q/p)

            kldivergence_simetric = (kldivergence_pq + kldivergence_qp)/2
    else:
        print ("[error] Probabilities not computed.")
    print("*****************")
    print("Kullback-Leibler divergence calculated!")
    print("*****************")
    return kldivergence_simetric

'''
Nome: calc_occup_vector
Entrada:
 * machine: grupo de estados a serem analisados.
 * sequence: seqüência de símbolos usados nas transições de estados da máquina.
 * N: comprimento da subsequência da sequência usada na análise.
Saída:
 * occup_vector: vetor de ocupação da máquina obtida empiricamente.
Descrição:
 Calcula o Vetor de Ocupação da máquina executando transições de estado
 baseado em uma seqüência de símbolos. O número de visitações por estado é armazenado
 e depois é calculada a ocupação de cada estado.
 Atualmente suporta apenas labels de 1 tamanho.
'''


# def calc_occup_vector(machine, sequence, N):
#     for i in range(len(machine.states)):
#         states = machine.states
#         st_counter = [0 for j in machine.states]
#         st_index = i
#         erro = 0
#         for label in sequence[:N]:
#             cur_st = states[st_index]
#             cur_oedges = cur_st.outedges
#             for oedge in cur_oedges:
#                 if oedge[0] == label:
#                     st_index = oedge[1]
#                     break
#                 # decide action if label doesn't exist in current state
#             if not (oedge[0] == label):
#                 erro += 1
#                 st_index = cur_oedges[0][1]
#             st_counter[st_index] += 1
#         st_counter = np.array(st_counter)
#         # print(erro)
#         occup_vector = normalize(st_counter[:, np.newaxis], norm='l1', axis=0).ravel()
#         # print(occup_vector)
#         # print()
#         if not erro:
#             return occup_vector
#     return occup_vector


def calc_occup_vector_V2(machine, N):

    states = machine.states # recebe cada um dos esatdos da maquina

    curr_state = states[0] # aramazena o primeiro elemento do conjunto de estados da maquina

    idx = dict((s.name, states.index(s)) for s in states) # dicionario associando o nome do estado ao seu indice no conjunto de estados

    st_counter = np.zeros(len(states)) # cria uma lista de zeros na mesma quantidade de estados existentes

    for i in range(int(N)): # para uma itereção de N vezes

        # Set data parameters
        labels = [outedge[0] for outedge in curr_state.outedges] # armazenas as letras do alfabeto, contidas nos outedges

        probabilities = [outedge[-1] for outedge in curr_state.outedges] # armazenas a probabilidades de cada label
        # Weight formatting

        probabilities = [int(p * 10e16) for p in probabilities] # multiplica por 10e16 as probabilidades no outedge analisado
        # Chooses next state

        label = random.choices(labels, probabilities)[0] # armazena aleatoriamente um label do conjunto de labels
        # Goes to next state

        next_state_name = curr_state.name + label # armazena o estado seguinte a partir da concatenação do label aleatorio selecionado com o nome do estado atual
        #print(f"concatenação: {next_state_name}")

        while len(next_state_name) >= 1:
            if next_state_name in idx:
                curr_state = states[idx[next_state_name]]
                break
            else:
                next_state_name = next_state_name[1:]

        st_counter[idx[curr_state.name]] += 1
        #print(st_counter)

    occupation_vector = st_counter/st_counter.sum()

    occup_vect = {}

    for k1,v1 in idx.items():

        occup_vect[k1] = occupation_vector[v1]

    return occup_vect


def calc_euclidian_distance(seq_probs, base_probs, K):
    euclidian_distance = 0
    print('Calculating Euclidian Distance for sequence at: ')
    print('K={}'.format(K))

    if seq_probs:
        # Probabilities of subsequences of length K are stored in probabilities[K-1]
        for key in base_probs[K - 1].keys():
            p = base_probs[K - 1][key]
            if key in seq_probs[K - 1].keys():
                q = seq_probs[K - 1][key]

                if not q:
                    q = 0
            else:
                q = 0  # Default non-zero really small value

            euclidian_distance += abs(p - q)
    else:
        print("[error] Probabilities not computed.")
        print("Run calc_probs function before this one.")
    print("*****************")
    print("Sequence test: ")
    print("Euclidian Distance calculated!")
    print("*****************")
    return euclidian_distance


if __name__ == "__main__":
    print("testando função")

    lenght = 1000000
    x = logistic_map(lenght)
    l=8

    dic_subSeqs_probs, alph = calc_probs(x,l)

    # probs=[]
    print("\nProbabilidades calculadas:")
    for dsp in dic_subSeqs_probs:
        # probs.append(dsp)
        print("\n",dsp)

    # for i in probs[21:22]:
    #     print("\n",i.items())

    print("\nalfabeto da sequencia analisada:")
    print(alph)

    cond_probs = calc_cond_probs(dic_subSeqs_probs, alph,l)

    print(f"\nProbabilidade condicionais calculadas para L = {l}:")
    for i in cond_probs:
        print("\n",i)

    ents = calc_cond_entropy(dic_subSeqs_probs,cond_probs,8)
    # print(ents)
