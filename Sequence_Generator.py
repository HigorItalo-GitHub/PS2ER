# import yaml
import numpy as np
import random
import Sequences_Analyser as sa
# from timeit import default_timer as timer

'''
Name: generate_sequence
Input:
    *machine: DMarkov machine from which the sequence will be created.
    *D: memory of the machine.
    *L: sequence size.
Output:
    *sequence: sequence with size L obtained from machine with memory D.
Description:
    Starts with an predetermined state (all zeroes) and iterates L times,
    choosing next state in accord with labels probabilities.
'''


def generate_sequence(machine, L, label_size=1):

    sequence = ''

    print("\n-*-*-*-*-*-* REALIZANDO ANÁLISE DE OCUPAÇÕES DE ESTADOS DO PFSA -*-*-*-*-*-*-*-*-*")

    vetor_de_ocupacao = sa.calc_occup_vector_V2(machine, L)

    print("\n##### Vetor de ocupação dos estados do PFSA: #####")

    set_inital_state = False

    if set_inital_state:
        for k, v in vetor_de_ocupacao.items():
            print(f"'{k}': {v},")
            # if v == max(vetor_de_ocupacao.values()):
            #     curr_state_name = k
        curr_state_name = random.choices([seq for seq in vetor_de_ocupacao.keys()], [occup for occup in vetor_de_ocupacao.values()])[0]
        print("Estado inicial selecionado para geração da sequência experimental: ", curr_state_name)
    else:
        # Starts in machine's first state
        curr_state_name = machine.states[0].name
        print("Estado inicial adotado para geração da sequência experimental: ", curr_state_name)

    for state in machine.states:
        if (state.name == curr_state_name):
            curr_state = state

    # Generate a L length sequence from states of the PFSA
    for i in range(int(L/label_size)):

        # Set data parameters
        labels = [outedge[0] for outedge in curr_state.outedges]
        probabilities = [outedge[-1] for outedge in curr_state.outedges]

        # Weight formatting
        probabilities = [int(p * 10e16) for p in probabilities]

        # Chooses next state
        label = random.choices(labels, probabilities)[0]
        # print(f'Labe = {label}')

        sequence = sequence + label

        # Goes to next state
        curr_state_name = [outedge[1] for outedge in curr_state.outedges if \
                            outedge[0] == label][0]
        for state in machine.states:
            if (state.name == curr_state_name):
                curr_state = state

    return sequence


# thinking machine as a dict {state: outedge}
def generate_sequence_dict(machine, L):
    sequence = ''
    # Starts in machine's first state
    curr_state_name = list(machine.keys())[0]

    #Generate a L length sequence from DMarkov with D = curr_d
    for i in range(L):
        # Set data parameters
        labels = [oedge[0] for oedge in machine[curr_state_name]]
        probabilities = [oedge[-1] for oedge in machine[curr_state_name]]
        # Weight formatting
        probabilities = [int(p * 10e16) for p in probabilities]
        # Chooses next state
        label = random.choices(labels, probabilities)[0]
        # print(f'Label = {label}')
        sequence = sequence + label
        # Goes to next state
        curr_state_name = [oedge[1] for oedge in machine[curr_state_name] if oedge[0] == label[0]][0]
    return sequence


def logistic_map(lenght, x0 = 0.5, r = 3.75):
     x = [x0]
     s = ''
     for i in range(lenght):
             x.append(r*x[i]*(1-x[i]))
             if x[i] <= 0.67:
                     s += '0'
             elif x[i] <= 0.79:
                     s += '1'
             else:
                     s += '2'
     return s

if __name__ == "__main__":

    L = 10000000

    s = logistic_map(L)

    print("Sequencia gerada:")
    print(f"{s[0:150]}...")
    print("numero de caracteres na sequencia:",len(s))

    #arq=open("sequence.txt","w")
    #arq.write(s)
    #arq.close()

    #arq1=open("sequence.txt","r")
    #arq.seek(0,0)
    #print(arq1.read())
    #print(arq1.tell())
    # Gera um arquivo no formato YAML com a sequencia gerada

    # with open('original_len_10000000.yaml', 'w') as f:
    #     yaml.dump(s, f)
