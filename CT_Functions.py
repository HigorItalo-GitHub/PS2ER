from States_Data import States_data_generator
from States_Generator import states_generator
from Probabilistic_Graph import ProbabilisticGraph
from Sequence_Generator import generate_sequence
import Sequences_Analyser as sa
import random


def truncar(numero, casas_decimais=4):
    """Função que recebe uma variavel float 'numero' e a
    Trunca / preenche para n casas decimais sem arredondamento,
    retornado o resultado do truncamento/preenchimento"""

    s = '{}'.format(numero)

    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(numero, casas_decimais)

    i, p, d = s.partition('.')

    return '.'.join([i, (d + '0' * casas_decimais)[:casas_decimais]])


def random_word_creator():
    rw = ''

    for i in range(3):
        rw += str(random.randrange(0, 100))

    return rw


def text_files_manager(text, archiv_name, mode='a'):

    # criando e escrevendo arquivos de texto (modo 'w')
    if mode == 'w':
        with open(f'{archiv_name}.txt', 'w') as f:
            f.write(text)
        print("\nArquivo criado.")

    # lendo o arquivo criado
    elif mode == 'r':
        with open(f'{archiv_name}.txt', 'r') as f:
            print("Conteudo atual do arquivo:\n")
            for linha in f:
                linha = linha.rstrip()
                print(linha)

    # acrescentendo texto ao arquivo (modo 'a')
    elif mode == 'a':
        # print("\n")
        # texto = input("Informe texto a ser inserido no arquivo:\n")

        with open(f'{archiv_name}.txt', 'a') as f:
            f.write(text + "\n")
            # f.write(texto + "\n")
            # print("operação concluida no arquivo " + f.name + " usando o mode de acesso " + f.mode)

        # print("Texto editado:")
        # with open(f'{archiv_name}.txt', 'r') as f:
        #     for linha in f:
        #         linha = linha.rstrip()
        #         print(linha)
    else:
        print("\nModo de operação invalido. Reexecute o programa informando uma modo valido.")


def entropy_analyser(sequence, L_ent):
    """ Função que recebe uma sequencia de carateres e
     retorna uma lista com os valores de entropias de subsequencias de tamanho até L_ent"""

    print("\n---ANÁLISE DE ENTROPIAS INICIADA---")

    # Calculando probabilidades associadas a sequencia
    p, a = sa.calc_probs(sequence, L_ent)

    p_cond = sa.calc_cond_probs(p, a, L_ent)

    # Utilizando os dados probabilisticos para calcular o vetor contendo entropias até tamanho L
    condit_ent = sa.calc_cond_entropy(p, p_cond, L_ent)

    # utilizando função truncar() para exibir valores de entropias no vetor com menos casas decimais

    trunc_values = []
    for h_a in condit_ent:
        trunc_values.append(round(h_a,4))

    print("\n---ANÁLISE DE ENTROPIAS FINALIZADA---")

    # print(f'Entropia calculada para L = {L_seq}:\n{trunc_values}')

    return trunc_values


def occup_vector_analyser(machine, tam):
    """ Função que calcula o vetor de ocupação de estado proveniente de uma máquina de estados
    para um dado tamanho de sequência 'tam' """

    print("\n-*-*-*-*-*-* ROTINA DE ANÁLISE DE OCUPAÇÕES DE ESTADOS EM EXECUÇÃO -*-*-*-*-*-*-*-*-*")

    vetor_de_ocupacao = sa.calc_occup_vector_V2(machine, tam)

    occup_ctxs = []

    print("\n##### Vetor de ocupação atual: #####")

    for k, v in vetor_de_ocupacao.items():
        print(f"'{k}': {v},")


def new_sequence_generator_v2(Folhas, alf, probs, probs_conds, seq_lenght):

    sts_prob = {}

    dados_para_estados = States_data_generator(Folhas, alf, probs_conds)

    for k in dados_para_estados.keys():
        probs_data = probs[len(k)-1]
        if k in probs_data:
            sts_prob[k]=probs_data[k]
        else:
            print("\nEstado não possui probabilidade calculada. Adotando valor nulo")
            sts_prob[k] = 0.0

    # print('\n')
    # for item in d_e.items():
    #     print(item)

    estados = states_generator(dados_para_estados, sts_prob)  # lista de objetos da classe States, cada qual com nome e outedges

    # dic_estados={}
    # for i in estados:
    #     dic_estados[i.name] = i.outedges
    # print(dic_estados)

    maquina = ProbabilisticGraph(estados, alf)

    numero_de_estados = len(maquina.states)

    print(f"\nNumero de estados utilizados para geração da máquina: {numero_de_estados}")

    print(f"\n------------------PROBABILIDADE DAS SEQUÊNCIAS ASSOCIADAS AOS RÓTULOS DOS ESTADOS DA MÁQUINA-----------------------------")

    for i in maquina.states:
        print(f"{i.name}: {round(sts_prob[i.name], 6)}")

    # print("\nGerando nova sequencia a partir da máquina criada...")

    # sequencia_gerada = generate_sequence(maquina, seq_lenght)

    # print(f"\nNova sequencia gerada:\n{sequencia_gerada[0:250]}...")

    if seq_lenght:

        print("\nGerando nova sequencia a partir da máquina criada...")

        sequencia_gerada = generate_sequence(maquina, seq_lenght)

        print(f"\nNova sequencia gerada:\n{sequencia_gerada[0:250]}...")

        return estados, maquina, numero_de_estados, sequencia_gerada

    else:

        return estados, maquina, numero_de_estados
