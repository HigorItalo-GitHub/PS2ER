# from Comparing_Morphs import calc_morphs_dists


def leafs_complementer(Folhas_Atual, Folhas, Alfabeto, probs_conds):
    # cria um dicionario com elementos na forma {contexto: [(letra do alf.,contexto subsequente,prob. condicional do contexto associada a letra)]}

    print("\n-*-*- ROTINA DE CRIAÇÃO DE ESTADOS EM EXECUÇÃO -*-*-")

    new_ctx = []

    for f in Folhas_Atual:  # para cada folha no conjunto de folhas atualmente analisado...

        print(F"\nANALISANDO CONTEXTO {f}:")

        for letra in Alfabeto:  # ...tome cada uma das letras no alfabeto da sequencia...

            flag = True    # flag para detecção de ausencia de sufixo para uma concatenação

            next = f + letra  # ... e faça a contanação da folha com a letra à esquerda, criando o contexto 'next'.

            cont = 0  # variavel para realizar contabilização de não identificação de folhas como sufixo para 'next'

            for context in Folhas:  # para cada um dos contextos(folhas) no conjunto original de folhas analisado

                if next.endswith(context):  # se alguma folha é identificada como sufixo da concatenação 'next'

                    pass  # apenas indique a identificação

                    print(f"\nPALAVRA {next} --> FOLHA {context} IDENTIFICADA COMO SUFIXO PARA {next}")

                    flag = False

                    # outs.append((letra, context, pbs[letra+"|"+f]))

                # else:  # caso não se identifique qualquer folha como sufixo para 'next'...
                #
                #     # print(f"Concatenação {next} não associada à folha ",element)
                #
                #     cont += 1  # itere o contador

            # if cont == len(Folhas):  # se o valor em 'cont' na rodada é igual ao total de folhas originais...
            if flag:

                print(
                    f"\nPALAVRA {next} --> Não foi identificada folha entre as atuais como sufixo para esse contexto. "
                    f"Armazenada para adição no conjunto de folhas atuias.")

                new_ctx.append(f)  # armazene 'next' em "new_ctx' para ser adiconado às folhas originais

        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    if new_ctx:  # se houver algum contexto armazenado em 'new_ctx', realiza nova analise das folhas...

        for ctx in new_ctx:

            for simb in Alfabeto:
                add_ctx = simb + ctx
                print(f"ADICONANDO CONTEXTO {add_ctx} AO CONJUNTO ATUAL DE FOLHAS")
                Folhas.append(add_ctx)

            # if ctx in Folhas:
            Folhas.remove(ctx)

        print("=================> REAVALIANDO CONJUNTO DE FOLHAS APÓS INSEÇÕES <=========================")
        leafs_complementer(Folhas, Folhas, Alfabeto, probs_conds)  # ...e reexecutando esta função.

    return Folhas  # RETORNA AS FOLHAS DEVIDAMENTE COMPLEMENTADAS


# def find_similar_conex(nodes, alf, probs_conds):

#     to_add = []
#     to_remove = []

#     for n in nodes:  # para cada folha no conjunto de folhas atualmente analisado...

#         # pbs = cop[f]

#         print(F"\nANALISANDO CONTEXTO {n}:")

#         for letter in alf:  # ...tome cada uma das letras no alfabeto da sequencia...

#             next = n + letter  # ... e faça a contanação da folha com a letra à esquerda, criando o contexto 'next'.

#             cont = 0  # variavel para realizar contabilização de não identificação de folhas como sufixo para 'next'

#             for context in nodes:  # para cada um dos contextos(folhas) no conjunto original de folhas analisado

#                 if next.endswith(context):  # se alguma folha é identificada como sufixo da concatenação 'next'

#                     # pass  # apenas indique a identificação

#                     print(f"\nPALAVRA {next} --> FOLHA {context} IDENTIFICADA COMO SUFIXO PARA {next}")

#                     suffix = context

#                 else:  # caso não se identifique qualquer folha como sufixo para 'next'...

#                     # print(f"Concatenação {next} não associada à folha ",element)

#                     cont += 1  # itere o contador

#             d = calc_morphs_dists([next, suffix], alf, probs_conds)

#             print(f"\n==========================================================="
#                   f"\nDistancia euclidiana entre morphs de {[next, suffix]}: {d}\n"
#                   f"===========================================================")

#             if d >= 0.0605:

#                 to_add.append(next)

#                 if next.endswith(suffix):

#                     if suffix not in to_remove:

#                         to_remove.append(suffix)

#     for obsolet_suffix in to_remove:

#         nodes.remove(obsolet_suffix)

#     for relevant_context in to_add:

#         nodes.append(relevant_context)

#     return nodes


# if __name__ == "__main__":

#     import Sequences_Analyser
#     L = 35  # Limita o comprimento de palavras para calculos probabilisticos e spliting

#     """Leitura de Sequência para sistema MaFaulDa"""
#     with open('MaFaulDa_Normal_C3_12288_BIN.txt', 'r') as f:
#     # with open('MaFaulDa_Normal_12288-C3_TERN.txt', 'r') as f:
#         for linha in f:
#             sequencia = linha
#     sequence_lenght = len(sequencia)

#     """Calculos probabilisticos utilizados para obtenção das máquina D-markov"""
#     probabilidades, seq_alf = Sequences_Analyser.calc_probs(sequencia, L)
#     probabilidades_condicionais = Sequences_Analyser.calc_cond_probs(probabilidades, seq_alf, L)

#     alfabeto_da_sequencia = sorted(seq_alf)  # organiza o alfabeto de simbolos da sequencia analisada

#     conj_folhas = ['01', '00', '0010', '1010', '011', '110', '111']

#     print(find_similar_conex(conj_folhas, alfabeto_da_sequencia, probabilidades_condicionais))
