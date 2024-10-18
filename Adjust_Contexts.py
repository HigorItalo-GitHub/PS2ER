def leafs_complementer(Folhas_Atual, Folhas, Alfabeto, probs_conds):
    # cria um dicionario com elementos na forma {contexto: [(letra do alf.,contexto subsequente,prob. condicional do contexto associada a letra)]}

    print("\n-*-*- RUNNING STATE CREATION ROUTINE -*-*-")

    new_ctx = []

    for f in Folhas_Atual:  # para cada folha no conjunto de folhas atualmente analisado...

        # print(F"\nANALYZING CONTEXT {f}:")

        for letra in Alfabeto:  # ...tome cada uma das letras no alfabeto da sequencia...

            flag = True    # flag para detecção de ausencia de sufixo para uma concatenação

            next = f + letra  # ... e faça a contanação da folha com a letra à esquerda, criando o contexto 'next'.

            cont = 0  # variavel para realizar contabilização de não identificação de folhas como sufixo para 'next'

            for context in Folhas:  # para cada um dos contextos(folhas) no conjunto original de folhas analisado

                if next.endswith(context):  # se alguma folha é identificada como sufixo da concatenação 'next'

                    pass  # apenas indique a identificação

                    # print(f"\nWORD {next} --> LEAF {context} IDENTIFIED AS A SUFFIX FOR {next}")

                    flag = False

                    # outs.append((letra, context, pbs[letra+"|"+f]))

                # else:  # caso não se identifique qualquer folha como sufixo para 'next'...
                #
                #     # print(f" concatenation {next} not associated with leaf ",element)
                #
                #     cont += 1  # itere o contador

            # if cont == len(Folhas):  # se o valor em 'cont' na rodada é igual ao total de folhas originais...
            if flag:

                # print(
                #     f"\nWORD {next} --> No leaf was identified among the current ones as a suffix for this context. "
                #     f"Stored for addition to the current leafs set.")

                new_ctx.append(f)  # armazene 'next' em "new_ctx' para ser adiconado às folhas originais

        # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    if new_ctx:  # se houver algum contexto armazenado em 'new_ctx', realiza nova analise das folhas...

        for ctx in new_ctx:

            for simb in Alfabeto:
                add_ctx = simb + ctx
                # print(f"ADDING CONTEXT {add_ctx} TO THE CURRENT SET OF SHEETS")
                Folhas.append(add_ctx)

            # if ctx in Folhas:
            Folhas.remove(ctx)

        print("=================> REASSESSING LEAF SET AFTER INSECTIONS <=========================")
        leafs_complementer(Folhas, Folhas, Alfabeto, probs_conds)  # ...e reexecutando esta função.

    return Folhas  # RETORNA AS FOLHAS DEVIDAMENTE COMPLEMENTADAS
