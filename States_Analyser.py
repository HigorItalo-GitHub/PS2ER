def states_analyser(Dic_states):

    print("\n >>> EXECUTANDO ROTINA DE REMOÇÃO DE ESTADOS STRANDEDS <<<")

    sts = []
    destination_sts = []
    ajusted_sts = {}

    # input(f"Dicionario de estados a analisar:\n{Dic_states}")

    # o trecho de codigo abaixo realiza o armazenamento de estados de destino de cada estado presente em 'Dic_states'
    for key_Dic, val_Dic in Dic_states.items():
        sts.append(key_Dic)  # armazena o rotulo do estado
        for i in range(len(Dic_states[key_Dic])):
            if Dic_states[key_Dic][i][1] not in destination_sts:
                destination_sts.append(Dic_states[key_Dic][i][1])    # armazenas estados de destino em 'destination_sts'

    print(f"\nEstados iniciais:                         {sts}")

    print(f"Estados que são saida de outros estados:  {destination_sts}")

    sts_not_strandeds = []  # variavel utilizada para armazenar os estadados uteis (sem strandeds)
    strandeds = []

    for caso in sts:  # para cada estado...
        if caso in destination_sts:  # ...se ele é estado de destino de algum outro...
            sts_not_strandeds.append(caso)  # ...é armazenado como estado util.
        else:
            # print(f"estado {caso} não se é outedge de outro")
            strandeds.append(caso)
    print(f"ESTADOS STRANDEDS:                        {strandeds}")

    print(f"Estados após remoção de strandeds:        {sts_not_strandeds}")

    for s in sts_not_strandeds:

        ajusted_sts[s] = Dic_states[s]

    if strandeds:
        print("\nOcorrencia de estados strandeds detectada.\nReexecutando remoção de strandeds...")
        usefull_sts = states_analyser(ajusted_sts)
    else:
        usefull_sts = ajusted_sts

        print("\n<<< ROTINA DE REMOÇÃO DE ESTADOS STRANDEDS CONCLUIDA >>>")
        print("\nEstados terminais resultantes da analise:")

        for rslt_sts in ajusted_sts.items():
            print(rslt_sts)

    return usefull_sts


if __name__ == "__main__":

    dic_3_0p528 = {
        '0': [('0', '0', 0.11674087944868432), ('1', '01', 0.18644355655231729), ('2', '02', 0.6968155639989984)],
        '001': [('0', '0', 0.0), ('1', '011', 0.0), ('2', '012', 0.0)],
        '101': [('0', '0', 0.0), ('1', '011', 0.0), ('2', '012', 0.0)],
        '201': [('0', '0', 0.10970838637345698), ('1', '011', 0.5037780139441997), ('2', '012', 0.3865135996823432)],
        '011': [('0', '0', 0.43415165635069664), ('1', '111', 0.42594524422189767), ('2', '112', 0.13990309942740572)],
        '111': [('0', '0', 0.20008361980860356), ('1', '111', 0.5732137879773297), ('2', '112', 0.22670259221406672)],
        '211': [('0', '0', 0.0), ('1', '111', 0.0), ('2', '112', 0.0)],
        '21': [('0', '0', 0.0), ('1', '211', 0.0), ('2', '212', 0.0)],
        '02': [('0', '0', 1.0), ('1', '21', 0.0), ('2', '22', 0.0)],
        '012': [('0', '0', 1.0), ('1', '21', 0.0), ('2', '22', 0.0)],
        '112': [('0', '0', 1.0), ('1', '21', 0.0), ('2', '22', 0.0)],
        '212': [('0', '0', 0.0), ('1', '21', 0.0), ('2', '22', 0.0)],
        '22': [('0', '0', 0.0), ('1', '21', 0.0), ('2', '22', 0.0)],
        '01': [('0', '0', 0.10970838637345698), ('1', '011', 0.5037780139441997), ('2', '012', 0.3865135996823432)]
    }

    dic_3_0p528_reduz = {
        '0': [('0', '0', 0.11645359464823733), ('1', '01', 0.18626724484941634), ('2', '02', 0.6972791605023464)],
        '001': [],
        '101': [],
        '201': [('0', '0', 0.10968483560021097), ('1', '011', 0.506102502394342), ('2', '012', 0.3842126620054471)],
        '011': [('0', '0', 0.431143120550661), ('1', '111', 0.42728673787919746), ('2', '112', 0.14157014157014158)],
        '111': [('0', '0', 0.19986927559208423), ('1', '111', 0.5713792352158575), ('2', '112', 0.2287514891920583)],
        '211': [],
        '21': [],
        '02': [('0', '0', 1.0)],
        '012': [('0', '0', 1.0)],
        '112': [('0', '0', 1.0)],
        '212': [],
        '22': [],
        '01': [('0', '0', 0.10968483560021097), ('1', '011', 0.506102502394342), ('2', '012', 0.3842126620054471)]
    }

    dic_sts_3 = {
        '000': [],
        '100': [],
        '200': [('2', '02', 1.0)],
        '010': [('2', '02', 1.0)],
        '110': [('2', '02', 1.0)],
        '210': [],
        '020': [('0', '200', 0.16701143708975863), ('1', '201', 0.2671343923647773), ('2', '02', 0.5658541705454642)],
        '120': [('2', '02', 1.0)],
        '220': [],
        '001': [],
        '101': [],
        '201': [('0', '010', 0.10968483560021097), ('1', '011', 0.506102502394342), ('2', '012', 0.3842126620054471)],
        '011': [('0', '110', 0.431143120550661), ('1', '111', 0.42728673787919746), ('2', '112', 0.14157014157014158)],
        '111': [('0', '110', 0.19986927559208423), ('1', '111', 0.5713792352158575), ('2', '112', 0.2287514891920583)],
        '211': [],
        '21':  [],
        '02':  [('0', '020', 1.0)],
        '012': [('0', '120', 1.0)],
        '112': [('0', '120', 1.0)],
        '212': [],
        '22':  [],
    }

    dic_sts_4 = {
        '000': [],
        '100': [],
        '0200': [('2', '02', 1.0)],
        '1200': [],
        '2200': [],
        '0010': [],
        '1010': [],
        '2010': [('2', '02', 1.0)],
        '0110': [('2', '02', 1.0)],
        '1110': [('2', '02', 1.0)],
        '2110': [],
        '210': [],
        '0020': [('2', '02', 1.0)],
        '1020': [('2', '02', 1.0)],
        '2020': [('0', '0200', 0.23245692222739153), ('1', '0201', 0.37181498406459385), ('2', '02', 0.3957280937080147)],
        '0120': [('2', '02', 1.0)],
        '1120': [('2', '02', 1.0)],
        '2120': [],
        '220': [],
        '001': [],
        '101': [],
        '0201': [('0', '2010', 0.10968483560021097), ('1', '2011', 0.506102502394342), ('2', '2012', 0.3842126620054471)],
        '1201': [],
        '2201': [],
        '0011': [],
        '1011': [],
        '2011': [('0', '0110', 0.431143120550661), ('1', '0111', 0.42728673787919746), ('2', '0112', 0.14157014157014158)],
        '0111': [('0', '1110', 0.18592943085034475), ('1', '1111', 0.5710747600378532), ('2', '1112', 0.24299580911180213)],
        '1111': [('0', '1110', 0.21032626553138323), ('1', '1111', 0.5716076376071427), ('2', '1112', 0.21806609686147405)],
        '2111': [],
        '211': [],
        '21': [],
        '02': [('0', '020', 1.0)],
        '0012': [],
        '1012': [],
        '2012': [('0', '0120', 1.0)],
        '0112': [('0', '1120', 1.0)],
        '1112': [('0', '1120', 1.0)],
        '2112': [],
        '212': [],
        '22': [],
        '020': [('0', '0200', 0.16701143708975863), ('1', '0201', 0.2671343923647773), ('2', '02', 0.5658541705454642)],
    }

    dic_sts_5 = {
        '000': [],
        '100': [],
        '00200': [],
        '10200': [],
        '20200': [('2', '02', 1.0)],
        '1200': [],
        '2200': [],
        '0010': [],
        '1010': [],
        '02010': [('2', '02', 1.0)],
        '12010': [],
        '22010': [],
        '00110': [],
        '10110': [],
        '20110': [('2', '02', 1.0)],
        '01110': [('2', '02', 1.0)],
        '11110': [('2', '02', 1.0)],
        '21110': [],
        '2110': [],
        '210': [],
        '00020': [],
        '10020': [],
        '20020': [('2', '02', 1.0)],
        '01020': [('2', '02', 1.0)],
        '11020': [('2', '02', 1.0)],
        '21020': [],
        '02020': [('0', '20200', 0.2906963263611631), ('1', '20201', 0.33131812440377345),
                  ('2', '02', 0.37798554923506344)],
        '12020': [('0', '20200', 0.016509081530094023), ('1', '20201', 0.5219735568115058),
                  ('2', '02', 0.46151736165840024)],
        '22020': [],
        '00120': [],
        '10120': [],
        '20120': [('2', '02', 1.0)],
        '01120': [('2', '02', 1.0)],
        '11120': [('2', '02', 1.0)],
        '21120': [],
        '2120': [],
        '220': [],
        '001': [],
        '101': [],
        '00201': [],
        '10201': [],
        '20201': [('0', '02010', 0.10968483560021097), ('1', '02011', 0.506102502394342),
                  ('2', '02012', 0.3842126620054471)],
        '1201': [],
        '2201': [],
        '0011': [],
        '1011': [],
        '02011': [('0', '20110', 0.431143120550661), ('1', '20111', 0.42728673787919746),
                  ('2', '20112', 0.14157014157014158)],
        '12011': [],
        '22011': [],
        '00111': [],
        '10111': [],
        '20111': [('0', '01110', 0.18592943085034475), ('1', '01111', 0.5710747600378533),
                  ('2', '01112', 0.24299580911180213)],
        '01111': [('0', '11110', 0.21640815862735077), ('1', '11111', 0.5718803855841524),
                  ('2', '11112', 0.21171145578849687)],
        '11111': [('0', '11110', 0.2057681799149818), ('1', '11111', 0.5714032261498393),
                  ('2', '11112', 0.22282859393517895)],
        '21111': [],
        '2111': [],
        '211': [],
        '21': [],
        '02': [('0', '020', 1.0)],
        '0012': [],
        '1012': [],
        '02012': [('0', '20120', 1.0)],
        '12012': [],
        '22012': [],
        '00112': [],
        '10112': [],
        '20112': [('0', '01120', 1.0)],
        '01112': [('0', '11120', 1.0)],
        '11112': [('0', '11120', 1.0)],
        '21112': [],
        '2112': [],
        '212': [],
        '22': [],
        '020': [('0', '0200', 0.16701143708975863), ('1', '0201', 0.2671343923647773), ('2', '02', 0.5658541705454642)],
        '0200': [('2', '02', 1.0)],
        '0201': [('0', '02010', 0.10968483560021097), ('1', '02011', 0.506102502394342),
                 ('2', '02012', 0.3842126620054471)],
    }

    dic_sts_8 = {
        '000': [],
        '100': [],
        '00200': [],
        '10200': [],
        '020200': [('2', '02', 1.0)],
        '00120200': [],
        '10120200': [],
        '20120200': [('2', '02', 1.0)],
        '01120200': [('2', '02', 1.0)],
        '11120200': [('2', '02', 1.0)],
        '21120200': [],
        '2120200': [],
        '220200': [],
        '1200': [],
        '2200': [],
        '0010': [],
        '1010': [],
        '002010': [],
        '102010': [],
        '00202010': [],
        '10202010': [('2', '02', 1.0)],
        '20202010': [('2', '02', 1.0)],
        '01202010': [('2', '02', 1.0)],
        '11202010': [('2', '02', 1.0)],
        '21202010': [],
        '2202010': [],
        '12010': [],
        '22010': [],
        '00110': [],
        '10110': [],
        '0020110': [],
        '1020110': [],
        '02020110': [('2', '02', 1.0)],
        '12020110': [('2', '02', 1.0)],
        '22020110': [],
        '120110': [],
        '220110': [],
        '001110': [],
        '101110': [],
        '00201110': [],
        '10201110': [],
        '20201110': [('2', '02', 1.0)],
        '1201110': [],
        '2201110': [],
        '0011110': [],
        '1011110': [],
        '02011110': [('2', '02', 1.0)],
        '12011110': [],
        '22011110': [],
        '00111110': [],
        '10111110': [],
        '20111110': [('2', '02', 1.0)],
        '01111110': [('2', '02', 1.0)],
        '11111110': [('2', '02', 1.0)],
        '21111110': [],
        '2111110': [],
        '211110': [],
        '21110': [],
        '2110': [],
        '210': [],
        '00020': [],
        '10020': [],
        '0020020': [],
        '1020020': [],
        '02020020': [('2', '02', 1.0)],
        '12020020': [('2', '02', 1.0)],
        '22020020': [],
        '120020': [],
        '220020': [],
        '001020': [],
        '101020': [],
        '00201020': [],
        '10201020': [],
        '20201020': [('2', '02', 1.0)],
        '1201020': [],
        '2201020': [],
        '0011020': [],
        '1011020': [],
        '02011020': [('2', '02', 1.0)],
        '12011020': [],
        '22011020': [],
        '00111020': [],
        '10111020': [],
        '20111020': [('2', '02', 1.0)],
        '01111020': [('2', '02', 1.0)],
        '11111020': [('2', '02', 1.0)],
        '21111020': [],
        '2111020': [],
        '211020': [],
        '21020': [],
        '0002020': [],
        '1002020': [],
        '02002020': [('0', '020200', 0.5417325220572693), ('1', '020201', 0.4582674779427306)],
        '12002020': [],
        '22002020': [],
        '00102020': [],
        '10102020': [],
        '20102020': [('0', '020200', 0.4507228453238944), ('1', '020201', 0.35810695552049127), ('2', '02', 0.1911701991556143)],
        '01102020': [('0', '020200', 0.18755158258036164), ('1', '020201', 0.586974929526137), ('2', '02', 0.22547348789350138)],
        '11102020': [('0', '020200', 0.4403701642081826), ('1', '020201', 0.3612927915391038), ('2', '02', 0.19833704425271362)],
        '21102020': [],
        '2102020': [],
        '00202020': [],
        '10202020': [('2', '02', 1.0)],
        '20202020': [('0', '020200', 0.20029329760654668), ('1', '020201', 0.29769707064489265), ('2', '02', 0.5020096317485607)],
        '01202020': [('2', '02', 1.0)],
        '11202020': [('2', '02', 1.0)],
        '21202020': [],
        '2202020': [],
        '0012020': [],
        '1012020': [],
        '02012020': [('0', '20120200', 0.016329027623905815), ('1', '20120201', 0.5084490923921645), ('2', '02', 0.4752218799839296)],
        '12012020': [],
        '22012020': [],
        '00112020': [],
        '10112020': [],
        '20112020': [('0', '01120200', 0.01578260159947772), ('1', '01120201', 0.5447364125999674), ('2', '02', 0.43948098580055495)],
        '01112020': [('0', '11120200', 0.018403952287698058), ('1', '11120201', 0.5583941605839415), ('2', '02', 0.4232018871283603)],
        '11112020': [('0', '11120200', 0.016853619063563818), ('1', '11120201', 0.5482448797365924), ('2', '02', 0.4349015011998438)],
        '21112020': [],
        '2112020': [],
        '212020': [],
        '22020': [],
        '00120': [],
        '10120': [],
        '0020120': [],
        '1020120': [],
        '02020120': [('2', '02', 1.0)],
        '12020120': [('2', '02', 1.0)],
        '22020120': [],
        '120120': [],
        '220120': [],
        '001120': [],
        '101120': [],
        '00201120': [],
        '10201120': [],
        '20201120': [('2', '02', 1.0)],
        '1201120': [],
        '2201120': [],
        '0011120': [],
        '1011120': [],
        '02011120': [('2', '02', 1.0)],
        '12011120': [],
        '22011120': [],
        '00111120': [],
        '10111120': [],
        '20111120': [('2', '02', 1.0)],
        '01111120': [('2', '02', 1.0)],
        '11111120': [('2', '02', 1.0)],
        '21111120': [],
        '2111120': [],
        '211120': [],
        '21120': [],
        '2120': [],
        '220': [],
        '001': [],
        '101': [],
        '00201': [],
        '10201': [],
        '020201': [('0', '0202010', 0.08965136874799008), ('1', '0202011', 0.4923876983472382), ('2', '0202012', 0.4179609329047717)],
        '00120201': [],
        '10120201': [],
        '20120201': [('0', '01202010', 0.15423755192931626), ('1', '01202011', 0.5387718941180696), ('2', '01202012', 0.3069905539526141)],
        '01120201': [('0', '11202010', 0.1645793384467881), ('1', '11202011', 0.5389201821668265), ('2', '11202012', 0.2965004793863854)],
        '11120201': [('0', '11202010', 0.1600505828018474), ('1', '11202011', 0.5368558023605307), ('2', '11202012', 0.30309361483762187)],
        '21120201': [],
        '2120201': [],
        '220201': [],
        '1201': [],
        '2201': [],
        '0011': [],
        '1011': [],
        '002011': [],
        '102011': [],
        '00202011': [('0', '02020110', 1.0)],
        '10202011': [('0', '02020110', 0.2856010214378385), ('1', '02020111', 0.5422475751148087), ('2', '02020112', 0.1721514034473527)],
        '20202011': [('0', '02020110', 0.24011562752018764), ('1', '02020111', 0.5670775772682998), ('2', '02020112', 0.19280679521151245)],
        '01202011': [('0', '12020110', 0.24148926714368252), ('1', '12020111', 0.572096795697969), ('2', '12020112', 0.1864139371583485)],
        '11202011': [('0', '12020110', 0.23874788494077837), ('1', '12020111', 0.5631979695431473), ('2', '12020112', 0.19805414551607448)],
        '21202011': [],
        '2202011': [],
        '12011': [],
        '22011': [],
        '00111': [],
        '10111': [],
        '0020111': [],
        '1020111': [],
        '02020111': [('0', '20201110', 0.18584161938467875), ('1', '20201111', 0.5673557665299702), ('2', '20201112', 0.24680261408535104)],
        '12020111': [('0', '20201110', 0.18604948860072454), ('1', '20201111', 0.5761594490456867), ('2', '20201112', 0.23779106235358877)],
        '22020111': [],
        '120111': [],
        '220111': [],
        '001111': [],
        '101111': [],
        '00201111': [],
        '10201111': [],
        '20201111': [('0', '02011110', 0.21640815862735074), ('1', '02011111', 0.5718803855841524), ('2', '02011112', 0.21171145578849687)],
        '1201111': [],
        '2201111': [],
        '0011111': [],
        '1011111': [],
        '02011111': [('0', '20111110', 0.20194059012484686), ('1', '20111111', 0.5702553233764944), ('2', '20111112', 0.22780408649865883)],
        '12011111': [],
        '22011111': [],
        '00111111': [],
        '10111111': [],
        '20111111': [('0', '01111110', 0.209465737514518), ('1', '01111111', 0.5729094076655052), ('2', '01111112', 0.2176248548199768)],
        '01111111': [('0', '11111110', 0.20693325224266382), ('1', '11111111', 0.5668744615072728), ('2', '11111112', 0.22619228625006338)],
        '11111111': [('0', '11111110', 0.20883625597813707), ('1', '11111111', 0.5754573749335762), ('2', '11111112', 0.21570636908828666)],
        '21111111': [],
        '2111111': [],
        '211111': [],
        '21111': [],
        '2111': [],
        '211': [],
        '21': [],
        '02': [('0', '020', 1.0)],
        '0012': [],
        '1012': [],
        '002012': [],
        '102012': [],
        '0202012': [('0', '02020120', 1.0)],
        '01202012': [('0', '12020120', 1.0)],
        '11202012': [('0', '12020120', 1.0)],
        '21202012': [],
        '2202012': [],
        '12012': [],
        '22012': [],
        '00112': [],
        '10112': [],
        '0020112': [],
        '1020112': [],
        '02020112': [('0', '20201120', 1.0)],
        '12020112': [('0', '20201120', 1.0)],
        '22020112': [],
        '120112': [],
        '220112': [],
        '001112': [],
        '101112': [],
        '00201112': [],
        '10201112': [],
        '20201112': [('0', '02011120', 1.0)],
        '1201112': [],
        '2201112': [],
        '0011112': [],
        '1011112': [],
        '02011112': [('0', '20111120', 1.0)],
        '12011112': [],
        '22011112': [],
        '00111112': [],
        '10111112': [],
        '20111112': [('0', '01111120', 1.0)],
        '01111112': [('0', '11111120', 1.0)],
        '11111112': [('0', '11111120', 1.0)],
        '21111112': [],
        '2111112': [],
        '211112': [],
        '21112': [],
        '2112': [],
        '212': [],
        '22': [],
        '0202010': [('2', '02', 1.0)],
        '0202011': [('0', '02020110', 0.5196900221654456), ('1', '02020111', 0.36143550870543645), ('2', '02020112', 0.11887446912911796)],
        '020': [('0', '0200', 0.16701143708975863), ('1', '0201', 0.2671343923647773), ('2', '02', 0.5658541705454642)],
        '0200': [('2', '02', 1.0)],
        '0201': [('0', '02010', 0.10968483560021097), ('1', '02011', 0.506102502394342), ('2', '02012', 0.3842126620054471)],
        '02010': [('2', '02', 1.0)],
        '02011': [('0', '020110', 0.431143120550661), ('1', '020111', 0.42728673787919746), ('2', '020112', 0.14157014157014158)],
        '02012': [('0', '020120', 1.0)],
        '020110': [('2', '02', 1.0)],
        '020111': [('0', '0201110', 0.18592943085034475), ('1', '0201111', 0.5710747600378533), ('2', '0201112', 0.24299580911180213)],
        '020112': [('0', '0201120', 1.0)],
        '020120': [('2', '02', 1.0)],
        '0201110': [('2', '02', 1.0)],
        '0201111': [('0', '02011110', 0.21640815862735074), ('1', '02011111', 0.5718803855841523), ('2', '02011112', 0.21171145578849687)],
        '0201112': [('0', '02011120', 1.0)],
        '0201120': [('2', '02', 1.0)]
    }

    print("----------TESTE DO MODULULO-----------")

    x = states_analyser(dic_sts_8)

    # print("\n")
    # for i in x.items():
    #     print(i)
