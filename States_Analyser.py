def states_analyser(Dic_states):

    # print("\n >>> RUNNING STRANDEDS STATE REMOVAL ROUTINE <<<")

    sts = []
    destination_sts = []
    ajusted_sts = {}

    # the code snippet below stores the target states of each state present in 'Dic_states'
    for key_Dic, val_Dic in Dic_states.items():
        sts.append(key_Dic)  # armazena o rotulo do estado
        for i in range(len(Dic_states[key_Dic])):
            if Dic_states[key_Dic][i][1] not in destination_sts:
                destination_sts.append(Dic_states[key_Dic][i][1])    # armazenas estados de destino em 'destination_sts'

    # print(f"\ninitial States:                         {sts}")

    # print(f"States that are outedges from other states:  {destination_sts}")

    sts_not_strandeds = []  # variable used to store useful data (without strands)
    strandeds = []

    for caso in sts:  # for each state...
        if caso in destination_sts:  # ...if it is the destination state of some other...
            sts_not_strandeds.append(caso)  # ...is stored as useful state.
        else:
            # print(f"state {case} one is not outedge of another")
            strandeds.append(caso)
    # print(f"STRANDEDS STATES:                        {strandeds}")

    # print(f"States after stranded removal:        {sts_not_strandeds}")

    for s in sts_not_strandeds:

        ajusted_sts[s] = Dic_states[s]

    if strandeds:
        # print("\nOccurrence of stranded states detected.\nRerunning stranded removal...")
        usefull_sts = states_analyser(ajusted_sts)
    else:
        usefull_sts = ajusted_sts

        # print("\n<<< STRANDEDS STATE REMOVAL ROUTINE COMPLETED >>>")
        print("\nData ('state label', [state morphs]) resulting from the process:")

        for rslt_sts in ajusted_sts.items():
            print(rslt_sts)

    return usefull_sts
