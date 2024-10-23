"""The 'leafs_complementer' function implements the action of
identifying suffixes and dividing sequences when necessary.
It is associated with Step 2 of the PS2ER algorithm."""

def leafs_complementer(Actual_Leafs, Leafs, Alphabet, probs_conds):
    # Creates a dictionary with elements in the form
    # {context: [
    #           (letter of the alf., subsequent context, conditional prob. of the context associated with the letter)
    #           ]
    # }

    # print("\n-*-*- RUNNING STATE CREATION ROUTINE -*-*-")

    new_ctx = []

    for f in Actual_Leafs:  # for each leaf in the currently analyzed leaf set...

        # print(F"\nANALYZING CONTEXT {f}:")

        for letra in Alphabet:  # ...take each of the symbols in the alphabet associated with sequence...

            flag = True    # flag for detecting missing suffix for a concatenation

            next = f + letra  # ... and contaminate the leaf with the symbol on the left, creating the 'next' context.

            cont = 0  # variable to account for non-identification of leafs as a suffix for 'next'

            for context in Leafs:  # For each of the contexts (leaves) in the original set of leaves analyzed...

                if next.endswith(context):  # ...if any leaf is identified as suffix of the 'next' concatenation...

                    # print(f"\nWORD {next} --> LEAF {context} IDENTIFIED AS A SUFFIX FOR {next}")

                    flag = False    # flag to register the identification of a suffix for the analyzed concatenation

            if flag:

                # print(
                #     f"\nWORD {next} --> No leaf was identified among the current ones as a suffix for this context. "
                #     f"Stored for addition to the current leafs set.")

                new_ctx.append(f)  # store 'next' in "new_ctx' to be added to the original leafs

        # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    if new_ctx:  # If there is any context stored in 'new_ctx', perform a new analysis of the leaves...

        for ctx in new_ctx:

            for simb in Alphabet:
                add_ctx = simb + ctx
                # print(f"ADDING CONTEXT {add_ctx} TO THE CURRENT SET OF SHEETS")
                Leafs.append(add_ctx)

            Leafs.remove(ctx)

        # print("=================> REASSESSING LEAF SET AFTER INSECTIONS <=========================")
        leafs_complementer(Leafs, Leafs, Alphabet, probs_conds)  # ...and re-executing this function.

    return Leafs  # RETURN THE LEAVES SET PROPERLY COMPLETED
