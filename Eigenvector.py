import numpy as np
from scipy.linalg import eig
from sklearn.preprocessing import normalize
import pickle


def near(a, b, rtol=1e-5, atol=1e-8):
    return np.abs(a - b) < (atol + rtol * np.abs(b))


def trans_prob_matrix(machine):
    n = len(machine.states)
    P = np.zeros((n, n))
    st_names = [st.name for st in machine.states]
    idxs = {k: v for v, k in enumerate(st_names)}

    for st in machine.states:
        cur_st = idxs[st.name]
        for oedg in st.outedges:
            next_st = idxs[oedg[1]]
            P[cur_st][next_st] += oedg[-1]
    return P


def occup_vector(machine):
    # P = trans_prob_matrix(machine)
    P = np.array([[0.4,0.6],[0.1,0.9]])
    values, vectors = eig(P, right=False, left=True)
    vectors = np.matrix.transpose(vectors)
    occup = vectors[near(values, 1, 0.02)][0]
    # try:
    normalized_occup = normalize(occup.real[:, np.newaxis], norm='l1', axis=0).ravel()
    # except:
    #     normalized_occup = normalize(occup.real[:,np.newaxis], norm='l1', axis=0).ravel()
    return abs(normalized_occup)


if __name__ == "__main__":

    machine_archive = 'DFC_bin_MACHINE_DMK_D5.pkl'

    with open(machine_archive, 'rb') as mach:
        machine = pickle.load(mach)

    ov = occup_vector(machine)

    D_occup = {s.name: occup for s,occup in zip(machine.states, ov)}

    print("Vetor de ocupações calculado:\n", ov)
    print("\nOcupações de cada estado:")
    for k, v in D_occup.items():
        print(f"{k}:{round(v,5)}")
