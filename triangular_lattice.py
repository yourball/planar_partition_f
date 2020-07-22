from __future__ import division
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
import itertools
import matplotlib.pyplot as plt

# =====================================================================================================================================


def add(a, B):
    if len(B) == 0:
        B = a
    else:
        B = np.vstack((B, a))
    return B


def angle(n):
    n = n - 1e-12
    n = n / np.sqrt(np.sum(n ** 2))
    phi = 2 * np.pi * (n[1] < 0) + np.arccos(n[0]) * np.sign(n[1])
    return phi % (2 * np.pi)


def trnglr_lattice(L, T):
    # -------------------------------------------------------------------------
    basisx = np.tile(np.hstack((np.arange(L), np.arange(L - 1))), int(T / 2))
    basisy = np.repeat(np.arange(T), np.tile([L, L - 1], int(T / 2)))
    # --------------------------------------------------------------------------
    X, Y = np.meshgrid(basisx, basisy)
    adjmatrix = AND(X - X.T == 1, Y - Y.T == 0)
    adjmatrix += AND(X - X.T == 0, np.abs(Y - Y.T) == 1)
    adjmatrix += (X - X.T) * (Y - Y.T) == 1 - 2 * (Y % 2)
    adjmatrix = np.triu(adjmatrix)
    adjmatrix += adjmatrix.T
    return basisx, basisy, adjmatrix


def list_edges(adjmatrix, basisx, basisy):
    S = len(adjmatrix)
    edges = []
    for i in range(S):
        for j in range(i + 1, S):
            if adjmatrix[i][j] != 0:
                edges = add([i, j], edges)
    E = len(edges)
    database = np.zeros([E, 7], int)
    for i in range(E):
        database[i][0] = i
        database[i][1:3] = edges[i]
        # ---------------------------------------
        vertex = edges[i][0]
        nn = np.arange(S)[adjmatrix[vertex]]
        nnidx, nnang = np.zeros(len(nn), int), np.zeros(len(nn))
        for s in range(len(nn)):
            nnidx[s] = np.arange(E)[
                OR(
                    AND(edges.T[0] == vertex, edges.T[1] == nn[s]),
                    AND(edges.T[1] == vertex, edges.T[0] == nn[s]),
                )
            ][0]
            n = np.array(
                [basisx[nn[s]] - basisx[vertex], basisy[nn[s]] - basisy[vertex]]
            )
            nnang[s] = angle(n)
        nnidx = nnidx[np.argsort(nnang)]
        indx0 = np.arange(len(nnidx))[nnidx == i][0]
        database[i][3], database[i][4] = (
            nnidx[(indx0 + 1) % len(nnidx)],
            nnidx[(indx0 - 1) % len(nnidx)],
        )
        # ---------------------------------------
        vertex = edges[i][1]
        nn = np.arange(S)[adjmatrix[vertex]]
        nnidx, nnang = np.zeros(len(nn), int), np.zeros(len(nn))
        for s in range(len(nn)):
            nnidx[s] = np.arange(E)[
                OR(
                    AND(edges.T[0] == vertex, edges.T[1] == nn[s]),
                    AND(edges.T[1] == vertex, edges.T[0] == nn[s]),
                )
            ][0]
            n = np.array(
                [basisx[nn[s]] - basisx[vertex], basisy[nn[s]] - basisy[vertex]]
            )
            nnang[s] = angle(n)
        nnidx = nnidx[np.argsort(nnang)]
        indx0 = np.arange(len(nnidx))[nnidx == i][0]
        database[i][5], database[i][6] = (
            nnidx[(indx0 + 1) % len(nnidx)],
            nnidx[(indx0 - 1) % len(nnidx)],
        )
    return database


def visualize(database, basisx, basisy, adjmatrix):
    for i in range(len(adjmatrix)):
        for j in range(i + 1, len(adjmatrix)):
            if adjmatrix[i][j] != 0:
                plt.plot(
                    [
                        basisx[i] + 0.5 * (basisy[i] % 2),
                        basisx[j] + 0.5 * (basisy[j] % 2),
                    ],
                    [basisy[i], basisy[j]],
                    c="k",
                )
    plt.axis("off")
    for i in range(len(adjmatrix)):
        plt.text(basisx[i] + 0.5 * (basisy[i] % 2) - 0.05, basisy[i] + 0.1, str(i))
    for k in range(len(database)):
        i, j = database[k][1:3]
        plt.text(
            0.5 * (basisx[i] + basisx[j])
            + 0.25 * (basisy[j] % 2)
            + 0.25 * (basisy[i] % 2)
            - 0.025,
            0.5 * (basisy[i] + basisy[j]) - 0.025,
            str(k),
            bbox={"facecolor": "white", "alpha": 1, "pad": 1},
        )


# =====================================================================================================================================
# generate random couplings from adjacency matrix
def random_couplings(adjmatrix):
    jmatrix = np.zeros([len(adjmatrix), len(adjmatrix)])
    for i in range(len(adjmatrix)):
        for j in range(i + 1, len(adjmatrix)):
            if adjmatrix[i][j] != 0:
                jmatrix[i, j] = 0.5 * np.random.normal(0, 1)
                jmatrix[j, i] = jmatrix[i, j]
    return jmatrix


# basis of classical spins
def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([-1, 1], repeat=num_sites)))
    return full_basis


def partition_function(jmatrix):
    basis = spin_basis(len(jmatrix))
    Z = 0
    for si in range(len(basis)):
        vec = basis[si]
        Z += np.exp(-np.dot(vec, np.dot(jmatrix, vec)))
    return Z


# ======================================================================================================================================

if __name__ == "main":
    # length and time for the lattice
    L, T = 4, 2
    # x-positions of vertices, y-positions of vertices, adjacency matrix for triagular lattice
    basisx, basisy, adjmatrix = trnglr_lattice(L, T)

    # database of the edges is in the format:
    # [index,vertex1,vertex2,c1,cc1,c2,cc2]
    database = list_edges(adjmatrix, basisx, basisy)

    # visualize the lattice with labels for edges and vertices
    visualize(database, basisx, basisy)

    # jmatrix -- matrix of couplings, generate randomly for all edges
    jmatrix = random_couplings(adjmatrix)
    # computing partition
    Z = partition_function(jmatrix)
    print("Z:", Z)
