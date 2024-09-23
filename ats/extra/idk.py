from math import log2, ceil
import numpy as np
import parasail
import string
np.set_printoptions(linewidth=500)

gap_extend = 1
gap_open = 1
match = 1
mismatch = 1

# def next_power(N):
#     return pow(2, ceil(log2(N)));

# Ht = [0, 5, 2, 0, 5, 15, 2, 5, 2, 5, 7, 0, 0, 0]

# Et = [i for i in Ht] + [0] * (next_power(len(Ht)) - len(Ht))
# # Upward sweep
# for i in range(int(log2(len(Et)))):
#     for j in range(0, len(Et), 2**(i+1)):
#         Et[j + 2**(i+1) - 1] = max(
#                     Et[j + 2**(i+1) - 1] + 2**i * gap_extend,
#                     Et[j + 2**i - 1],
#                 )

# # Downward  sweep
# Et[-1] = 0 #-np.inf
# for i in range(int(log2(len(Et)))-1, -1, -1):
#     for j in range(0, len(Et), 2**(i+1)):
#         T = Et[j + 2**i - 1]
#         Et[j + 2**i - 1] = Et[j + 2**(i+1) - 1]
#         Et[j + 2**(i+1) - 1] = max(T, Et[j + 2**(i+1) - 1]) - 2**i * gap_extend
# print(Et[:len(Ht)])


# nHt = np.array(Ht)
# ar = np.arange(0, gap_extend*nHt.shape[0], gap_extend)
# print((nHt + ar).tolist())
# print(nHt.tolist())
# Et = np.maximum.accumulate(nHt + ar) - ar - gap_extend
# print(Et.tolist())


SIMD_ELEM = 8

def striped(query, database):
    vQuery = query.reshape(SIMD_ELEM, -1).T
    pH = -np.arange(0, query.shape[0]*gap_extend, gap_extend).reshape(SIMD_ELEM, -1).T.astype(float) - gap_open
    E = pH - gap_open
    H = np.full_like(pH, -np.inf)
    for i in range(database.shape[0]):
        Y = database[i]

        vF = np.full(SIMD_ELEM, -np.inf)
        vF[0] = -2*gap_open - i*gap_extend

        vH = pH[-1]
        vH[1:] = vH[:-1]
        vH[0] =  -gap_open - (i-1)*gap_extend if i > 0 else 0

        for j in range(vQuery.shape[0]):
            sim = vQuery[j] == Y
            score = sim * match - ~sim * mismatch
            vH = vH + score

            vE = E[j]

            vH = np.maximum(vH, vE)
            vH = np.maximum(vH, vF)
            H[j] = vH

            E[j] = np.maximum(vE - gap_extend, vH - gap_open)
            vF = np.maximum(vF - gap_extend, vH - gap_open)

            vH = pH[j]

        ex = False
        for k in range(SIMD_ELEM):
            vF[1:] = vF[:-1]
            vF[0] = -2*gap_open - i*gap_extend
            for j in range(vQuery.shape[0]):
                vH = np.maximum(H[j], vF)
                H[j] = vH

                vH = vH - gap_open
                vF = vF - gap_extend
                if not np.any(vF > vH):
                    ex = True
                    break
                # vF = np.maximum(vH, vF)
            if ex:
                break

        pH, H = H, pH
    return pH

def scan3(query, database):
    vQuery = query.reshape(SIMD_ELEM, -1).T
    pH = -np.arange(0, query.shape[0]*gap_extend, gap_extend).reshape(SIMD_ELEM, -1).T.astype(float) - gap_open
    # H, F, E = np.full_like(pH, -np.inf), np.full_like(pH, -np.inf), pH - gap_open
    H, F, E = np.full_like(pH, -np.inf), np.full_like(pH, -np.inf), np.full_like(pH, -np.inf)

    vStride = np.arange(0, vQuery.shape[1]*gap_extend*vQuery.shape[0], gap_extend*vQuery.shape[0])
    for i in range(database.shape[0]):
        Y = database[i]

        vF = np.full(SIMD_ELEM, -np.inf)
        # vF[0] = -gap_open - i*gap_extend

        vH = pH[-1].copy()
        vH[1:] = vH[:-1]
        vH[0] =  -gap_open - (i-1)*gap_extend if i > 0 else 0

        for j in range(vQuery.shape[0]):
            sim = vQuery[j] == Y
            score = sim * match - ~sim * mismatch
            vH = vH + score

            E[j] = np.maximum(E[j] - gap_extend, pH[j] - gap_open)
            vH = np.maximum(vH, E[j])
            vH = np.maximum(vH, vF)
            H[j] = vH
            vF = np.maximum(vF - gap_extend, vH - gap_open)
            vH = pH[j]

        vF = np.maximum.accumulate(vF + vStride) - vStride
        vF[1:] = vF[:-1]
        vF[0] = -2*gap_open - i*gap_extend

        for j in range(vQuery.shape[0]):
            F[j] = vF
            H[j] = np.maximum(H[j], vF)
            # E[j] = np.maximum(E[j] - gap_extend, H[j] - gap_open)
            vF = np.maximum(vF - gap_extend, H[j] - gap_open)
        pH, H = H, pH
    print(F)
    return pH

def scan2(query, database):
    vQuery = query.reshape(SIMD_ELEM, -1).T
    pH = -np.arange(0, query.shape[0]*gap_extend, gap_extend).reshape(SIMD_ELEM, -1).T.astype(float) - gap_open
    # H, F, E = np.full_like(pH, -np.inf), np.full_like(pH, -np.inf), pH - gap_open
    H, F, E = np.full_like(pH, -np.inf), np.full_like(pH, -np.inf), np.full_like(pH, -np.inf)

    vStride = np.arange(0, vQuery.shape[1]*gap_extend*vQuery.shape[0], gap_extend*vQuery.shape[0])
    for i in range(database.shape[0]):
        Y = database[i]

        vF = np.full(SIMD_ELEM, -np.inf)
        vF[0] = -2*gap_open - i*gap_extend

        vH = pH[-1].copy()
        vH[1:] = vH[:-1]
        vH[0] =  -gap_open - (i-1)*gap_extend if i > 0 else 0

        for j in range(vQuery.shape[0]):
            sim = vQuery[j] == Y
            score = sim * match - ~sim * mismatch
            vH = vH + score

            E[j] = np.maximum(E[j] - gap_extend, pH[j] - gap_open)
            vH = np.maximum(vH, E[j])
            vH = np.maximum(vH, vF)
            H[j] = vH

            vF = np.maximum(vF - gap_extend, vH - gap_open)
            vH = pH[j]

        vF = np.maximum.accumulate(vF + vStride) - vStride
        vF[1:] = vF[:-1]
        vF[0] = -2*gap_open - i*gap_extend

        for j in range(vQuery.shape[0]):
            F[j] = vF
            H[j] = np.maximum(H[j], vF)
            # E[j] = np.maximum(E[j] - gap_extend, H[j] - gap_open)
            vF = np.maximum(vF - gap_extend, H[j] - gap_open)
        pH, H = H, pH
    print(F)
    return pH

def scan(query, database):
    vQuery = query.reshape(SIMD_ELEM, -1).T
    pH = -np.arange(0, query.shape[0]*gap_extend, gap_extend).reshape(SIMD_ELEM, -1).T.astype(float) - gap_open
    print(pH)
    k = np.arange(0, vQuery.shape[1]*gap_extend*vQuery.shape[0], gap_extend*vQuery.shape[0])
    print(k)
    # print(pH)
    E = pH - gap_open
    H = np.full_like(pH, -np.inf)
    vW = np.full(SIMD_ELEM, -np.inf)
    for i in range(database.shape[0]):
        Y = database[i]

        vF = np.full(SIMD_ELEM, -np.inf)
        vF[0] = -2*gap_open - i*gap_extend

        vH = pH[-1]
        vH[1:] = vH[:-1]
        vH[0] =  -gap_open - (i-1)*gap_extend if i > 0 else 0

        for j in range(vQuery.shape[0]):
            sim = vQuery[j] == Y
            score = sim * match - ~sim * mismatch
            vH = vH + score

            vE = E[j]

            vH = np.maximum(vH, vE)
            vH = np.maximum(vH, vF)
            H[j] = vH

            E[j] = np.maximum(vE - gap_extend, vH - gap_open)
            vF = np.maximum(vF - gap_extend, vH - gap_open)

            # print(vW)
            vH = pH[j] = np.maximum(pH[j], vW)
            vW = np.maximum(vH - gap_open, vW - gap_extend)
        # print()

        # print(vF)
        vW = np.maximum.accumulate(vF + k) - k
        vW[1:] = vW[:-1]
        vW[0] = -2*gap_open - i*gap_extend
        # print(vW)
        pH, H = H, pH

    for j in range(vQuery.shape[0]):
        # print(vW)
        pH[j] = np.maximum(pH[j], vW)
        vW = np.maximum(pH[j] - gap_open, vW - gap_extend)

    return pH

alphabet = np.array(list(string.printable))
def do_parasail_striped(x, y):
    x = ''.join(alphabet[x.astype(int)])
    y = ''.join(alphabet[y.astype(int)])
    matrix = parasail.matrix_create(''.join(list(np.union1d(x, y))), match=match, mismatch=-mismatch, case_sensitive=True)
    r = parasail.nw_rowcol_striped_16(x, y, open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
    return np.copy(r.score_col).astype(float).reshape(SIMD_ELEM, -1).T


def do_parasail_scan(x, y):
    x = (alphabet[x.astype(int)])
    y = (alphabet[y.astype(int)])
    alpha2 = ''.join(list(np.union1d(x, y)))
    matrix = parasail.matrix_create(alpha2, match=match, mismatch=-mismatch, case_sensitive=True)
    r = parasail.nw_rowcol_scan_16(''.join(x), ''.join(y), open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
    return np.copy(r.score_col).astype(float).reshape(SIMD_ELEM, -1).T


def padto(arr):
    w = SIMD_ELEM - 1
    a = np.pad(arr, (0, (arr.shape[0] + w & ~w) - arr.shape[0]))
    return a

ql, dl = 200, 15
ql = (ql + SIMD_ELEM - 1) & -SIMD_ELEM
print(ql)
seed = 50
np.random.seed(seed)
query = np.random.randint(0, 10, size=ql).astype(float)
database = np.random.randint(0, 10, size=dl).astype(float)

s1 = scan(query, database)
sp = do_parasail_scan(query, database)
r = s1 == sp
print(r)
# query = "heLLOZZZZZ"*5
# database = "Hello"*6
# alphabet = np.union1d(list(query), list(database))
# query = np.searchsorted(alphabet, list(query)) + 1
# database = np.searchsorted(alphabet, list(database)) + 1
# scan(padto(query), padto(database))
# r = striped(query, database) == do_parasail_striped(query, database)
# print(s1)
# print(sp)
# print(r.shape)
# print(r)
# print(np.all(r))

# import string
# np.set_printoptions(linewidth=500)


def nw_full(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1):
    lx, ly = len(x), len(y)

    h = np.zeros((lx+1, ly+1))
    e = np.full((lx+1, ly+1), fill_value=-np.inf)
    f = np.full((lx+1, ly+1), fill_value=-np.inf)

    for i in range(1, ly+1):
        e[0, i] = max(e[0, i-1]+gap_extend, h[0, i-1]+gap_open)
        h[0, i] = e[0, i]

    for i in range(1, lx+1):
        f[i, 0] = max(f[i-1, 0]+gap_extend, h[i-1, 0]+gap_open)
        h[i, 0] = f[i, 0]

    for i in range(1, lx+1):
        for j in range(1, ly+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i, j] = max(e[i, j-1]+gap_extend, h[i, j-1]+gap_open)
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)

    # return f
    return h
    # return traceback(x, y, h, e, f, lx, ly, match, mismatch, gap_open, gap_extend, start=True)

# print(nw_full([0]
# query = ''.join(alphabet[query.astype(int)])
# database = ''.join(alphabet[database.astype(int)])
# f = nw_full(query, database, match=10, mismatch=-5, gap_extend=-3, gap_open=-8)
# print(f.T[-1][1:].reshape(16, -1).T)
# # print(.T[-1][1:].reshape(16, -1).T)




