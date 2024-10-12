from ats import calign
import numpy as np
from ats import extra
import parasail
match = 3
mismatch = 2
gap_open = 5
gap_extend = 2

qfactor = 500 # 10000
dfactor = 500 # 20000
query = "Hello world"*qfactor
database = "This hello"*dfactor
print(len(query), len(database))
rc = calign.pyhirschberg(query, database, match=match, mismatch=-mismatch, gap_open=-gap_open, gap_extend=-gap_extend)
print(np.maximum(0, rc[:, :500]))
print(rc.shape)


def do_parasail_scan(x, y):
    # x = ''.join(alphabet[x.astype(int)])
    # y = ''.join(alphabet[y.astype(int)])
    alphabet = ''.join(list(np.union1d(list(x), list(y))))
    matrix = parasail.matrix_create(alphabet, match=match, mismatch=-mismatch, case_sensitive=True)
    print(matrix)
    r = parasail.nw_rowcol_scan_32(x, y, open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
    return np.copy(r.score_col)#.astype(float).reshape(SIMD_ELEM, -1).T

def do_parasail_table(x, y):
    # x = ''.join(alphabet[x.astype(int)])
    # y = ''.join(alphabet[y.astype(int)])
    alphabet = ''.join(list(np.union1d(list(x), list(y))))
    matrix = parasail.matrix_create(alphabet, match=match, mismatch=-mismatch, case_sensitive=True)
    r = parasail.nw_table_scan_32(x, y, open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
    return np.copy(r.score_table)

def nw_full(x, y, match=match, mismatch=-mismatch, gap_open=-gap_open, gap_extend=-gap_extend):
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

    return f

print(extra.chirschberg(query, database, match=match, mismatch=-mismatch, gap_open=-gap_open, gap_extend=-gap_extend)[:, :500])
# rpa = do_parasail_table(query, database)
# print(rpa)
# print(rpa.shape)

# rpp = nw_full(query, database)
# print(rpp[1:, 1:])
# print(rpp.shape)
