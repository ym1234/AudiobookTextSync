import numpy as np
from pprint import pprint

def traceback(x, y, H, E, F, cx, cy, match, mismatch, gap_open, gap_extend, start=False):
    cur, traceback = 0, []
    while cx > 0 and cy > 0:
        if cur == 0:
            score = match if x[cx-1] == y[cy-1] else mismatch
            if H[cx, cy] == E[cx, cy]:
                cur = 1
            elif H[cx, cy] == F[cx, cy]:
                cur = 2
            elif (H[cx-1, cy-1] + score) == H[cx, cy]:
                traceback.append((cx, cy))
                cx, cy = cx-1, cy-1
        elif cur == 1:
            traceback.append((cx, cy))
            if (H[cx, cy-1] + gap_open) == E[cx, cy] and (E[cx, cy-1] + gap_extend) != E[cx, cy]:
                cur = 0
            cy -= 1
        elif cur == 2:
            traceback.append((cx, cy))
            if (H[cx-1, cy] + gap_open) == F[cx, cy] and (F[cx-1, cy] + gap_extend) != F[cx, cy]:
                cur = 0
            cx -= 1

    if start:
        while cx > 0:
            traceback.append((cx, cy))
            cx -= 1

        while cy > 0:
            traceback.append((cx, cy))
            cy -= 1

    return np.array(traceback)[::-1].swapaxes(-1, -2)-1

def semiglobal(x, y, match=-1, mismatch=-1, gap_open=-1, gap_extend=-1, recursive=False, end=False):
    lx, ly = len(x), len(y)

    h = np.zeros((lx+1, ly+1))
    e = np.full((lx+1, ly+1), fill_value=-np.inf)
    f = np.full((lx+1, ly+1), fill_value=-np.inf)

    # Gap extends being MORE than than gap opens doesn't make much sense
    for i in range(1, ly+1):
        e[0, i] = max(e[0, i-1]+gap_extend, h[0, i-1]+gap_open)
        h[0, i] = e[0, i]

    for i in range(1, lx+1):
        for j in range(1, ly+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i, j] = max(e[i, j-1]+gap_extend, h[i, j-1]+gap_open)
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)

    return h, e, f
    # cx = lx if end else h[:, -1].T.argmax()
    # return traceback(x, y, h, e, f, cx, ly, match, mismatch, gap_open, gap_extend)

def tracep(x, y, trace):
    x, y = np.array(list(x)), np.array(list(y))
    print(''.join(x[trace[0]].tolist()))
    print(''.join(y[trace[1]].tolist()))
    print()

def slastcol(x, y, match, mismatch, gap_open, gap_extend, reverse=False):
    if reverse:
        x = x[::-1]
        y = y[::-1]
    lx, ly = len(x), len(y)
    h = np.zeros((lx+1,))
    e = np.full((lx+1,), fill_value=-np.inf)
    h_prev = 0

    for j in range(1, ly+1):
        f = -np.inf
        h_prev, h[0] = h[0], gap_open + (j-1) * gap_extend
        for i in range(1, lx+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i] = max(e[i]+gap_extend, h[i]+gap_open)
            f = max(f+gap_extend, h[i-1]+gap_open)
            h_prev, h[i] = h[i], max(e[i], f, h_prev + score)
    return h

def lastcol(x, y, match, mismatch, gap_open, gap_extend, reverse=False):
    if reverse:
        x = x[::-1]
        y = y[::-1]
    lx, ly = len(x), len(y)
    h = np.zeros((lx+1,))
    e = np.full((lx+1,), fill_value=-np.inf)
    h_prev = 0

    f = -np.inf
    for i in range(1, lx+1):
        f = max(f+gap_extend, h[i-1]+gap_open)
        h[i] = f

    for j in range(1, ly+1):
        f = -np.inf
        h[0], h_prev = gap_open + (j-1) * gap_extend, h[0]
        for i in range(1, lx+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i] = max(e[i]+gap_extend, h[i]+gap_open)
            f = max(f+gap_extend, h[i-1]+gap_open)
            h_prev, h[i] = h[i], max(e[i], f, h_prev + score)
    return h, e

def nw(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1):
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

    return traceback(x, y, h, e, f, lx, ly, match, mismatch, gap_open, gap_extend, start=True)

def hirschberg_inner(x, y, match, mismatch, gap_open, gap_extend):
    lx, ly = len(x), len(y)
    if lx < 2 or ly < 2:
        return nw(x, y, match, mismatch, gap_open, gap_extend)

    f, fe = lastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
    s, se = lastcol(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)

    j =  f + s[::-1]
    k =  fe + se[::-1] - gap_open
    # mid, mid2 = len(j) - j[::-1].argmax() - 1, len(k) - k[::-1].argmax() - 1
    mid, mid2 = j.argmax(), k.argmax()

    if j[mid] >= k[mid2]:
        split1 = hirschberg_inner(x[:mid], y[:ly//2], match, mismatch, gap_open, gap_extend)
        split2 = hirschberg_inner(x[mid:], y[ly//2:], match, mismatch, gap_open, gap_extend)
        return np.concatenate([split1, np.array([[mid], [ly//2]]) + split2], axis=1)

    split1 = hirschberg_inner(x[:mid2], y[:ly//2-1], match, mismatch, gap_open, gap_extend)
    split2 = hirschberg_inner(x[mid2:], y[ly//2+1:], match, mismatch, gap_open, gap_extend)
    return np.concatenate([split1, np.array([[mid2-1], [ly//2-1]]), np.array([[mid2-1], [ly//2]]), np.array([[mid2], [ly//2+1]]) + split2], axis=1)

# Check if this is correct?
def tracecheck(x, y, trace, H, E, F, match, mismatch, gap_open, gap_extend, start=False):
    trace = trace.T
    cx, cy = trace[-1]
    cur, idx = 0, 1
    while idx < trace.shape[0] and cx >= 0 and cy >= 0:
        nx, ny = trace[trace.shape[0] - idx - 1]
        if cur == 0:
            score = match if x[cx] == y[cy] else mismatch
            if nx == cx-1 and ny == cy-1:
                if (H[nx+1, ny+1] + score) != H[cx+1, cy+1]:
                    return idx
            elif nx == cx and ny == cy-1:
                if H[cx+1, cy+1] != E[cx+1, cy+1]:
                    return idx
                cur = 1
                continue
            elif nx == cx-1 and ny == cy:
                if H[cx+1, cy+1] != F[cx+1, cy+1]:
                    return idx
                cur = 2
                continue
            else:
                return idx
        elif cur == 1:
            if nx != cx or ny != cy - 1:
                return idx
            if H[nx+1, ny+1] + gap_open == E[cx+1, cy+1]:
                cur = 0
            elif E[nx+1, ny+1] + gap_extend == E[cx+1, cy+1]:
                pass
            else:
                return idx
        elif cur == 2:
            if nx != cx - 1 or ny != cy:
                return idx
            if H[nx+1, ny+1] + gap_open == F[cx+1, cy+1]:
                cur = 0
            elif F[nx+1, ny+1] + gap_extend == F[cx+1, cy+1]:
                pass
            else:
                return idx
        cx, cy = nx, ny
        idx += 1
    return True

def pyhirschberg(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1, end=False):
    start, last = 0, len(x)
    if not end:
        last = slastcol(x, y, match, mismatch, gap_open, gap_extend).argmax()
        start = last - slastcol(x[:last], y, match, mismatch, gap_open, gap_extend, reverse=True).argmax()
    trace = np.array([[start], [0]]) + hirschberg_inner(x[start:last], y, match, mismatch, gap_open, gap_extend)
    if not end:
        h, e, f = semiglobal(x, y, match, mismatch, gap_open, gap_extend, end=end)
        if (k := tracecheck(x, y, trace, h, e, f, match, mismatch, gap_open, gap_extend)) != True:
            print(trace)
            raise Exception(f"Bad trace {k}")
    return trace
