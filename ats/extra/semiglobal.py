import numpy as np
from pprint import pprint

def traceback(x, y, H, E, F, cx, cy, match, mismatch, gap_open, gap_extend):
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
            if (H[cx, cy-1] + gap_open) == E[cx, cy]:
                cur = 0
            cy -= 1
        elif cur == 2:
            traceback.append((cx, cy))
            if (H[cx-1, cy] + gap_open) == F[cx, cy]:
                cur = 0
            cx -= 1
    return np.array(traceback)[::-1].swapaxes(-1, -2)-1
    # return traceback


def semiglobal(x, y, match=-1, mismatch=-1, gap_open=-1, gap_extend=-1, recursive=False, end=False):
    lx, ly = len(x), len(y)

    h = np.zeros((lx+1, ly+1))
    e = np.full((lx+1, ly+1), fill_value=-np.inf)
    f = np.full((lx+1, ly+1), fill_value=-np.inf)

    Th = np.zeros((lx+1, ly+1), dtype=np.int8)
    Te = np.zeros((lx+1, ly+1), dtype=np.int8)
    Tf = np.zeros((lx+1, ly+1), dtype=np.int8)

    # Gap extends being MORE than than gap opens doesn't make much sense
    for i in range(1, ly+1):
        e[0, i] = max(e[0, i-1]+gap_extend, h[0, i-1]+gap_open)
        h[0, i] = e[0, i]

    for i in range(1, lx+1):
        for j in range(1, ly+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i, j] = max(e[i, j-1]+gap_extend, h[i, j-1]+gap_open)
            Te[i, j] = 0 if e[i, j] == (e[i, j-1]+gap_extend) else 2
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            Tf[i, j] = 1 if f[i, j] == (f[i-1, j]+gap_extend) else 2
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)
            Th[i, j] = 0 if h[i, j] == e[i, j] else 1 if h[i, j] == f[i, j] else 2

    cx, cy = lx if end else h[:, -1].T.argmax(), ly
    return traceback(x, y, h, e, f, cx, cy, match, mismatch, gap_open, gap_extend)
    # cur = Th[cx, cy]
    # traceback = []
    # while cx > 0 and cy > 0:
    #     if cur == 2:
    #         cur = Th[cx, cy]
    #         if cur != 2:
    #             continue
    #         dx, dy = -1, -1
    #     elif cur == 1:
    #         cur = Tf[cx, cy]
    #         dx, dy = -1, 0
    #     else:
    #         cur = Te[cx, cy]
    #         dx, dy = 0, -1
    #     traceback.append((cx-1, cy-1))
    #     cx, cy = cx+dx, cy+dy
    # return np.array(traceback)[::-1].swapaxes(-1, -2)

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


    Th = np.zeros((lx+1, ly+1), dtype=np.int8)
    Te = np.zeros((lx+1, ly+1), dtype=np.int8)
    Tf = np.zeros((lx+1, ly+1), dtype=np.int8)

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
            Te[i, j] = 0 if e[i, j] == (e[i, j-1]+gap_extend) else 2
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            Tf[i, j] = 1 if f[i, j] == (f[i-1, j]+gap_extend) else 2
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)
            Th[i, j] = 0 if h[i, j] == e[i, j] else 1 if h[i, j] == f[i, j] else 2

    cx, cy = lx, ly
    cur = Th[cx, cy]
    traceback = []
    while cx > 0 and cy > 0:
        if cur == 2:
            cur = Th[cx, cy]
            if cur != 2:
                continue
            dx, dy = -1, -1
        elif cur == 1:
            cur = Tf[cx, cy]
            dx, dy = -1, 0
        else:
            cur = Te[cx, cy]
            dx, dy = 0, -1
        traceback.append((cx-1, cy-1))
        cx, cy = cx+dx, cy+dy

    return np.array(traceback)[::-1].swapaxes(-1, -2)

def hirschberg_inner(x, y, match, mismatch, gap_open, gap_extend):
    lx, ly = len(x), len(y)
    if lx == 0:
        return np.array([[0]*ly, range(ly)])
    if ly == 0:
        return np.array([range(lx), [0]*lx])

    if lx < 3 or ly < 3:
        return nw(x, y, match, mismatch, gap_open, gap_extend)

    f, fe = lastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
    s, se = lastcol(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)

    j =  f + s[::-1]
    k =  fe + se[::-1] - gap_open
    mid, mid2 = j.argmax(), k.argmax()
    print(mid, mid2)

    if mid == mid2 or j[mid] >= k[mid2]:
        split1 = hirschberg_inner(x[:mid], y[:ly//2], match, mismatch, gap_open, gap_extend)
        split2 = hirschberg_inner(x[mid:], y[ly//2:], match, mismatch, gap_open, gap_extend)
        return np.concatenate([split1, np.array([[mid], [ly//2]]) + split2], axis=1)
    else:
        print("HERE", mid2, k[mid2], mid, j[mid])
        split1 = hirschberg_inner(x[:mid2], y[:ly//2-1], match, mismatch, gap_open, gap_extend)
        split2 = hirschberg_inner(x[mid2:], y[ly//2+1:], match, mismatch, gap_open, gap_extend)
        return np.concatenate([split1, np.array([[-np.inf], [-np.inf]]), np.array([[mid2], [ly//2+1]]) + split2], axis=1)


def pyhirschberg(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1, end=False):
    if end:
        start = 0
        last = len(x)
    else:
        last = slastcol(x, y, match, mismatch, gap_open, gap_extend)
        last = last.argmax()
        start = slastcol(x[:last], y, match, mismatch, gap_open, gap_extend, reverse=True)
        start = last - start.argmax()
    return np.array([[start], [0]]) + hirschberg_inner(x[start:last], y, match, mismatch, gap_open, gap_extend)

