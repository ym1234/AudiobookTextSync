import numpy as np
from pprint import pprint

def idk(cx, cy, m):
    if m == 0:
        return cx-1, cy-1
    elif m == 1:
        return cx-1, cy
    else:
        return cx, cy-1

def get_traceback(h, cx, cy, recursive=False):
    traceback = []
    while cx > 0 and cy > 0:
        traceback.append((cx, cy))
        nodes = h[[cx-1, cx-1, cx], [cy-1, cy, cy-1]]
        maxes = (nodes == nodes.max()).nonzero()[0]
        if recursive and len(maxes) > 1:
            tracebacks, added = [], False
            for m in maxes:
                tcx, tcy = idk(cx, cy, m)
                if tcx == 0 or tcy == 0:
                    if not added:
                        tracebacks.append(traceback)
                    added = True
                    continue

                z = get_traceback(h, tcx, tcy, recursive)
                z = z if type(z[0]) is list else [z]
                for k in z:
                    tracebacks.append(traceback + k)
            return tracebacks

        cx, cy = idk(cx, cy, maxes[0])
    return traceback

def semiglobal(x, y, gap_open=-1, gap_extend=-1, match=1, mismatch=-1, recursive=False):
    lx, ly = len(x), len(y)

    h = np.zeros((lx+1, ly+1))
    e = np.zeros((lx+1, ly+1))
    f = np.zeros((lx+1, ly+1))

    # Gap extends being MORE than than gap opens doesn't make much sense
    f[0, 1], h[0, 1] = gap_open, gap_open
    for i in range(2, ly+1):
        f[0, i] = f[0, i-1] + gap_extend
        h[0, i] = f[0, i]

    for i in range(1, lx+1):
        for j in range(1, ly+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i, j] = max(e[i, j-1]+gap_extend, h[i, j-1]+gap_open)
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)

    maximum = int(h[:, -1].argmax())
    try:
        traceback = get_traceback(h, maximum, ly, recursive)
        traceback = np.array(traceback) # This isn't guaranteed to  be homogeneous
        traceback = traceback[:, ::-1] if recursive else traceback[::-1]
        traceback = traceback.swapaxes(-1, -2) - 1
    except:
        traceback = None

    return traceback, h, e, f

def semiglobal_print(x, y, gap_open=-1, gap_extend=-1, match=1, mismatch=-1, recursive=False):
    traceback = semiglobal(x, y, gap_open, gap_extend, match, mismatch, recursive)[0]
    if not recursive: traceback = traceback.reshape(1, *traceback.shape)

    x, y = np.array(list(x)), np.array(list(y))
    for t1, t2 in traceback:
        print(''.join(x[t1].tolist()))
        print(''.join(y[t2].tolist()))
        print()


def lastcol(x, y, match, mismatch, gap_open, gap_extend):
    lx, ly = len(x), len(y)
    h = np.zeros((lx+1))
    e = np.zeros((lx+1))
    f = np.zeros((lx+1))
    h_prev = 0

    for j in range(1, ly+1):
        f[0] = gap_open + (j-1) * gap_extend
        h[0], h_prev = f[0], h[0]
        for i in range(1, lx+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i] = max(e[i]+gap_extend, h[i]+gap_open)
            f[i] = max(f[i-1]+gap_extend, h[i-1]+gap_open)
            h_prev, h[i] = h[i], max(e[i], f[i], h_prev + score)
    return h

def hirschberg_inner(x, y, match, mismatch, gap_open, gap_extend):
    lx, ly = len(x), len(y)
    if lx == 0:
        return np.vstack((np.arange(len(y)), np.zeros(len(y)))).T
    if ly == 0:
        return np.vstack((np.arange(len(x)), np.zeros(len(x)))).T

    if lx == 1:
        return np.array([(0, lastcol(x, y, match, mismatch, gap_open, gap_extend).argmax()-1)])
    if ly == 1:
        return np.array([(lx - lastcol(y, x, match, mismatch, gap_open, gap_extend).argmax(), 0)])

    f = lastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
    s = lastcol(x[::-1], y[ly//2:][::-1], match, mismatch, gap_open, gap_extend)
    print(f)
    print(s)
    mid = (f + s[::-1]).argmax()

    return np.concatenate((hirschberg_inner(x[:mid], y[:ly//2], match, mismatch, gap_open, gap_extend),
                           [(mid, ly//2)],
                           np.array([(mid+1, ly//2+1)]) + hirschberg_inner(x[mid+1:], y[ly//2+1:], match, mismatch, gap_open, gap_extend)), axis=0)

def hirschberg(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1):
    # last = lastcol(x, y, gap_open, gap_extend, match, mismatch).argmax()
    last = len(x)
    return hirschberg_inner(x[:last], y, match, mismatch, gap_open, gap_extend).T

