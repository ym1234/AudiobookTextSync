import numpy as np

def align_sub(coords1, coords2, ends1, ends2):
    ends2idx = np.searchsorted(coords2, ends2, side='right')
    segstart, segend, prev, ret = 0, 0, 0, []
    for k in ends2idx[:-1]:
        cend = coords1[k]
        while cend >= ends1[segend]:
            segend += 1
        f = max(int(cend-ends1[segend-1]), 0)
        ret.append([segstart, segend, prev, f])
        segstart, prev = segend-1, ret[-1][-1]
    ret.append([segstart, ends1.shape[-1]-1, prev, -1])
    return ret


# # """"""Heuristics"""""""
# # Move to lang?
# def fix_punc(text, segments, prepend, append, nopend):
#     for l, s in enumerate(segments):
#         if not s: continue
#         t = text[l]
#         for p, f in zip(s, s[1:] + [s[-1]]):
#             connected = f[0] == p[1]
#             loop = 0
#             while True:
#                 if loop > 20:
#                     break
#                 if p[1] < len(t) and t[p[1]] in append:
#                     p[1] += 1
#                 elif t[p[1]-1] in prepend:
#                     p[1] -= 1
#                 elif (p[1] > 0 and t[p[1]-1] in nopend) or (p[1] < len(t) and t[p[1]] in nopend) or (p[1] < len(t)-1 and t[p[1]+1] in nopend):
#                     start, end = p[1]-1, p[1]
#                     if  p[1] < len(t)-1 and (t[p[1]+1] in nopend and 0x4e00 > ord(t[p[1]]) or ord(t[p[1]]) > 0x9faf): # Bail out if we end on a kanji
#                         end += 1

#                     while start > 0 and t[start] in nopend:
#                         start -= 1
#                     while end < len(t)-1 and t[end] in nopend:
#                         end += 1


#                     if t[start] in prepend:
#                         if p[1] == start:
#                             break
#                         p[1] = start
#                     elif t[start] in append:
#                         if p[1] == start+1:
#                             break
#                         p[1] = start+1
#                     elif end < len(t) and t[end] in prepend:
#                         if p[1] == end:
#                             break
#                         p[1] = end
#                     elif end < len(t) and t[end] in append:
#                         if p[1] == end+1:
#                             break
#                         p[1] = end+1
#                     else:
#                         break
#                 else:
#                     break
#                 loop += 1
#             if connected: f[0] = p[1]


def fix(lang, original, edited, segments):
    for s in segments:
        for i in range(2):
            t, to = s[i] - i, s[i+2]
            if to == -1:
                s[i+2] = 2*len(original[t]) # hack lol
                continue
            if to == 0:
                continue
            o, e = lang.translate(original[t]), edited[t]
            oi = 0
            for ei in range(len(e)):
                while oi < len(o) and e[ei] != o[oi]:
                    oi += 1
                while oi < len(o) and e[ei] == o[oi]:
                    oi += 1
                if ei == to:
                    to = oi-1
                    break
            s[i+2] = to


# This is structured like this to deal with references later
def align(model, aligner, lang, transcript, text, references, prepend, append, nopend):
    transcript_clean = [lang.clean(i) for i in transcript]
    transcript_joined = ''.join(transcript_clean)

    def inner(text):
        text_clean = [lang.clean(i) for i in text]
        text_joined = ''.join(text_clean)

        if not len(text_joined) or not len(transcript_joined): return []
        score, coords = aligner.hirschberg(text_joined, transcript_joined)
        segments = align_sub(coords[0], coords[1], np.cumsum([0]+[len(x) for x in text_clean]), np.cumsum([len(x) for x in transcript_clean]))
        del coords

        fix(lang, text, text_clean, segments)
        # fix_punc(text, segments, prepend, append, nopend)
        return segments

    return inner(text), [] #references
