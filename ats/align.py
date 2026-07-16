import numpy as np
from tqdm.auto import tqdm
import unicodedata

def merge_punctuation(text, segment_ends, indices, prepend, append):
    for i in range(1, len(indices)-1):
        while indices[i] not in segment_ends and indices[i] > 0 and text[indices[i]-1] in prepend:
            indices[i] -= 1

    for i in range(1, len(indices)):
        while indices[i] not in segment_ends and indices[i] < len(text) and text[indices[i]] in append:
            indices[i] += 1

def align_sub(transcript, text, transcript_joined, text_joined, text_indices, transcript_indices, prepend, append):
    transcript_lens = [0] + [len(t) for t in transcript]

    text_ends_idx = np.searchsorted(transcript_indices, np.cumsum(transcript_lens))
    text_ends = text_indices[text_ends_idx]

    text_lens = [0] + [len(t) for t in text]
    text_lens_cum = np.cumsum(text_lens)
    merge_punctuation(text_joined, text_lens_cum, text_ends,
                      prepend, append)

    segments_pos = np.clip(np.searchsorted(text_lens_cum, text_ends)-1, min=0)
    offsets = text_ends - text_lens_cum[segments_pos]

    total = np.concatenate([a.reshape(-1, 1) for a in [segments_pos[:-1], segments_pos[1:], offsets[:-1], offsets[1:]]],
                           axis=1)
    return total

# This is structured like this to deal with references later
def align(aligner, lang, transcript, text, references, prepend, append):
    transcript_clean = [lang.clean(i.text()) for i in transcript]
    # transcript_clean = [i.text() for i in transcript]
    transcript_joined = ''.join(transcript_clean)

    def inner(text):
        text_clean = [lang.clean(i.text()) for i in text]
        text_joined = ''.join(text_clean)

        if not len(text_joined) or not len(transcript_joined): return []
        score, coords = aligner.hirschberg(text_joined, transcript_joined)

        segments = align_sub(transcript_clean, text_clean,
                             transcript_joined, text_joined,
                             coords[0], coords[1],
                             prepend, append)
        del coords

        return segments

    return inner(text), [] #references
