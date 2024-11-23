from ats import align
from ats.calign import Aligner
from ats.lang import get_lang

from ats.audio import Container
from ats.text import TextFile, SubLine
from ats.model import Model, available_models

from pathlib import Path
from itertools import chain
from tqdm.auto import tqdm
from functools import partialmethod

def joinuntil(a, n):
    l, end = 0, 0
    while end < len(a) and l < n:
        l += len(a[end].text())
        end += 1
    return ''.join(seg.text() for seg in a[:end])

def match_start(aligner, audio, text, prepend, append, nopend):
    ats, sta = {}, {}
    textcache = {}
    for ai, afile in enumerate(tqdm(audio)):
        for i, ach in enumerate(tqdm(afile)):
            if (ai, i) in ats: continue

            lang = get_lang(ach.language, prepend, append, nopend)
            acontent = joinuntil(ach.segments, 2000)

            best = (-1, -1, 0)
            for ti, tfile in enumerate(text):
                for j, tch in enumerate(tfile.chapters):
                    if (ti, j) in sta: continue

                    if (ti, j) not in textcache:
                        textcache[ti, j] = joinuntil(tfile.chapters[j].text(), 2000)
                    tcontent = textcache[ti, j]
                    if len(acontent) < 100 or len(tcontent) < 100: continue

                    limit = min(len(tcontent), len(acontent))
                    score = aligner.similarity(acontent[:limit], tcontent[:limit]) / (len(acontent[:limit]) + len(tcontent[:limit])) * 100 + 50
                    if score > 40 and score > best[-1]:
                        best = (ti, j, score)

            if best[:-1] in sta:
                tqdm.write("WARNING match_start")
            elif best != (-1, -1, 0):
                ats[ai, i] = best
                sta[best[:-1]] = (ai, i, best[-1])

    return ats, sta

def expand_matches(audio, text, ats, sta):
    batches = []
    for ai, a in enumerate(audio):
        batch = []

        i = 0
        while i < len(a.chapters):
            astart = i
            aend = i+1
            if (ai, astart) not in ats:
                i = aend
                continue
            while aend < len(a.chapters) and (ai, aend) not in ats:
                aend += 1

            book, tstart, score = ats[ai, astart]
            tend = tstart+1
            while tend < len(text[book].chapters) and (book, tend) not in sta:
                tend += 1
            batch.append((astart, aend, book, tstart, tend, score))
            i = aend
        batches.append(batch)
    return batches


def print_batches(batches, audio, text, spacing=2, sep1='=', sep2='-', sep3='::'):
    from wcwidth import wcswidth

    rows = [1, ["Audio", "Text", "Score"]]
    width = [wcswidth(h) for h in rows[-1]]

    for ai, batch in enumerate(batches):
        use_audio_header = len(audio[ai].chapters) > 1

        text_unique = len(set(b[-4] for b in batch)) == 1
        use_text_header = text_unique and (batch[0][-2] - batch[0][-3]) > 3

        if use_audio_header or use_text_header:
            rows.append(1)
            rows.append([audio[ai].title, '', ''])
            use_audio_header = True
            if text_unique:
                rows[-1][1] = text[batch[0][-4]].title
                use_text_header = True
            width[0] = max(width[0], wcswidth(rows[-1][0]))
            width[1] = max(width[1], wcswidth(rows[-1][1]))
        rows.append(1)
        for astart, aend, book, tstart, tend, score in batch:
            a = [audio[ai].chapters[i] for i in range(astart, aend)]
            t = [text[book].chapters[i] for i in range(tstart, tend)]
            for i in range(max(len(a), len(t))):
                row = ['', '' if t else '?', '']
                if i < len(a):
                    row[0] = (audio[ai].title + sep3 if not use_audio_header else '') + a[i].title.strip()
                    width[0] = max(width[0], wcswidth(row[0]))
                if i < len(t):
                    row[1] = (text[book].title + sep3 if not use_text_header else '') + t[i].title.strip()
                    width[1] = max(width[1], wcswidth(row[1]))
                if i == 0:
                    row[2] = format(score/100, '.2%') if score is not None else '?'
                    width[2] = max(width[2], wcswidth(row[2]))
                rows.append(row)
            rows.append(2)
        rows = rows[:-1]
    rows.append(1)

    for row in rows:
        csep = ' ' * spacing
        if isinstance(row, int):
            sep = sep1 if row == 1 else sep2
            print(csep.join([sep*w for w in width]))
            continue
        print(csep.join([r.ljust(width[i]-wcswidth(r)+len(r)) for i, r in enumerate(row)]))

def to_epub():
    pass

def to_subs(text, subs, alignment):
    segments = []
    for ai, a in enumerate(alignment):
        if a[0] == -1:
            continue
        ts, te = a[0], a[1]
        tso, teo = a[2], a[3]
        line = ''.join([text[i].text() for i in range(ts, te)])
        line =  line[tso:-len(text[te-1].text())+teo]
        s = subs[ai]
        if False and line.strip(): # Debug
            line = s.text()+'\n'+line
        segments.append(SubLine(content=line if line.strip() else '＊'+s.text(), start=s.start, end=s.end))
    return segments

def prompt(message, lchoices):
    if lchoices == 0:
        return []
    while True:
        inp = input(message) # Taken from yay
        r = set()
        for a in inp.split():
            try:
                if a[0] == '^':
                    val = int(a[1:])
                    r = r.union(range(l)) - {val}
                elif len(k := a.split('-')) > 1:
                    val1 = min(int(k[0]), l-1)
                    val2 = min(int(k[1]), l-1)
                    r = r.union(range(val1, val2+1))
                else:
                    if (val1 := int(a)) < l:
                        r.add(val1)
            except ValueError:
                print("Parsing failed")
                continue
        return r

def select_streams(audio):
    return [a.streams[a.default_stream] for a in audio]

def whisper(audio, text, language, output_dir, output_format, file_overwrite,
            model, device, batch_size,
            local_only, memsize, quantize,
            use_cache, cache_dir, overwrite_cache,
            prepend_punctuations, append_punctuations, nopend_punctuations,
            **model_args):
    # TODO redo the cache
    model = Model(model, device, quantize=quantize, local_files_only=local_only)
    print(f"Using device: {model.device} with {model.compute_type} compute.")
    transcription = model.transcribe(select_streams(audio), batch_size=batch_size, num_chunks=10,
                                     language=language, use_stream_language=False, **model_args)

    aligner = Aligner(memsize=memsize, match=1, mismatch=-1, gap_open=-1, gap_extend=-1)
    print('Fuzzy matching chapters...')
    ats, sta = match_start(aligner, transcription, text, prepend_punctuations, append_punctuations, nopend_punctuations)
    audio_batches = expand_matches(audio, text, ats, sta)
    print_batches(audio_batches, audio, text)

    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (audio[ai].path.stem + '.' + output_format)
            if not file_overwrite and out.exists():
                bar.write(f"{out.name} already exists, skipping.")
                continue

            bar.set_description(audio[ai].path.name)
            segments = []
            for astart, aend, book, tstart, tend, _ in tqdm(batches):
                language = get_lang(transcription_grouped[ai][astart].language, prepend_punctuations, append_punctuations, nopend_punctuations)
                tcontent = [s for i in range(tstart, tend) for s in text[book].chapters[i].text()]
                acontent = [s for i in range(astart, aend) for s in transcription_grouped[ai][i].segments]
                alignment, references = align.align(aligner, language, acontent, tcontent, [], set(), set(), set())
                segments.extend(to_subs(tcontent, acontent, alignment))

            if not segments:
                continue

            with out.open("w", encoding='utf8') as o:
                if output_format == "srt":
                    o.write('\n\n'.join(str(i+1)+'\n'+s.vtt(use_comma=True) for i, s in enumerate(segments)))
                elif output_format == 'vtt':
                    o.write("WEBVTT\n\n"+'\n\n'.join(s.vtt() for s in segments))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument("--text", type=Path, required=True, default=[], action='append', help="path to the script file")

    parser.add_argument("--audio", type=Path, required=True, default=[], action='append', help="list of audio files to process")
    parser.add_argument("--language", default=None, help="language of the script and audio")

    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", default=False,  help="overwrite any destination files", action=argparse.BooleanOptionalAction)

    parser.add_argument("--output-dir", default=u'.', type=Path, help="output directory")
    parser.add_argument("--output-format", default='srt', help="output format currently only supports vtt and srt")

    parser.add_argument("--model", default="tiny", help=f"whisper model to use, can be a huggingface path or one of {available_models()}")
    parser.add_argument("--device", default='auto', help="device to do inference on")
    parser.add_argument("--local-only", default=False, help="Don't download models", action=argparse.BooleanOptionalAction)
    parser.add_argument("--memsize", type=int, default=int(1*1024**3), help="amount of memory to use for alignment (in bytes)")

    parser.add_argument("--use-cache", default=True, help="use the transcription cache", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite-cache", default=False, help="always overwrite the cache", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="Cache directory")

    parser.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch-size", type=int, default=4, help="number of batches to do at once")

    parser.add_argument("--beam-size", type=int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--num-hypotheses", type=int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--length-penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--repetition-penalty", type=float, default=1, help="penalty applied to the score of previously generated tokens")
    parser.add_argument("--no-repeat-ngram-size", type=float, default=0, help="penalty applied to the score of previously generated tokens")
    parser.add_argument("--max-initial-timestamp-index", type=lambda x: int(x)//0.02, default=1500, help="maximum index of the first predicted timestamp")

    parser.add_argument("--suppress-blank", default=True, help="suppress blank tokens at the start of sampling", action=argparse.BooleanOptionalAction)
    parser.add_argument("--suppress-tokens", type=str, default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")

    parser.add_argument("--temperatures", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1], nargs='+', help="temperature(s) to use for sampling")
    parser.add_argument("--sampling-topk", type=int, default=0, help="only use the top k tokens for sampling")
    parser.add_argument("--logprob-threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--nospeech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `log_prob_threshold`, consider the segment as silence")

    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    args = parser.parse_args().__dict__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop('progress'))

    language = args.pop('language')

    print("Loading...")
    audio = list(chain.from_iterable([Container.from_file(f)] if f.is_file() else Container.from_dir(f) for f in args.pop('audio')))
    text  = list(chain.from_iterable([TextFile.from_file(f)] if f.is_file() else TextFile.from_dir(f) for f in args.pop('text')))

    output_dir = args.pop('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')

    whisper(audio, text, language, output_dir, output_format, args.pop('overwrite'), **args)
