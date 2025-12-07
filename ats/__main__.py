from ats.lang import get_lang
from pprint import pprint
from tqdm.auto import tqdm
from ats.args import make_forward_option, make_input_option
import os
import pickle

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
        for i, ach in enumerate(tqdm(afile.chapters)):
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

# Batch = namedtuple('Batch', ['book', 'start', 'text', 'score'])
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
    from ats.text import SubLine
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
    from ats import align
    from ats.calign import Aligner
    # TODO redo the cache

    aligner = Aligner(memsize=memsize, match=1, mismatch=-2, gap_open=-2, gap_extend=-1)
    print('Fuzzy matching chapters...')
    ats, sta = match_start(aligner, transcription, text, prepend_punctuations, append_punctuations, nopend_punctuations)
    audio_batches = expand_matches(audio, text, ats, sta)
    print_batches(audio_batches, audio, text)

    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (audio[ai].path.stem + '.' + output_format)

            bar.set_description(audio[ai].path.name)
            segments = []
            for astart, aend, book, tstart, tend, _ in tqdm(batches):
                language = get_lang(transcription[ai].chapters[astart].language, prepend_punctuations, append_punctuations, nopend_punctuations)
                tcontent = [s for i in range(tstart, tend) for s in text[book].chapters[i].text()]
                acontent = [s for i in range(astart, aend) for s in transcription[ai].chapters[i].segments]
                alignment, references = align.align(aligner, language, acontent, tcontent, [], set(), set(), set())
                segments.extend(to_subs(tcontent, acontent, alignment))

            if not segments:
                continue

            with out.open("w", encoding='utf8') as o:
                if output_format == "srt":
                    o.write('\n\n'.join(str(i+1)+'\n'+s.vtt(use_comma=True) for i, s in enumerate(segments)))
                elif output_format == 'vtt':
                    o.write("WEBVTT\n\n"+'\n\n'.join(s.vtt() for s in segments))



def cache_main(args):
    pass

if __name__ == "__main__":
    import argparse

    from ats.text import TextFile
    from ats.audio import Container
    from ats.model import available_models, Model
    from ats.cache import Cache

    from functools import partialmethod
    from pathlib import Path
    from itertools import chain


    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    sync_parser = subparsers.add_parser('sync', help="Sync!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sync_parser.add_argument("--cache-path", type=Path, default="TranscriptionCache.sqlite", help="path to cache database")
    sync_parser.add_argument("--progress", default=True, action=argparse.BooleanOptionalAction,  help="progress bar on/off")
    sync_parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction,  help="overwrite any destination files")

    sync_parser.add_argument("--output-dir", default=u'.', type=Path, help="output directory")
    sync_parser.add_argument("--output-format", default='srt', choices=['srt', 'vtt', 'epub'], help="output format, epub will create an epub file with a media overlay")

    global_option =  make_forward_option('')
    sync_parser.add_argument("--language", action=global_option, help="language of the script and audio")

    text_action = make_input_option('text')
    text_option =  make_forward_option('text')

    text_group = sync_parser.add_argument_group("Text options")
    text_group.add_argument("--text", type=TextFile.from_file, action=text_action, required=True, help="path to an epub or text file")

    audio_action = make_input_option('audio')
    audio_option =  make_forward_option('audio')
    audio_group = sync_parser.add_argument_group("Audio options")
    audio_group.add_argument("--audio", type=Container.from_file, action=audio_action, required=True, help="path to an audio file")
    audio_group.add_argument("--stream", type=int, action=audio_option, help="stream language or index with in the audio file to use")
    audio_group.add_argument("--ignore", nargs="*", action=audio_option, help="chapters to ignore while aligning and transcriping")
    audio_group.add_argument("--cache-entry", type=int, action=audio_option, help="cache id for the audio file")

    model_group = sync_parser.add_argument_group("Model options")
    model_group.add_argument("--model", default="tiny", help=f"whisper model to use, can be a huggingface path or one of {available_models()}")
    model_group.add_argument("--device", default='auto', help="device to do inference on")
    model_group.add_argument("--local-files-only", default=False, action=argparse.BooleanOptionalAction, help="Don't download models")
    model_group.add_argument('--quantize', default=True, action=argparse.BooleanOptionalAction, help="use fp16 on gpu or int8 on cpu")
    model_group.add_argument('--download-root', default=None, help="directory to download the model to")
    model_group.add_argument('--device-index', type=int, default=0, help="device to load the model on")

    aligner_group = sync_parser.add_argument_group("Aligner options")
    aligner_group.add_argument("--memsize", type=int, default=int(1*1024**3), help="amount of memory to use for alignment (in bytes)")
    aligner_group.add_argument("--prepend_punctuations", default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    aligner_group.add_argument("--append_punctuations", default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    aligner_group.add_argument("--nopend_punctuations", default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    transcription_group = sync_parser.add_argument_group("Transcription options")
    transcription_group.add_argument("--use-stream-language", type=bool, action=argparse.BooleanOptionalAction)

    transcription_group.add_argument("--num-chunks", type=int, default=10, help="todo")
    transcription_group.add_argument("--batch-size", type=int, default=4, help="number of batches to do at once")
    transcription_group.add_argument("--beam-size", type=int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    transcription_group.add_argument("--patience", type=float, default=1, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    transcription_group.add_argument("--num-hypotheses", type=int, default=5, help="number of candidates when sampling with non-zero temperature")
    transcription_group.add_argument("--length-penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    transcription_group.add_argument("--repetition-penalty", type=float, default=1, help="penalty applied to the score of previously generated tokens")
    transcription_group.add_argument("--no-repeat-ngram-size", type=float, default=0, help="penalty applied to the score of previously generated tokens")
    transcription_group.add_argument("--max-initial-timestamp-index", type=lambda x: int(x)//0.02, default=1500, help="maximum index of the first predicted timestamp")

    transcription_group.add_argument("--suppress-blank", default=True, action=argparse.BooleanOptionalAction, help="suppress blank tokens at the start of sampling")
    transcription_group.add_argument("--suppress-tokens", default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")

    transcription_group.add_argument("--temperatures", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1], nargs='+', help="temperature(s) to use for sampling")
    transcription_group.add_argument("--sampling-topk", type=int, default=0, help="only use the top k tokens for sampling")
    transcription_group.add_argument("--logprob-threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    transcription_group.add_argument("--nospeech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `log_prob_threshold`, consider the segment as silence")

    cache_parser = subparsers.add_parser('cache', help='manage the cache')

    cache_parser.add_argument("--cache-path", type=Path, default="TranscriptionCache.sqlite", help="path to cache database")
    cache_parser.add_argument('--list', help="list all entries in the cache")
    cache_parser.add_argument('--remove', type=int, nargs="+", help="delete cache entries")

    args = parser.parse_args()

    cache = Cache(args.cache_path)
    if args.command == 'cache':
        cache_main(args, cache)
        exit(0)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.progress)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ao = {a.dest: getattr(args, a.dest) for a in aligner_group._group_actions}

    print("Loading...")
    text = args.text
    # pprint(text)

    cacheless_streams, cached_streams = [], []
    for r in args.audio:
        if 'cache_entry' not in r:
            cacheless_streams.append(r)
            continue
        entry_id = r['cache_entry']
        try:
            entry = cache.get(entry_id)
            cached_streams.append({**r, 'transcript': entry})
        except:
            cacheless_streams.append(r)
            print(f"couldn't find cache entry {entry_id}, transcribing {r[file].path.name}")

    if len(cacheless_streams):
        model = Model(**{a.dest: getattr(args, a.dest) for a in model_group._group_actions})
        print(f"Using device: {model.device} with {model.compute_type} compute.")

        transcripts = model.transcribe(cacheless_streams, **{a.dest: getattr(args, a.dest) for a in transcription_group._group_actions})
        for r, t in zip(cacheless_streams, transcript):
            cache.put(r, transcript)
            cached_streams.append({**r, 'transcript': t})

