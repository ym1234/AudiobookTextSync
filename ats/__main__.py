import ats.align
import ats.calign

import time

from ats.lang import get_lang

from ats.audio import AudioFile #, TranscribedAudioStream, TranscribedAudioFile
from ats.text import TextFile, SubFile, SubLine
from ats.model import Model

from pathlib import Path
from itertools import chain
from wcwidth import wcswidth
from dataclasses import dataclass
from tqdm.auto import tqdm
from functools import partialmethod
from pprint import pprint

# this feels bad, idk
# sqlite?
@dataclass(eq=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool

    def get_name(self, filename, chid):
        return filename + '.' + str(chid) +  '.' + self.model_name + ".subs"

    def get(self, filename, chid): # TODO Fix this crap
        if not self.enabled: return
        fn = self.get_name(filename, chid)
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        fn =  self.get_name(filename, chid)
        p = cd / fn

        if 'text' in content:
            del content['text']
        if 'ori_dict' in content:
            del content['ori_dict']

        # Some of these may be useful but they just take so much space
        for i in content['segments']:
            if 'words' in i:
                del i['words']
            del i['id']
            del i['tokens']
            del i['avg_logprob']
            del i['temperature']
            del i['seek']
            del i['compression_ratio']
            del i['no_speech_prob']

        p.write_bytes(repr(content).encode('utf-8'))
        return content

def match_start(aligner, audio, text, prepend, append, nopend):
    ats, sta = {}, {}
    textcache = {}
    for ai, afile in enumerate(tqdm(audio)):
        for i, ach in enumerate(tqdm(afile.chapters)):
            if (ai, i) in ats: continue

            lang = get_lang(ach.language, prepend, append, nopend)
            # acontent = lang.normalize(lang.clean(''.join(seg['text'] for seg in ach.segments)))
            l, end = 0, 0
            while end < len(ach.segments) and l < 2000:
                l += len(ach.segments[end]['text'])
                end += 1
            acontent = ''.join(seg['text'] for seg in ach.segments[:end])

            best = (-1, -1, 0)
            for ti, tfile in enumerate(text):
                for j, tch in enumerate(tfile.chapters):
                    if (ti, j) in sta: continue

                    if (ti, j) not in textcache:
                        li = tfile.chapters[j].text()
                        l, end = 0, 0
                        while end < len(li) and l < 2000:
                            l += len(li[end].text())
                            end += 1
                        textcache[ti, j] = ''.join(p.text() for p in li[:end])
                        # textcache[ti, j] = lang.normalize(lang.clean(''.join(p.text() for p in tfile.chapters[j].text())))
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

def to_subs(text, subs, alignment, offset, references):
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
            line = s['text']+'\n'+line
        segments.append(SubLine(idx=ai, content=line if line.strip() else '＊'+s['text'], start=s['start']+offset, end=s['end']+offset))
    return segments

# def do_batch(aligner, ach, tch, prepend, append, nopend, offset):
#     acontent = []
#     boff = 0
#     for a in ach:
#         for p in a[0].segments:
#             p['start'] += boff
#             p['end'] += boff
#             acontent.append(p)
#         boff += a[1]

#     language = get_lang(ach[0][0].language, prepend, append, nopend)

#     tcontent = [p for t in tch for p in t.text()]
#     alignment, references = align.align(None, aligner, language, [p['text'] for p in acontent], [p.text() for p in  tcontent], [], set(prepend), set(append), set(nopend))
#     return to_subs(tcontent, acontent, alignment, offset, None)

# def faster_transcribe(model, audiofile, idx, **args):
#     gen, info = model.transcribe(audiofile.chapters[idx].audio(), best_of=1, **args)
#     segments, prev_end = [], 0
#     with tqdm(total=info.duration, unit_scale=True, unit=" seconds") as pbar:
#         pbar.set_description(audiofile.chapters[idx].title)
#         for segment in gen:
#             segments.append(segment._asdict())
#             pbar.update(segment.end - prev_end)
#             prev_end = segment.end
#         pbar.update(info.duration - prev_end)
#         pbar.refresh()

#     return {'segments': segments, 'language': args['language'] if 'language' in args else info.language}


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


def alass(audio, text, language, output_dir, output_format, overwrite,
          path, args, sort):
    import tempfile
    import subprocess
    import torch # TODO
    from natsort import natsorted

    audio = natsorted(audio, lambda x: x.path.name) if sort else audio
    text = natsorted(text, lambda x: x.path.name) if sort else text

    if not all(isinstance(t, SubFile) for t in text):
        print('alass inputs should be subtitle files not epubs or text files')
        return
    if len(audio) != len(text):
        print("len(audio) != len(text), input needs to be in order for alass alignment")
        return
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True) # onnx is much faster
    (get_speech_timestamps, *_) = utils

    with tqdm(zip(audio, text), total=len(audio)) as bar:
        for a, t in bar:
            bar.set_description(f'Running VAD on {a.title}')
            v = get_speech_timestamps(a.audio(), model, sampling_rate=16000, return_seconds=True)
            bar.set_description(f'Aligning {t.title} with {a.title}')
            segments = [Segment(text='h', start=s['start'], end=s['end']) for s in v]
            with tempfile.NamedTemporaryFile(mode="w", suffix='.srt') as f:
                f.write('\n\n'.join(str(i+1)+'\n'+s.vtt(use_comma=True) for i, s in enumerate(segments)))
                cmd = [path, *['-'+h for h in args], f.name, str(t.path), str(output_dir / (a.path.stem + ''.join(t.path.suffixes)))]
                bar.write(' '.join(cmd))
                try:
                    subprocess.run(cmd)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Alass command failed: {e.stderr.decode()}\n args: {' '.join(cmd)}") from e


def whisper(audio, text, language, output_dir, output_format, file_overwrite,
            model, device, batch_size,
            local_only, memsize, quantize,
            use_cache, cache_dir, overwrite_cache,
            prepend_punctuations, append_punctuations, nopend_punctuations,
            **model_args):
    # cache = Cache(model_name=model, enabled=use_cache, cache_dir=cache_dir)
    model = Model(model, device, quantize=quantize, local_files_only=local_only)
    print(f"Using device: {model.device} with {model.compute_type} compute.")

    print('Transcribing...')

    # in_cache = [(i, j) for i, a in enumerate(audio) for j, c in enumerate(a.chapters) if cache.get(a.path.name, c.id)] if not overwrite_cache else set()
    # for i, v in enumerate(in_cache):
    #     name = audio[v[0]].title+'/'+audio[v[0]].chapters[v[1]].title
    #     print(('{0: >' + str(len(str(len(in_cache))))+ '} {1}').format(i, name))
    # in_cache = set(in_cache) - {in_cache[i] for i in prompt('Choose cache files to overwrite: (eg: "1 2 3", "1-3", "^4" (empty for none))\n>> ', len(in_cache))}


    streams = []
    bars = []
    for a in audio:
        s = [i for i, s in enumerate(a.streams) if s.default][0]
        # streams.append(a.mel(cid=None, sid=s, n_mels=model.n_mels))
        streams.extend([a.mel(cid=i, sid=s, n_mels=model.n_mels) for i, _ in enumerate(a.chapters)])
        bars.extend([tqdm(total=float(c.end)-float(c.start), unit_scale=True, unit=" seconds", unit_divisor=60, desc=f"{a.title}/{c.title}", position=len(bars)+i) for i, c in enumerate(a.chapters)])

    s = time.monotonic()
    results = model.transcribe(streams, bars, batch_size, language=language, **model_args)
    grouped = []
    idk = 0
    for a in audio:
        grouped.append(results[idk:idk+len(a.chapters)])
        idk += len(a.chapters)

    f = []
    for i, a in enumerate(audio):
        f.append([])
        chapters = grouped[i]
        for i, k in enumerate(chapters):
            c = a.chapters[i]
            offset = float(c.start)
            f[-1].extend([SubLine(idx=-1, start=j.start + offset, end=j.end + offset, content=j.content) for j in k])

    for i, segments in enumerate(f):
        out = output_dir / (audio[i].path.stem + '.' + output_format)
        with out.open("w", encoding='utf8') as o:
            o.write("WEBVTT\n\n"+'\n\n'.join(s.vtt() for s in segments))

    exit(0)

    # transcribed_audio = []
    # for i, a in enumerate(audio):
    #     cf = []
    #     for j, c in enumerate(a.chapters):
    #         t = cache.get(a.path.name, c.id) if (i, j) in in_cache else cache.put(a.path.name, c.id, faster_transcribe(model, a, j, **model_args))
    #         cf.append(TranscribedAudioStream.from_map(c, t))
    #     transcribed_audio.append(TranscribedAudioFile(file=a, chapters=cf))
    print(f"Transcribing took: {time.monotonic()-s:.2f}s")

    aligner = calign.Aligner(memsize=memsize, match=1, mismatch=-1, gap_open=-1, gap_extend=-1)

    print('Fuzzy matching chapters...')
    ats, sta = match_start(aligner, transcribed_audio, text, prepend_punctuations, append_punctuations, nopend_punctuations)
    audio_batches = expand_matches(transcribed_audio, text, ats, sta)
    print_batches(audio_batches, audio, text)

    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (audio[ai].path.stem + '.' + output_format)
            if not file_overwrite and out.exists():
                bar.write(f"{out.name} already exists, skipping.")
                continue

            bar.set_description(audio[ai].path.name)
            offset, segments = sum(audio[ai].chapters[i].duration for i in range(0, batches[0][0])), []
            for astart, aend, book, tstart, tend, _ in tqdm(batches):
                ach = [(transcribed_audio[ai].chapters[i], audio[ai].chapters[i].duration) for i in range(astart, aend)]
                tch = [text[book].chapters[i] for i in range(tstart, tend)]
                segments.extend(do_batch(aligner, ach, tch, prepend_punctuations, append_punctuations, nopend_punctuations, offset))
                offset += sum(a[1] for a in ach)

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

    parser.add_argument("--audio", type=Path, required=True, default=[], action='append', help="list of audio files to process (in the correct order)")
    parser.add_argument("--language", default=None, help="language of the script and audio")

    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", default=False,  help="overwrite any destination files", action=argparse.BooleanOptionalAction)

    parser.add_argument("--output-dir", default=u'.', type=Path, help="output directory")
    parser.add_argument("--output-format", default='srt', help="output format currently only supports vtt and srt")

    subparsers = parser.add_subparsers(title='Modes')

    alass_parser = subparsers.add_parser('alass', help='Use vad+alass to realign')
    alass_parser.set_defaults(mode=alass)
    alass_parser.add_argument("--path", default='alass', help="path to alass")
    alass_parser.add_argument("--args", default=['O0'], nargs="+", help="additional arguments to alass (pass without the dash, eg: O1)")
    alass_parser.add_argument("--sort", default=True, help="sort the files (natural sort) before grouping", action=argparse.BooleanOptionalAction)

    whisper_parser = subparsers.add_parser('whisper', help='use whisper to align')
    whisper_parser.set_defaults(mode=whisper)

    whisper_parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    whisper_parser.add_argument("--device", default='auto', help="device to do inference on")
    whisper_parser.add_argument("--local-only", default=False, help="Don't download models", action=argparse.BooleanOptionalAction)
    whisper_parser.add_argument("--memsize", type=int, default=int(1*1024**3), help="amount of memory to use for alignment (in bytes)")

    whisper_parser.add_argument("--use-cache", default=True, help="use the transcription cache", action=argparse.BooleanOptionalAction)
    whisper_parser.add_argument("--overwrite-cache", default=False, help="always overwrite the cache", action=argparse.BooleanOptionalAction)
    whisper_parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="Cache directory")

    whisper_parser.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)
    whisper_parser.add_argument("--batch-size", type=int, default=4, help="number of batches to do at once")

    whisper_parser.add_argument("--beam-size", type=int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    whisper_parser.add_argument("--patience", type=float, default=1, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    whisper_parser.add_argument("--num-hypotheses", type=int, default=5, help="number of candidates when sampling with non-zero temperature")
    whisper_parser.add_argument("--length-penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    whisper_parser.add_argument("--repetition-penalty", type=float, default=1, help="penalty applied to the score of previously generated tokens")
    whisper_parser.add_argument("--no-repeat-ngram-size", type=float, default=0, help="penalty applied to the score of previously generated tokens")
    whisper_parser.add_argument("--max-initial-timestamp-index", type=lambda x: int(x)//0.02, default=1500, help="maximum index of the first predicted timestamp")

    whisper_parser.add_argument("--suppress-blank", default=True, help="suppress blank tokens at the start of sampling", action=argparse.BooleanOptionalAction)
    whisper_parser.add_argument("--suppress-tokens", type=str, default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")

    whisper_parser.add_argument("--temperatures", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1], nargs='+', help="temperature(s) to use for sampling")
    whisper_parser.add_argument("--sampling-topk", default=0, help="only use the top k tokens for sampling")
    whisper_parser.add_argument("--logprob-threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    whisper_parser.add_argument("--nospeech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `log_prob_threshold`, consider the segment as silence")

    whisper_parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    whisper_parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    whisper_parser.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    args = parser.parse_args().__dict__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop('progress'))

    language = args.pop('language')

    print("Loading...")
    audio = list(chain.from_iterable([AudioFile.from_file(f)] if f.is_file() else AudioFile.from_dir(f) for f in args.pop('audio')))
    text  = list(chain.from_iterable([TextFile.from_file(f)] if f.is_file() else TextFile.from_dir(f) for f in args.pop('text')))

    output_dir = args.pop('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')

    args.pop('mode')(audio, text, language, output_dir, output_format, args.pop('overwrite'), **args)
