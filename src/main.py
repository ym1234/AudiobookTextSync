import os
import argparse
from pprint import pprint
from types import MethodType
from lang import get_lang
from wcwidth import wcswidth

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell == 'google.colab._shell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return True  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

import mimetypes
from functools import partial, partialmethod, reduce
from itertools import groupby, takewhile, chain
from dataclasses import dataclass
from pathlib import Path

import multiprocessing
import concurrent.futures as futures

import torch
import numpy as np
import whisper

import align
from huggingface import modify_model
from quantization import ptdq_linear
from faster_whisper import WhisperModel

from rapidfuzz import fuzz

from bs4 import element
from bs4 import BeautifulSoup

from os.path import basename, splitext
import time

from audio import AudioFile, TranscribedAudioStream, TranscribedAudioFile
from text import TextFile


def sexagesimal(secs, use_comma=False):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    r = f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'
    if use_comma:
        r = r.replace('.', ',')
    return r

@dataclass(eq=True)
class Segment:
    text: str
    # words: Segment
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={sexagesimal(self.start)}, end={sexagesimal(self.end)})"
    def vtt(self, use_comma=False):
        return f"{sexagesimal(self.start, use_comma)} --> {sexagesimal(self.end, use_comma)}\n{self.text}"

def write_srt(segments, o):
    o.write('\n\n'.join(str(i+1)+'\n'+s.vtt(use_comma=True) for i, s in enumerate(segments)))

def write_vtt(segments, o):
    o.write("WEBVTT\n\n"+'\n\n'.join(s.vtt() for s in segments))

@dataclass(eq=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    ask: bool
    overwrite: bool
    memcache: dict

    def get_name(self, filename, chid):
        return filename + '.' + str(chid) +  '.' + self.model_name + ".subs"

    def get(self, filename, chid):
        fn = self.get_name(filename, chid)
        fn2 = filename + '.' + str(chid) +  '.' + 'small' + ".subs"
        fn3 = filename + '.' + str(chid) +  '.' + 'base' + ".subs"
        if fn in self.memcache: return self.memcache[fn]
        if fn2 in self.memcache: return self.memcache[fn2]
        if fn3 in self.memcache: return self.memcache[fn2]
        if not self.enabled: return
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))
        if (q := Path(self.cache_dir) / fn2).exists():
            return eval(q.read_bytes().decode("utf-8"))
        if (q := Path(self.cache_dir) / fn3).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        # if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        fn =  self.get_name(filename, chid)
        p = cd / fn
        if p.exists():
            if self.ask:
                prompt = f"Cache for file {filename}, chapter id {chid} already exists. Overwrite?  [y/n/Y/N] (yes, no, yes/no and don't ask again) "
                while (k := input(prompt).strip()) not in ['y', 'n', 'Y', 'N']: pass
                self.ask = not (k == 'N' or k == 'Y')
                self.overwrite = k == 'Y' or k == 'y'
            if not self.overwrite: return content

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

        self.memcache[fn] = content
        p.write_bytes(repr(content).encode('utf-8'))
        return content

def match_start(audio, text, cache):
    ats, sta = {}, {}
    textcache = {}
    for ai, afile in enumerate(tqdm(audio)):
        for i, ach in enumerate(tqdm(afile.chapters)):
            if (ai, i) in ats: continue

            lang = get_lang(ach.language)
            acontent = lang.normalize(lang.clean(''.join(seg['text'] for seg in ach.segments)))

            best = (-1, -1, 0)
            for ti, tfile in enumerate(text):
                for j, tch in enumerate(tfile.chapters):
                    if (ti, j) in sta: continue

                    if (ti, j) not in textcache:
                        textcache[ti, j] = lang.normalize(lang.clean(''.join(p.text() for p in tfile.chapters[j].text())))
                    tcontent = textcache[ti, j]
                    if len(acontent) < 100 or len(tcontent) < 100: continue

                    limit = min(len(tcontent), len(acontent), 2000)
                    score = fuzz.ratio(acontent[:limit], tcontent[:limit])
                    if score > 40 and score > best[-1]:
                        best = (ti, j, score)

            if best[:-1] in sta:
                tqdm.write("WARNING match_start")
            elif best != (-1, -1, 0):
                ats[ai, i] = best
                sta[best[:-1]] = (ai, i, best[-1])

    return ats, sta

# I hate it
def expand_matches(audio, text, ats, sta):
    audio_batches = []
    for ai, a in enumerate(audio):
        batch = []
        def add(idx, other=[]):
            chi, chj, _ = ats[ai, idx]
            z = [chj] + list(takewhile(lambda j: (chi, j) not in sta, range(chj+1, len(text[chi].chapters))))
            batch.append(([idx]+other, (chi, z), ats[ai, idx][-1]))

        prev = None
        for t, it in groupby(range(len(a.chapters)), key=lambda aj: (ai, aj) in ats):
            k = list(it)
            if t:
                for i in k[:-1]: add(i)
                prev = k[-1]
            elif prev is not None:
                add(prev, k)
            else:
                batch.append((k, (-1, []), None))

        if prev == len(a.chapters)-1:
            add(prev)
        audio_batches.append(batch)
    return audio_batches

# Takes in the original not the transcribed classss
def print_batches(batches, audio, text, spacing=2, sep1='=', sep2='-'):
    rows = [1, ["Audio", "Text", "Score"]]
    width = [wcswidth(h) for h in rows[-1]]

    for ai, batch in enumerate(batches):
        afn = basename(audio[ai].title)
        asuf = afn + '::'
        idk = [b[1][0] for b in batch if b[1][0] != -1]
        tsuf = all([i == idk[0] for i in idk]) and sum([len(b[1][1]) for b in batch]) > 3
        if tsuf or len(audio.chapters) > 1:
               rows.append(1)
               rows.append([afn, '', ''])
               width[0] = max(width[0], wcswidth(rows[-1][0]))
               width[1] = max(width[1], wcswidth(rows[-1][1]))
               asuf = ''
        rows.append(1)
        for ajs, (chi, chjs), score in batch:
            a = [audio[ai].chapters[aj] for aj in ajs]
            t = [text[chi].chapters[chj] for chj in chjs]
            for i in range(max(len(a), len(t))):
                row = ['', '' if t else '?', '']
                if i < len(a):
                    row[0] = asuf + a[i].title
                    width[0] = max(width[0], wcswidth(row[0]))
                if i < len(t):
                    row[1] = (t[i].title if not tsuf else '') + t[i].title.strip()
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
        if type(row) is int:
            sep = sep1 if row == 1 else sep2
            print(csep.join([sep*w for w in width]))
            continue
        print(csep.join([r.ljust(width[i]-wcswidth(r)+len(r)) for i, r in enumerate(row)]))

def to_epub():
    pass

def to_subs(text, subs, alignment, offset, references):
    alignment = [t + [i] for i, a in enumerate(alignment) for t in a]
    start, end = 0, 0
    segments = []
    for si, s in enumerate(subs):
        while end < len(alignment) and alignment[end][-2] == si:
            end += 1

        r = ''
        for a in alignment[start:end]:
            r += text[a[-1]].text()[a[0]:a[1]]

        if r.strip():
            if False: # Debug
                r = s['text']+'\n'+r
            segments.append(Segment(text=r, start=s['start']+offset, end=s['end']+offset))
        else:
            segments.append(Segment(text='＊'+s['text'], start=s['start']+offset, end=s['end']+offset))

        start = end
    return segments

def do_batch(ach, tch, prepend, append, nopend, offset):
    acontent = []
    boff = 0
    for a in ach:
        for p in a[0]['segments']:
            p['start'] += boff
            p['end'] += boff
            acontent.append(p)
        boff += a[1]

    language = get_lang(ach[0][0]['language'])

    tcontent = [p for t in tch for p in t.text()]
    alignment, references = align.align(None, language, [p['text'] for p in acontent], [p.text() for p in  tcontent], [], prepend, append, nopend)
    return to_subs(tcontent, acontent, alignment, offset, None)

def faster_transcribe(self, audio, **args):
    name = args.pop('name')

    args['log_prob_threshold'] = args.pop('logprob_threshold')
    args['beam_size'] = args['beam_size'] if args['beam_size'] else 1
    args['patience'] = args['patience'] if args['patience'] else 1
    args['length_penalty'] = args['length_penalty'] if args['length_penalty'] else 1

    gen, info = self.transcribe2(audio, best_of=1, **args)

    segments, prev_end = [], 0
    with tqdm(total=info.duration, unit_scale=True, unit=" seconds") as pbar:
        pbar.set_description(f'{name}')
        for segment in gen:
            segments.append(segment._asdict())
            pbar.update(segment.end - prev_end)
            prev_end = segment.end
        pbar.update(info.duration - prev_end)
        pbar.refresh()

    return {'segments': segments, 'language': args['language'] if 'language' in args else info.language}


def load_model(_model, device, threads, faster_whisper, local_only, quantize, fast_decoder, overlap, batches, dq):
    global model
    if faster_whisper:
        model = WhisperModel(_model, device, local_files_only=local_only, compute_type='float32' if not quantize else ('int8' if device == 'cpu' else 'float16'), num_workers=1)
        model.transcribe2 = model.transcribe
        model.transcribe = MethodType(faster_transcribe, model)
    else:
        model = whisper.load_model(_model, device)
        if quantize and device != 'cpu': # TODO fp16 to arguments
            model = model.half()
        elif dq:
            ptdq_linear(model)

        if fast_decoder:
            args["overlap"] = overlap
            args["batches"] = batches
            modify_model(model)

def transcribe3(c, temperature, **args):
    global model
    return TranscribedAudioStream.from_map(c, model.transcribe(c.audio(), name=c.title, temperature=temperature, **args))

def main():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument("--audio", nargs="+", type=Path, required=True, help="list of audio files to process (in the correct order)")
    parser.add_argument("--text", nargs="+", type=Path, required=True, help="path to the script file")

    parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to do inference on")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    parser.add_argument("--language", default=None, help="language of the script and audio")
    parser.add_argument("--local-only", default=False, help="Don't download outside models", action=argparse.BooleanOptionalAction)

    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", default=False,  help="Overwrite any destination files", action=argparse.BooleanOptionalAction)

    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    parser.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")

    parser.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dynamic-quantization", "--dq", default=False, help="Use torch's dynamic quantization (cpu only)", action=argparse.BooleanOptionalAction)

    parser.add_argument("--faster-whisper", default=True, help='Use faster_whisper, doesn\'t work with hugging face\'s decoding method currently', action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder", default=False, help="Use hugging face's decoding method, currently incomplete", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder-overlap", type=int, default=10,help="Overlap between each batch")
    parser.add_argument("--fast-decoder-batches", type=int, default=1, help="Number of batches to operate on")

    parser.add_argument("--beam_size", type=int, default=None, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", action=argparse.BooleanOptionalAction)

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--temperature_increment_on_fallback", type=float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    parser.add_argument("--word_timestamps", default=False, help="(experimental) extract word-level timestamps and refine the results based on them", action=argparse.BooleanOptionalAction)
    parser.add_argument("--highlight_words", default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_line_width", type=int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")

    parser.add_argument("--output-dir", default=None, help="Output directory, default uses the directory for the first audio file")
    parser.add_argument("--output-format", default='srt', help="Output format, currently only supports vtt and srt")

    args = parser.parse_args().__dict__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop('progress'))
    if (threads := args.pop("threads")) > 0: torch.set_num_threads(threads)

    output_dir = Path(k) if (k := args.pop('output_dir')) else Path('.')#os.path.dirname(args['audio'][0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')

    model, device = args.pop("model"), args.pop('device')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    overwrite, overwrite_cache = args.pop('overwrite'), args.pop('overwrite_cache')
    cache = Cache(model_name=model, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
                  ask=not overwrite_cache, overwrite=overwrite_cache,
                  memcache={})

    faster_whisper, local_only, quantize = args.pop('faster_whisper'), args.pop('local_only'), args.pop('quantize')
    fast_decoder, overlap, batches = args.pop('fast_decoder'), args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    dq = args.pop('dynamic_quantization')
    load_model_args = (model, device, threads, faster_whisper, local_only, quantize, fast_decoder, overlap, batches, dq)
    # if faster_whisper:
    #     model = WhisperModel(model, device, local_files_only=local_only, compute_type='float32' if not quantize else ('int8' if device == 'cpu' else 'float16'), num_workers=threads)
    #     model.transcribe2 = model.transcribe
    #     model.transcribe = MethodType(faster_transcribe, model)
    # else:
    #     model = whisper.load_model(model, device)
    #     args['fp16'] = quantize and device != 'cpu'
    #     if args['fp16']:
    #         model = model.half()
    #     elif dq:
    #         ptdq_linear(model)

    #     if fast_decoder:
    #         args["overlap"] = overlap
    #         args["batches"] = batches
    #         modify_model(model)

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")

    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    word_timestamps = args.pop("word_timestamps")

    nopend = set(args.pop('nopend_punctuations'))

    print("Loading...")

    audio = list(chain.from_iterable(AudioFile.from_dir(f) for f in args.pop('audio')))
    text = list(chain.from_iterable(TextFile.from_dir(f) for f in args.pop('text')))

    print('Transcribing...')
    s = time.monotonic()
    transcribed_audio = []

    # Trash code
    # TODO: This really doesn't have much of a perf improvement
    # Get rid of it and update faster-whisper to support batching
    with futures.ProcessPoolExecutor(max_workers=threads, initializer=load_model, initargs=load_model_args) as p:
        in_cache = []
        for i, a in enumerate(audio):
            for j, c in enumerate(a.chapters):
                if (t := cache.get(a.path.name, c.id)):
                    in_cache.append((i, j))
        if overwrite_cache:
            overwrite = set(in_cache)
        else:
            # TODO ask the user for which ones to override
            overwrite = set()

        fs = []
        for i, a in enumerate(audio):
            cf = []
            for j, c in enumerate(a.chapters):
                if (i, j) not in overwrite and (t := cache.get(a.path.name, c.id)):
                    l = partial(TranscribedAudioStream.from_map, c, t)
                else:
                    l = partial(transcribe3, c, temperature, **args)
                cf.append(p.submit(l))
            fs.append(cf)

        transcribed_audio =  [TranscribedAudioFile(file=audio[i], chapters=[r.result() for r in f]) for i, f in enumerate(fs)]
    print(f"Transcribing took: {time.monotonic()-s:.2f}s")

    print('Fuzzy matching chapters...')
    ats, sta = match_start(transcribed_audio, text, cache)
    audio_batches = expand_matches(transcribed_audio, text, ats, sta)
    print_batches(audio_batches, audio, text)

    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (splitext(basename(streams[ai][2][0].path))[0] + '.' + output_format)
            if not overwrite and out.exists():
                bar.write(f"{out.name} already exists, skipping.")
                continue

            bar.set_description(basename(streams[ai][2][0].path))
            offset, segments = 0, []
            for ajs, (chi, chjs), _ in tqdm(batches):
                ach = [(streams[ai][2][aj].transcribe(model, cache, temperature=temperature, **args), streams[ai][2][aj].duration) for aj in ajs]
                tch = [chapters[chi][1][chj] for chj in chjs]
                if tch:
                    segments.extend(do_batch(ach, tch, set(args['prepend_punctuations']), set(args['append_punctuations']), nopend, offset))

                offset += sum(a[1] for a in ach)

            if not segments:
                continue

            with out.open("w", encoding="utf8") as o:
                if output_format == 'srt':
                    write_srt(segments, o)
                elif output_format == 'vtt':
                    write_vtt(segments, o)

if __name__ == "__main__":
    main()
