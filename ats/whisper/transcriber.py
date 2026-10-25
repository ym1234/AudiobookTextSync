from queue import Queue, Empty
import multiprocessing
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from ats.text import SubLine
from ats.mel import MelWorker
from ats.np import np as cnp
from ats.whisper.state import TranscriptionState, Window
from ats.whisper.results import Chunk, ChapterTranscript, Transcript
from ats.whisper.options import TranscribeOptions, DecodeOptions

_MEL_LOOKAHEAD = 2

class Transcriber:
    def __init__(self, backend):
        self.backend = backend
        self.tokenizer = backend.tokenizer

    def _language_token(self, req, stream, use_stream_language):
        if not self.backend.is_multilingual:
            return self.tokenizer.token_to_id("<|en|>")
        code = req.get('language') or (stream.language if use_stream_language else None)
        return self.tokenizer.token_to_id(f"<|{code}|>") if code is not None else None

    def transcribe(self, requests, opts=None, decode_opts=None):
        opts, decode_opts = opts or TranscribeOptions(), decode_opts or DecodeOptions()
        if not self.backend.is_multilingual and (opts.use_stream_language or any(r.get('language') for r in requests)):
            raise ValueError("--language/--use-stream-language don't apply to English-only (.en) models")

        jobs = []
        for r in requests:
            file = r['file']
            stream = file[r['stream']] if 'stream' in r else file[file.default_stream]
            language = self._language_token(r, stream, opts.use_stream_language)
            for c in file.chapters:
                ignore = r.get('ignore', {})
                if str(c.id) in ignore or c.title in ignore:
                    continue
                jobs.append(dict(path=file.path, stream=stream.idx, language=language, chapter=c, queue=Queue(maxsize=_MEL_LOOKAHEAD)))

        results = self._transcribe(jobs, opts, decode_opts)
        idx = [0] + np.cumsum([len(r['file'].chapters) for r in requests], dtype=int).tolist()
        grouped = [results[s:e] for s, e in zip(idx, idx[1:])]

        def confidence(group):
            vals = np.array([(c.logprob, len(c.tokens)) for ch in group for c in ch.chunks])
            return np.exp(vals[:, 0].sum()/(vals[:, 1].sum()+1))

        return [Transcript(title=requests[i]['file'].title, model=self.backend.model_name, at=datetime.now(),
                           confidence=confidence(grouped[i]), chapters=grouped[i])
                for i in range(len(requests))]

    def _transcribe(self, jobs, opts, decode_opts):
        backend = self.backend
        temperatures, length_penalty = opts.temperatures, opts.length_penalty
        logprob_threshold, nospeech_threshold = opts.logprob_threshold, opts.nospeech_threshold
        main_bar = tqdm(total=len(jobs), desc="Transcribing", position=0,
                        bar_format="{l_bar}{bar}{n:3.2f}/{total:3.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                        leave=True)

        total_length = sum(j['chapter'].end - j['chapter'].start for j in jobs)
        jobs_sorted = sorted(range(len(jobs)), key=lambda x: jobs[x]['chapter'].end - jobs[x]['chapter'].start, reverse=True)

        mel_queue = Queue()
        for k in jobs_sorted: mel_queue.put(jobs[k])
        num_workers = min(len(jobs), max(opts.batch_size, multiprocessing.cpu_count()))
        mel_workers = [MelWorker(mel_queue, n_mels=backend.n_mels, num_chunks=opts.num_chunks) for _ in range(num_workers)]
        # mel_queue.shutdown() # This is pretty new
        for _ in mel_workers: mel_queue.put(None) # poison value
        for w in mel_workers: w.start()

        results = [None for _ in range(len(jobs))]
        def finalize_transcript(active):
            idx = active.idx
            chapter = jobs[idx]['chapter']
            results[idx] = ChapterTranscript(title=chapter.title, start=chapter.start, end=chapter.end,
                                             language=self.tokenizer.language_code(active.language),
                                             chunks=active.chunks, segments=active.lines)
            active.bar.close()
            main_bar.update((chapter.end - chapter.start)/total_length * len(jobs))

        def new_active(pending):
            idx = jobs_sorted[pending]
            chapter = jobs[idx]['chapter']
            bar = tqdm(total=chapter.end-chapter.start, unit_scale=True, unit=" seconds", desc=chapter.title)
            return TranscriptionState(idx=idx, buffer=cnp.zeros((backend.n_mels, 0)), lines=[], chunks=[],
                                      seek=0, bar=bar, language=jobs[idx]['language'], done=False)
        def gnmt(x): return x[1] / ((5 + len(x[0])) / 6)**length_penalty
        def norm(x): return x[1] / len(x[0])
        penalty = norm if length_penalty is None else gnmt
        ts_begin = self.tokenizer.timestamp_begin

        def decode(streams):
            # one rung per stream per round, so a window that failed only holds up itself and not the whole batch
            temps = [temperatures[a.window.attempt] for a in streams]
            prompts = [[self.tokenizer.sot, a.language, self.tokenizer.transcribe] if backend.is_multilingual
                       else [self.tokenizer.sot] for a in streams]
            rs = backend.generate([a.window.encoded for a in streams], prompts, temps, decode_opts)
            for a, temperature, r in zip(streams, temps, rs):
                w = a.window
                w.no_speech, w.attempt = r.no_speech_prob, w.attempt + 1
                for j, k in enumerate(r.sequences_ids):
                    if not k or k[-1] < ts_begin or k[-1] >= ts_begin + 1500: # only well formed sequences
                        continue
                    avg_logprob = r.scores[j] / (len(k)+1)
                    if not (avg_logprob < logprob_threshold and w.no_speech < nospeech_threshold):
                        w.accepted = True
                    if avg_logprob > logprob_threshold:
                        w.cands.append((k, r.scores[j], w.no_speech, temperature))

        def refill(a, block=False):
            got = False
            while not a.done and a.buffer.shape[-1] < 3000:
                try:
                    buf, end = jobs[a.idx]['queue'].get(timeout=0.05) if block else jobs[a.idx]['queue'].get_nowait()
                except Empty:
                    return got
                a.buffer = cnp.concatenate((a.buffer, buf), axis=-1)
                a.done, got, block = a.done or end, True, False
            return got

        active, pending = [], 0
        for i in range(min(len(jobs), opts.batch_size)):
            active.append(new_active(i))
            pending += 1

        while True:
            i = 0
            while i < len(active): # a chapter with nothing left hands its slot to the next one in the queue
                a = active[i]
                if a.done and a.buffer.shape[-1] == 0:
                    finalize_transcript(a)
                    if pending < len(jobs):
                        active[i] = new_active(pending)
                        pending += 1
                    else:
                        active.pop(i)
                        continue
                i += 1

            if not len(active):
                break

            for a in active: refill(a)
            ready = [a for a in active if a.buffer.shape[-1] and (a.done or a.buffer.shape[-1] >= 3000)]
            if not ready:
                for a in active:
                    if refill(a, block=True): break
                continue

            # a stream retrying a failed window already has its encoder output, only the new windows need encoding
            fresh = [a for a in ready if a.window is None]
            if fresh:
                padded = [cnp.pad(a.buffer[:, :3000], [(0, 0), (0, max(0, int(3000 - a.buffer.shape[-1])))])
                          for a in fresh]
                encoded = backend.encode(cnp.stack(padded))
                for i, a in enumerate(fresh): a.window = Window(encoded[i])

                assert backend.is_multilingual or all(a.language is not None for a in fresh), "english-only streams start with <|en|>"
                undetected = [a for a in fresh if a.language is None]
                if undetected:
                    languages = backend.detect_language([a.window.encoded for a in undetected])
                    for a, language in zip(undetected, languages): a.language = language

            decode(ready)

            for a in [a for a in ready if a.window.accepted or a.window.attempt == len(temperatures)]:
                w, a.window = a.window, None
                tokens, score, no_speech, temperature = max(w.cands, key=penalty) if w.cands else ([], -1, w.no_speech, -1)

                segments, seek = [], 1500
                if tokens:
                    segments = self.tokenizer.split_segments(tokens)
                    if len(segments[-1]) == 1:
                        seek = segments[-1][-1] - self.tokenizer.timestamp_begin
                        segments = segments[:-1]

                lines = [SubLine(content=self.tokenizer.decode(s[1:-1]),
                                 start=jobs[a.idx]['chapter'].start + (a.seek+s[0]-self.tokenizer.timestamp_begin)*0.02,
                                 end=jobs[a.idx]['chapter'].start + (a.seek+s[-1]-self.tokenizer.timestamp_begin)*0.02)
                         for s in segments]

                nc = Chunk(start=a.seek*2, end=a.seek*2+seek*2,
                           segment_start=len(a.lines), segment_end=len(a.lines)+len(lines),
                           temperature=temperature, nospeech_prob=no_speech,
                           logprob=score, tokens=tokens)
                a.lines.extend(lines)
                a.chunks.append(nc)
                a.seek += seek

                a.buffer = a.buffer[:, 2*seek:]
                a.bar.update(min(a.bar.total - a.bar.n, seek*0.02))

        for w in mel_workers: w.join()
        main_bar.close()
        return results
