import os
import json
import mimetypes
import numpy as np

from subprocess import Popen, PIPE, DEVNULL, run, CalledProcessError
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tqdm.auto import tqdm
import time

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
F32LE = np.dtype(np.float32).newbyteorder('<')

@cache
def mel_filters_window(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=80):
    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # Center freqs of each FFT bin
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = 0.0
    max_mel = 45.245640471924965

    mels = np.linspace(min_mel, max_mel, n_mels + 2)

    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    # If we have vector data, vectorize
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    mel_f = freqs

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights, np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def read_full(pipe, buffer, offset):
    nread, end = offset, False
    while nread < len(buffer):
        bread = pipe.readinto(buffer[nread:])
        if bread == 0: # I think this is correct?
            end = True
            break
        bread //= 4
        nread += bread
    return nread, end

class MelProcess:
    def __init__(self, stream, chapter, n_mels=80, num_chunks=60): # 120 on the gpu
        self.cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            '-ss', str(chapter.start),
            '-to', str(chapter.end),
            "-i",  str(stream.path),
            "-f", "f32le",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-map", f"0:{stream.idx}",
            "-"
        ]

        self.title = stream.title + ("/" + chapter.title if stream.title != chapter.title else '')
        self.offset = chapter.start
        self.duration = chapter.end - chapter.start
        self.num_chunks = num_chunks
        self.filters, self.window = mel_filters_window(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)

    def generator(self):
        buffer = np.zeros(self.num_chunks*CHUNK_LENGTH*SAMPLE_RATE + N_FFT - HOP_LENGTH, dtype=F32LE)
        process = Popen(self.cmd, bufsize=2*buffer.nbytes, stdout=PIPE, stderr=DEVNULL)

        nread, end = read_full(process.stdout, buffer, N_FFT//2)
        buffer[:N_FFT//2] = buffer[N_FFT//2:N_FFT][::-1] # reflect
        lmax = -np.inf
        while not end:
            mel, lmax = self.cpu_mel(buffer, lmax)
            yield mel
            buffer[:N_FFT - HOP_LENGTH] = buffer[-N_FFT+HOP_LENGTH:]
            nread, end = read_full(process.stdout, buffer, N_FFT - HOP_LENGTH)

        leftover = N_FFT - nread % N_FFT
        buffer[nread:nread+leftover] = buffer[nread-leftover:nread][::-1]
        yield self.cpu_mel(buffer[:nread+leftover], lmax)[0][:, :-1]

    def cpu_mel(self, buffer, lmax): # GPU mel?
        chunks = np.stack([buffer[i:i+N_FFT] for i in range(0, len(buffer), HOP_LENGTH)][:-2])

        stft = np.fft.fft(chunks*self.window).T[:(N_FFT >> 1) + 1]
        magnitudes = np.abs(stft) ** 2 # https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray

        mel_spec = self.filters @ magnitudes
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))

        lmax = max(lmax, log_spec.max())
        log_spec = np.maximum(log_spec, lmax - 8.0)
        return (log_spec + 4) / 4, lmax

@dataclass(eq=True, frozen=True)
class Chapter:
    id: int
    title: str
    start: float
    end: float

@dataclass(eq=True, frozen=True)
class Stream:
    idx: int
    duration: float
    language: str
    default: bool
    path: Path
    title: str

@dataclass(eq=True, frozen=True)
class AudioFile:
    path: Path

    title: str
    duration: float

    streams: list
    chapters: list

    @classmethod
    def from_file(cls, path):
        cmd = [
            "ffprobe",
            "-threads", "1",
            "-print_format", "json", # output_format/print_format
            "-show_format",
            "-show_chapters",
            "-show_streams",
            "-select_streams", "a",
            str(path),
        ]
        try:
            out = run(cmd, capture_output=True, check=True).stdout.decode('utf-8')
            info = json.loads(out)
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio:\n {e.stderr.decode('utf-8')}") from e

        title = info.get('format', {}).get('tags', {}).get('title', path.name)
        duration = float(info['duration'] if 'duration' in info else info['format']['duration'])
        chapters = [Chapter(id=c['id'], title=c.get('tags', {}).get('title', ''), start=float(c['start_time']), end=float(c['end_time']))
                    for c in info['chapters']] or [Chapter(id=0, title=title, start=0, end=duration)]
        streams  = [Stream(idx=s['index'], duration=s.get('duration', duration),
                           language=s['tags'].get('language', ''), default=bool(s['disposition']['default']), path=path, title=title)
                    for s in info['streams']]
        return cls(path=path, title=title, duration=duration, chapters=chapters, streams=streams)

    @classmethod
    def from_dir(cls, path):
        if not path.exists(): raise FileNotFoundError(f"{str(path)} doesn't exist")
        mt = {'video', 'audio'}
        for p, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            p = Path(p)
            for f in files:
                t, _ = mimetypes.guess_type(f)
                if p.suffix != ".ass" and t is not None and t.split('/', 1)[0] in mt:
                    yield cls.from_file(p/f)
