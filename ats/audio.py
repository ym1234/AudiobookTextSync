import os
import json
import time
import mimetypes
import numpy as np

from pprint import pprint
from subprocess import Popen, PIPE, run, CalledProcessError
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Union

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

@cache
def get_mel_filters(sr=16000, n_fft=400, n_mels=128):
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

    return weights


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

@dataclass(eq=True, frozen=True)
class Chapter:
    cid: int
    title: str
    start: Union[str, float]
    end: Union[str, float]

@dataclass(eq=True, frozen=True)
class Stream:
    idx: int
    duration: Union[str, float]
    language: str
    default: bool

@dataclass(eq=True, frozen=True)
class AudioFile:
    path: Path

    title: str
    duration: float

    streams: list
    chapters: list
    def mel(self, cid, sid, n_mels=80):
        filters = get_mel_filters(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
        window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
        num_fft_bins = (N_FFT >> 1) + 1

        mmel = -np.inf # This stupid thing, technically not the same but i doubt it matters
        def to_mel(arr):
            nonlocal mmel
            chunks = np.stack([arr[i:i+N_FFT] for i in range(0, len(arr), HOP_LENGTH)][:-2])

            stft = np.fft.fft(chunks*window).T[:num_fft_bins]
            magnitudes = np.abs(stft) ** 2 # https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray

            mel_spec = filters @ magnitudes
            log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))

            nmax = log_spec.max()
            if nmax > mmel: mmel = nmax

            log_spec = np.maximum(log_spec, mmel - 8.0)
            return (log_spec + 4) / 4

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            '-ss', str(self.chapters[cid].start),
            '-to', str(self.chapters[cid].end),
            "-i",  str(self.path),
            "-f", "f32le",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-map", f"0:{self.streams[sid].idx}",
            "-"
        ]

        dt = np.dtype(np.float32).newbyteorder('<')
        cur = np.zeros(CHUNK_LENGTH*SAMPLE_RATE + N_FFT - HOP_LENGTH, dtype=dt)
        process = Popen(cmd, bufsize=10*cur.nbytes, stdout=PIPE, stderr=PIPE)

        nread, end = read_full(process.stdout, cur, N_FFT//2)
        cur[:N_FFT//2] = cur[N_FFT//2:N_FFT][::-1] # reflect

        while not end:
            yield to_mel(cur)
            cur[:N_FFT - HOP_LENGTH] = cur[-N_FFT+HOP_LENGTH:]
            nread, end = read_full(process.stdout, cur, N_FFT - HOP_LENGTH)

        leftover = N_FFT - nread % N_FFT
        cur[nread:nread+leftover] = cur[nread-leftover:nread][::-1]
        yield to_mel(cur[:nread+leftover])[:, :-1]


    def get_wave(self, chapter, sidx):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            '-ss', str(self.chapters[chapter].start),
            '-to', str(self.chapters[chapter].end),
            "-i",  str(self.path),
            "-f", "f32le",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-map", f"0:{self.streams[sidx].idx}",
            "-"
        ]

        dt = np.dtype(np.float32).newbyteorder('<')
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio:\n {e.stderr.decode('utf-8')}") from e
        return np.frombuffer(out, dtype=dt)

    @classmethod
    def from_file(cls, path):
        cmd = [
            "ffprobe",
            "-threads", "0",
            "-output_format", "json",
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
        duration = info['duration'] if 'duration' in info else info['format']['duration']
        chapters = [Chapter(cid=c['id'], title=c.get('tags', {}).get('title', ''), start=c['start_time'], end=c['end_time'])
                    for c in info['chapters']]
        streams  = [Stream(idx=s['index'], duration=s['duration'], language=s['tags'].get('language', ''), default=bool(s['disposition']['default']))
                    for s in info['streams']]
        return cls(path=path, title=title, duration=duration, chapters=chapters, streams=streams)

    @classmethod
    def from_dir(cls, path):
        if not path.exists(): raise FileNotFoundError(f"{str(path)} doesn't exist")
        mt = {'video', 'audio'}
        for _, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            for f in files:
                t, _ = mimetypes.guess_type(f)
                if p.suffix != ".ass" and t is not None and t.split('/', 1)[0] in mt:
                    yield cls.from_file(p)

# @dataclass(eq=True, frozen=True)
# class TranscribedAudioStream:
#     stream: AudioStream
#     language: str
#     segments: list

#     @classmethod
#     def from_map(cls, stream, transcript): return cls(stream=stream, language=transcript['language'], segments=transcript['segments'])

# @dataclass(eq=True, frozen=True)
# class TranscribedAudioFile:
#     file: AudioFile
#     chapters: list[TranscribedAudioStream]
