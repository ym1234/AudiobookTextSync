import os
import ffmpeg
import numpy as np
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Union
import mimetypes
import pycountry


@cache
def get_mel_filters(sr=16000, n_fft=400, n_mels=128, dtype=np.float32):
    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

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

    def mel(self, cid, sid, sr=16000, n_fft=400, n_mels=128, dtype=np.float32):
        filters = get_mel_filters(sr=sr, n_fft=n_fft, n_mels=n_mels, dtype=dtype)
        window = np.hanning(self.n_fft + 1)[:-1]
        process = (
                ffmpeg
                .input(self.path, ss=self.chapters[cid].start, to=self.chapters[cid].end)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=sr, map=f'0:{self.streams[sid].idx}').
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

    @classmethod
    def from_file(cls, path):
        if not path.exists(): raise FileNotFoundError(f"{str(path)} doesn't exist")
        try:
            info = ffmpeg.probe(path, show_format=None, show_chapters=None, show_streams=None, select_streams='a')
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8')) from e

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
