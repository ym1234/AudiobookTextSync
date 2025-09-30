import numpy as np
from functools import cache
from threading import Thread
from queue import Queue
from subprocess import Popen, run, CalledProcessError, PIPE

try:
    import cupy as cp
    from cupyx.scipy import signal
    has_cupy = True
except ImportError:
    has_cupy = False

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

@cache
def mel_filters_window_gpu(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=80):
    filters, window = mel_filters_window(sr, n_fft, n_mels)
    return cp.asarray(filters), cp.asarray(window)

class MelReader(Thread):
    def __init__(self, stream, chapter=None, n_mels=80, num_chunks=10):
        super().__init__()

        self.queue = Queue(maxsize=1)

        self.num_chunks = num_chunks
        self.n_mels = n_mels
        self.lmax = -np.inf
        self.daemon = True

        self.duration = chapter.end - chapter.start
        self.title = chapter.title
        self.offset = chapter.start

        self.cmd = [
                "ffmpeg",
                "-nostdin", "-nostats", "-hide_banner",
                "-loglevel", "fatal",
                "-threads", "1",
            ]
        if chapter: self.cmd += [ '-ss', str(chapter.start), '-to', str(chapter.end) ]
        self.cmd += [
                "-i",  str(stream.parent.path),
                "-f", "f32le",
                "-ac", "1",
                "-acodec", "pcm_f32le",
                "-ar", str(SAMPLE_RATE),
                "-map", f"0:{stream.idx}",
                "-"
            ]

    def run(self):
        process = Popen(self.cmd, stdout=PIPE, stderr=PIPE)
        buffer = np.zeros(self.num_chunks*CHUNK_LENGTH*SAMPLE_RATE+N_FFT-HOP_LENGTH, dtype=F32LE)
        nread = process.stdout.readinto(buffer[N_FFT//2:]) + N_FFT*2
        buffer[:N_FFT//2] = buffer[N_FFT//2:N_FFT][::-1]

        stderr = b""
        while nread//4 >= len(buffer):
            self.queue.put((self.mel(buffer), False))
            buffer[:N_FFT-HOP_LENGTH] = buffer[-N_FFT+HOP_LENGTH:]
            nread = process.stdout.readinto(buffer[N_FFT-HOP_LENGTH:]) + 4*(N_FFT-HOP_LENGTH)
            if k := process.stderr.read(0):
                stderr += k

        buffer = buffer[:nread//4]
        leftover = N_FFT - len(buffer) % N_FFT
        if leftover > len(buffer):
            buffer = np.pad(buffer, (0, leftover-len(buffer)))
        buffer = np.pad(buffer, (0, leftover), mode='reflect')
        self.queue.put((self.mel(buffer), True))

        self.stderr = stderr + process.stderr.read()
        self.ret = process.wait()

    def mel(self, buffer):
        raise NotImplementedError()

class GPUMelReader(MelReader):
    def mel(self, buffer):
        filters, window = mel_filters_window_gpu(n_mels=self.n_mels)
        stft = signal.stft(cp.asarray(buffer), fs=SAMPLE_RATE, window=window,
                           nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH, nfft=N_FFT, return_onesided=False)[-1]
        magnitudes = cp.abs(stft[:(N_FFT >> 1) + 1 , :-1]) ** 2

        mel_spec = filters @ magnitudes
        log_spec = cp.log10(cp.clip(mel_spec, a_min=1e-10, a_max=None))

        self.lmax = max(self.lmax, log_spec.max())
        log_spec = cp.maximum(log_spec, self.lmax - 8.0)
        return (log_spec + 4) / 4

class CPUMelReader(MelReader):
    def mel(self, buffer):
        filters, window = mel_filters_window(n_mels=self.n_mels)
        chunks = [buffer[i:i+N_FFT] for i in range(0, len(buffer), HOP_LENGTH)]# if len(buffer[i:i+N_FFT]) == N_FFT]
        chunks = np.stack(chunks[:-2])

        stft = np.fft.fft(chunks*window).T[:(N_FFT >> 1) + 1]
        magnitudes = np.abs(stft) ** 2 # https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray

        mel_spec = filters @ magnitudes
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))

        self.lmax = max(self.lmax, log_spec.max())
        log_spec = np.maximum(log_spec, self.lmax - 8.0)
        return (log_spec + 4) / 4
