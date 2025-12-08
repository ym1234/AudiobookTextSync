import numpy as np
from ats.np import np as cnp, stft
from functools import cache
from threading import Thread
from subprocess import Popen, run, CalledProcessError, PIPE
from tqdm.auto import tqdm

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
F32LE = np.dtype(np.float32).newbyteorder('<')

@cache
def mel_filters_window(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=80):
    # Initialize the weights
    n_mels = int(n_mels)
    weights = cnp.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # Center freqs of each FFT bin
    fftfreqs = cnp.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = 0.0
    max_mel = 45.245640471924965

    mels = cnp.linspace(min_mel, max_mel, n_mels + 2)

    mels = cnp.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = cnp.log(6.4) / 27.0  # step size for log region

    # If we have vector data, vectorize
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * cnp.exp(logstep * (mels[log_t] - min_log_mel))

    mel_f = freqs

    fdiff = cnp.diff(mel_f)
    ramps = cnp.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = cnp.maximum(0, cnp.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, cnp.newaxis]

    return weights, cnp.hanning(N_FFT + 1)[:-1].astype(cnp.float32)

class MelWorker(Thread):
    def __init__(self, request_queue, n_mels=80, num_chunks=10):
        super().__init__()

        self.request_queue = request_queue
        self.num_chunks = num_chunks
        self.n_mels = n_mels
        self.daemon = True
        self.filters, self.window = mel_filters_window()

    def start_process(self, job):
        path, stream, chapter = job['path'], job['stream'], job['chapter']
        cmd = [
            "ffmpeg",
            "-nostdin", "-nostats", "-hide_banner",
            "-loglevel", "fatal",
            "-threads", "1",
            "-ss", str(chapter.start),
            "-to", str(chapter.end),
            "-i", str(path),
            "-f", "f32le",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-map", f"0:{stream}",
            '-',
        ]
        return Popen(cmd, stdout=PIPE, stderr=PIPE)

    def run(self):
        while request := self.request_queue.get():
            if not request: return
            queue = request['queue']
            process = self.start_process(request)

            self.lmax = -np.inf

            buffer = np.zeros(self.num_chunks*CHUNK_LENGTH*SAMPLE_RATE+N_FFT-HOP_LENGTH, dtype=F32LE)
            nread = process.stdout.readinto(buffer[N_FFT//2:]) + N_FFT*2
            buffer[:N_FFT//2] = buffer[N_FFT//2:N_FFT][::-1]

            stderr = b""
            while nread//4 >= len(buffer):
                queue.put((self.mel(cnp.asarray(buffer)), False))
                buffer[:N_FFT-HOP_LENGTH] = buffer[-N_FFT+HOP_LENGTH:]
                nread = process.stdout.readinto(buffer[N_FFT-HOP_LENGTH:]) + 4*(N_FFT-HOP_LENGTH)
                if k := process.stderr.read(0):
                    stderr += k

            buffer = buffer[:nread//4]
            leftover = N_FFT - len(buffer) % N_FFT
            if leftover > len(buffer):
                buffer = np.pad(buffer, (0, leftover-len(buffer)))
            buffer = np.pad(buffer, (0, leftover), mode='reflect')
            queue.put((self.mel(cnp.asarray(buffer)), True))

            tqdm.write(stderr + process.stderr.read())
            self.ret = process.wait()

    def mel(self, buffer):
        s = stft(buffer, self.window, SAMPLE_RATE, N_FFT, HOP_LENGTH)[:(N_FFT >> 1) + 1]
        magnitudes = cnp.abs(s) ** 2 # https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray
        mel_spec = self.filters @ magnitudes
        log_spec = cnp.log10(cnp.clip(mel_spec, a_min=1e-10, a_max=None))

        self.lmax = max(self.lmax, log_spec.max())
        log_spec = cnp.maximum(log_spec, self.lmax - 8.0)
        return (log_spec + 4) / 4
