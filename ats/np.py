import numpy as np
try:
    import cupy
    if cupy.cuda.is_available():
        no_cuda = False
        np = cupy
    no_cuda = True
except:
    no_cuda = True


def stft(buf, window, sample_rate, nfft, hoplength):
    if not no_cuda:
        stft = np.scipy.signal.stft(buf, fs=sample_rate, window=window, nperseg=nfft, nooverlap=nfft-hoplength, nfft=nfft, return_onesided=False)[-1]
        return stft[:, :-1]

    chunks = [buf[i:i+nfft] for i in range(0, len(buf), hoplength)]# if len(buf[i:i+N_FFT]) == N_FFT]
    chunks = np.stack(chunks[:-2])
    return np.fft.fft(chunks*window).T
