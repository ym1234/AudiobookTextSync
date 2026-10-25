import numpy as np
try:
    import cupy
    if cupy.cuda.is_available():
        no_cuda = False
        np = cupy
    else:
        no_cuda = True
except:
    no_cuda = True

if no_cuda:
    # every mel worker is its own thread, so one blas thread each. otherwise openblas fans every small matmul out over
    # all the cores it can see (the host's, in a container) and ~20 workers doing that measured slower than single
    # threaded by more than 2x on a 32 core box. a no-op for accelerate on macos
    from threadpoolctl import threadpool_limits
    _blas_limit = threadpool_limits(1, user_api="blas")

def stft(buf, window, nfft, hoplength):
    """the nfft//2+1 non-redundant bins of every full frame, as (bins, frames)

    the same code for numpy and cupy. the caller already does whisper's center padding, scipy's stft would pad again
    and scale by 1/sum(window) on top, which shifts every frame and makes the whole spectrogram ~4.6 log10 quieter"""
    frames = np.lib.stride_tricks.sliding_window_view(buf, nfft)[::hoplength]
    return np.fft.rfft(frames * window).T
