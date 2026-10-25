"""model names to huggingface repos for each backend, kept apart from the backends so listing them imports neither"""
import os

CT2_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# mlx needs its own conversion of the weights. those only ship the weights, the tokenizer comes from the ct2 repo
MLX_MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "distil-medium.en": "mlx-community/distil-whisper-medium.en",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

def available_models(): return list(dict.fromkeys([*CT2_MODELS, *MLX_MODELS]))

def download(repo_id, local_dir=None, local_files_only=False, allow_patterns=None):
    """the cached snapshot when there is one, a download otherwise. a model that's already there then costs no network
    round trip, and draws none of huggingface_hub's progress bars, which it otherwise does even for 0 bytes"""
    import multiprocessing
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    if local_dir is not None: # otherwise the weights and the tokenizer repo would fight over the same directory
        local_dir = os.path.join(local_dir, repo_id.replace('/', '--'))
    args = dict(local_dir=local_dir, allow_patterns=allow_patterns, max_workers=multiprocessing.cpu_count())
    try:
        return huggingface_hub.snapshot_download(repo_id, local_files_only=True, **args)
    except LocalEntryNotFoundError:
        if local_files_only: raise
        return huggingface_hub.snapshot_download(repo_id, **args)
