"""the knobs for loading a model, scheduling a transcription and decoding a window

each lives in exactly one place, a field with its default and its help, and the cli flags are made from the fields"""
import argparse
from dataclasses import dataclass, field, fields, MISSING
from ats.whisper.models import available_models
from ats.whisper.backend import BACKENDS

def option(default, help, **cli):
    """a field and the help for its --flag, cli holds what argparse can't guess from the default (type, choices)"""
    kw = dict(default_factory=lambda: list(default)) if isinstance(default, list) else dict(default=default)
    return field(**kw, metadata=dict(help=help, cli=cli))

def add_arguments(group, cls):
    """a --flag-name for every field of cls, typed after its default"""
    for f in fields(cls):
        default = f.default_factory() if f.default is MISSING else f.default
        kw = dict(default=default, help=f.metadata['help'])
        if isinstance(default, bool): kw['action'] = argparse.BooleanOptionalAction
        elif isinstance(default, list): kw.update(nargs='+', type=type(default[0]))
        elif default is not None: kw['type'] = type(default)
        kw.update(f.metadata['cli'])
        group.add_argument('--' + f.name.replace('_', '-'), **kw)

def from_args(cls, args): return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})

@dataclass
class ModelOptions:
    model: str = option("tiny", f"whisper model to use, a huggingface repo, a local directory or one of {', '.join(available_models())}")
    backend: str = option("auto", "mlx or ctranslate2, auto picks mlx on apple silicon and ctranslate2 everywhere else", choices=BACKENDS)
    device: str = option("auto", "device to run the model on")
    device_index: int = option(0, "which gpu, ctranslate2 only")
    quantize: bool = option(True, "fp16 on a gpu, int8 on a cpu")
    download_root: str = option(None, "directory to download models to, the huggingface cache by default")
    local_files_only: bool = option(False, "never download a model, only use what's already there")

@dataclass
class TranscribeOptions:
    batch_size: int = option(4, "how many chapters are transcribed at once, each is a row of every decode batch")
    num_chunks: int = option(10, "how many 30s windows of mel a worker computes at a time")
    use_stream_language: bool = option(False, "take the language from the audio stream's metadata instead of detecting it")
    temperatures: list = option([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "the fallback ladder: a window that fails at one temperature is retried at the next")
    length_penalty: float = option(None, "length penalty alpha (https://arxiv.org/abs/1609.08144) for picking between candidates, "
                                         "plain length normalization by default", type=float)
    logprob_threshold: float = option(-1.0, "a decode whose average log probability is below this failed, and is retried hotter")
    nospeech_threshold: float = option(0.6, "a failed decode whose <|nospeech|> probability is above this is silence, not a failure")

@dataclass
class DecodeOptions:
    num_hypotheses: int = option(5, "candidates sampled per window at a temperature above 0")
    sampling_topk: int = option(0, "only sample from the k likeliest tokens, 0 for all of them")
    suppress_blank: bool = option(True, "don't let a window start with a blank")
    suppress_tokens: list = option([-1], "token ids never to generate, -1 for the non speech symbols (music notes, brackets, ...)")
    max_initial_timestamp: float = option(30.0, "the latest the first timestamp of a window may be, in seconds")
    repetition_penalty: float = option(1.0, "divides the logits of tokens already generated in the window, 1 for none")
    no_repeat_ngram_size: int = option(0, "never repeat an n-gram of this size within a window, 0 for no limit. ctranslate2 only")

    @property
    def max_initial_timestamp_index(self): return int(round(self.max_initial_timestamp / 0.02))
