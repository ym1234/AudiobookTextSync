from dataclasses import dataclass, field
from tqdm.auto import tqdm

@dataclass
class TranscriptionState:
    idx: int
    buffer: any
    lines: list
    chunks: list
    seek: int
    bar: tqdm
    language: int
    done: bool
    window: any = None

@dataclass
class Window:
    encoded: any
    attempt: int = 0
    accepted: bool = False
    no_speech: float = 0.
    cands: list = field(default_factory=list)
