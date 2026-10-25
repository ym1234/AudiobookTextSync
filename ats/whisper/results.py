from dataclasses import dataclass
from datetime import datetime

@dataclass
class Chunk:
    start: int # in mel frames
    end: int
    segment_start: int
    segment_end: int
    temperature: float
    logprob: float
    nospeech_prob: float
    tokens: list # idk if i care about this

@dataclass
class ChapterTranscript:
    title: str
    start: float
    end: float
    language: str
    segments: list
    chunks: [Chunk]

@dataclass
class Transcript:
    model: str
    at: datetime
    confidence: float
    chapters: [ChapterTranscript]
