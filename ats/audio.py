import os
import pycountry
import json
import mimetypes
import numpy as np

from tqdm.auto import tqdm
from subprocess import run, CalledProcessError, PIPE

from dataclasses import dataclass
from pathlib import Path

@dataclass(eq=True, frozen=True)
class Container:
    path: Path

    title: str
    duration: float
    default_stream: int

    streams: list
    chapters: list

    def __getitem__(self, selector):
        try:
            selector = int(stream_selector)
        except:
            pass
        if isinstance(stream_selector, int):
            return self.streams[self.default_stream if selector == -1 else selector]
        for s in self.streams:
            if s.language == selector:
                return s
        raise KeyError("{self.title} doesn't have stream {selector}")

    def __contains__(self, selector):
        try:
            self[selector]
            return True
        except:
            return False

    @classmethod
    def from_file(cls, path):
        if not isinstance(path, Path):
            path = Path(path)
        cmd = [
            "ffprobe",
            "-hide_banner",
            "-loglevel", "fatal",
            "-threads", "1",
            "-print_format", "json", # output_format/print_format
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
        duration = float(info['duration'] if 'duration' in info else info['format']['duration'])
        chapters = [Chapter(id=c['id'], title=c.get('tags', {}).get('title', ''), start=float(c['start_time']), end=float(c['end_time']))
                    for c in info['chapters']] or [Chapter(id=0, title=title, start=0, end=duration)] # Can chapters be discontinuous?
        default_streams = [i for i, s in enumerate(info['streams']) if bool(s['disposition']['default'])]
        default_stream = default_streams[0] if len(default_streams) else 0

        self = cls(path=path, title=title, duration=duration, default_stream=default_stream, chapters=chapters, streams=[])
        for s in info['streams']:
            alpha3 = s['tags'].get('language', None)
            language = pycountry.languages.get(alpha_3=alpha3).alpha_2 if alpha3 is not None else None
            stream = Stream(idx=s['index'], duration=s.get('duration', duration), language=language)
            self.streams.append(stream)
        return self

@dataclass(eq=True, frozen=True)
class Chapter:
    id: int
    title: str
    start: float
    end: float

@dataclass(eq=True, frozen=True)
class Stream:
    idx: int
    duration: float
    language: str
