import os
import ffmpeg
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import mimetypes

@dataclass(eq=True, frozen=True)
class AudioChapter:
    stream: ffmpeg.Stream
    duration: float
    title: str
    id: int
    language: str

    def audio(self):
        try:
            data, _ = self.stream.run(quiet=True, input='')
            return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8')) from e

    def async_audio(self):
        p = self.stream.run_async(pipe_stdout=True, pipe_stderr=True)
        return p.communicate()

@dataclass(eq=True, frozen=True)
class AudioStream:
    path: Path
    title: str
    duration: float
    chapters: list

    def audio(self):
        return np.concatenate([c.audio() for c in self.chapters])

    @classmethod
    def from_file(cls, path, whole=False):
        if not path.is_file(): raise FileNotFoundError(f"file {str(path)} is not a file")

        try:
            info = ffmpeg.probe(path, show_chapters=None)
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8')) from e

        ftitle = info.get('format', {}).get('tags', {}).get('title', path.name)
        fduration = float(info.get('duration', info['format'].get('duration', 0)))
        if fduration == 0:
            print(f"WARNING: Couldn't determine container duration {path.name}")

        # This is kind of ugly
        streams = []
        for astream in info['streams']:
            if astream['codec_type'] != 'audio': continue

            language = astream.get('tags', {}).get('language', 'und').split('-', 1)[0]
            duration = float(astream.get('duration', fduration))
            if 'duration' not in astream:
                print(f"WARNING: Couldn't determine duration of {astream['index']}, this is normal for MKV and WEBM files")

            output_args = {'format': 's16le', 'acodec': 'pcm_s16le', 'ac': 1, 'ar': '16k', 'map': f'0:{astream[index]}'}
            if whole or len(info['chapters']) < 1:
                chapters = [AudioChapter(stream=ffmpeg.input(path).output('-', **output_args), duration=duration, title=ftitle, id=-1, language=language)]
            else:
                chapters = [AudioChapter(stream=ffmpeg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])).output('-', **output_args),
                                        duration=float(chapter['end_time']) - float(chapter['start_time']),
                                        title=chapter.get('tags', {}).get('title', str(i)),
                                        id=chapter['id'],
                                        language=language)
                            for i, chapter in enumerate(info['chapters'])]
            streams.append(cls(title=ftitle, path=path, duration=duration, chapters=chapters))
        return streams

    @classmethod
    def scandir(cls, path):
        if not path.exists(): raise FileNotFoundError(f"file {str(path)} does not exist")
        if path.is_file(): return path

        mt = {'video', 'audio'}
        files = []
        for root, _, files in os.walk(str(path)):
            root = Path(root)
            for f in file:
                p = root/f
                t, _ = mimetypes.guess_type(p)
                if p.suffix != '.ass' and t is not None and t.split('/', 1)[0] in mt:
                    files.append(p)
        return files

@dataclass(eq=True, frozen=True)
class TranscribedAudioChapter:
    stream: AudioChapter
    inferred_language: str # Whether this is actually from whisper or not is not determined
    segments: list

    @classmethod
    def from_map(cls, stream, transcript): return cls(stream=stream, inferred_language=transcript['language'], segments=transcript['segments'])

@dataclass(eq=True, frozen=True)
class TranscribedAudioStream:
    file: AudioStream
    chapters: list[TranscribedAudioChapter]
