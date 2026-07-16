from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
from ats.model import Transcript, ChapterTranscript, Chunk
from ats.text import SubLine
import pickle


_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename, title, stream,
    model, confidence,
    date datetime
);
CREATE TABLE IF NOT EXISTS chapter_transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER,
    language,
    idx,
    title,
    start, end,
    FOREIGN KEY(transcript_id) REFERENCES transcript(id)
);
CREATE TABLE IF NOT EXISTS chunk(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER,
    idx,
    tokens pickle,
    start, end,
    segment_start, segment_end,
    nospeech_prob, temperature, logprob,
    FOREIGN KEY(chapter_id) REFERENCES chapter_transcript(id)
);
CREATE TABLE IF NOT EXISTS segment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER,
    idx,
    content TEXT,
    start, end,
    FOREIGN KEY(chapter_id) REFERENCES chapter_transcript(id)
);
"""

_INSERT_TRANSCRIPT = """
INSERT INTO transcript (filename, title, stream, model, confidence, date)
VALUES (:filename, :title, :stream, :model, :confidence, :date) RETURNING id;
"""

_INSERT_CHAPTER_TRANSCRIPT = """
INSERT INTO chapter_transcript (idx, transcript_id, title, start, end)
VALUES (:idx, :transcript_id, :title, :start, :end) RETURNING id;
"""

_INSERT_CHUNKS = """
INSERT INTO chunk (chapter_id, idx, tokens, start, end, segment_start, segment_end, nospeech_prob, temperature, logprob)
VALUES (:chapter_id, :idx, :tokens, :start, :end, :segment_start, :segment_end, :nospeech_prob, :temperature, :logprob);
"""

_INSERT_SEGMENTS = """
INSERT INTO segment (chapter_id, idx, content, start, end)
VALUES (:chapter_id, :idx, :content, :start, :end);
"""

_LIST_TRANSCRIPTS = """
SELECT * from transcript ORDER BY date DESC LIMIT :num;
"""

_GET_TRANSCRIPT = """
select * from transcript where id=:id;
"""

_GET_CHAPTERS = """
SELECT * FROM chapter_transcript WHERE transcript_id=:transcript_id ORDER BY idx ASC;
"""

_GET_CHUNKS = """
SELECT * FROM chunk WHERE chapter_id=:chapter_id ORDER BY idx ASC;
"""

_GET_SEGMENTS = """
SELECT * FROM segment WHERE chapter_id=:chapter_id ORDER BY idx ASC;
"""

# adapter: list -> blob
sqlite3.register_adapter(list, pickle.dumps)
# converter: blob -> list
sqlite3.register_converter("pickle", pickle.loads)

sqlite3.register_adapter(datetime, lambda x: int(x.timestamp()))
sqlite3.register_converter("datetime", lambda x: datetime.fromtimestamp(int(x)))


@dataclass
class CachedTranscript:
    id: int
    filename: str
    title: str
    stream: int
    model: str
    confidence: float
    date: datetime
    chapters: list

class Cache:
    def __init__(self, database):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(database, autocommit=False, detect_types=sqlite3.PARSE_DECLTYPES)
        with self.conn:
            self.conn.executescript(_SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def list(self, limit):
        transcripts = self.conn.execute(_LIST_TRANSCRIPTS, dict(num=limit)).fetchall()

    def get(self, id):
        transcript = dict(self.conn.execute(_GET_TRANSCRIPT, dict(id=id)).fetchall()[0])
        transcript = CachedTranscript(**transcript, chapters=[])
        chapter_rows = self.conn.execute(_GET_CHAPTERS, dict(transcript_id=id)).fetchall()
        chapters = []
        for c in chapter_rows:
            chapter = dict(c)
            chapter_id = chapter.pop("id")
            chapter.pop('transcript_id')
            chapter.pop('idx')
            chapter = ChapterTranscript(**chapter, chunks=[], segments=[])

            chunk_rows =  self.conn.execute(_GET_CHUNKS, dict(chapter_id=chapter_id)).fetchall()
            segment_rows = self.conn.execute(_GET_SEGMENTS, dict(chapter_id=chapter_id)).fetchall()
            for cr in chunk_rows:
                k = dict(cr)
                k.pop('id')
                k.pop('chapter_id')
                k.pop('tokens')
                k.pop('idx')
                chapter.chunks.append(Chunk(**k, tokens=[]))

            for sr in segment_rows:
                k = dict(sr)
                k.pop('id')
                k.pop('chapter_id')
                k.pop('idx')
                chapter.segments.append(SubLine(**k))

            transcript.chapters.append(chapter)
        return transcript

    def put(self, r, transcript):
        with self.conn:
            container = r['file']
            transcript_id = self.conn.execute(_INSERT_TRANSCRIPT,
                                              dict(filename=container.path.name, title=container.title, stream=r['stream'] if 'stream' in r else container.default_stream,
                                                   confidence=transcript.confidence, model=transcript.model, date=transcript.at)).fetchone()['id']

            for i, c in enumerate(transcript.chapters):
                cid = self.conn.execute(_INSERT_CHAPTER_TRANSCRIPT,
                                        dict(transcript_id=transcript_id, idx=i, language=c.language, start=c.start, end=c.end, title=c.title)).fetchone()['id']
                self.conn.executemany(_INSERT_SEGMENTS, [dict(chapter_id=cid, idx=i, **asdict(s)) for i, s in enumerate(c.segments)])
                self.conn.executemany(_INSERT_CHUNKS,   [dict(chapter_id=cid, idx=i, **asdict(k)) for i, k in enumerate(c.chunks)])

    def close(self):
        self.conn.execute('PRAGMA optimize')
        self.conn.close()
