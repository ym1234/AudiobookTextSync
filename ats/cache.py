from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
from ats.model import Transcript, ChapterTranscript, Chunk
from ats.text import SubLine
import pickle

# AAAAAAAAAAAA i want an orm, should've probably used sqlalchemy
# start, end are all audio stream positions, sometimes in different formats, check model.py
schema = """
BEGIN;
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename, title, stream,
    model, confidence,
    date DATETIME
);
CREATE TABLE IF NOT EXISTS chapter_transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER,
    idx,
    title,
    start, end,
    FOREIGN KEY(transcript_id) REFERENCES transcript(id)
);
CREATE TABLE IF NOT EXISTS chunk(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER,
    idx,
    tokens PICKLE,
    start, end,
    segment_start, segment_end,
    nospeech_prob, temperature, logprob,
    FOREIGN KEY(chapter_id) REFERENCES chapter_transcript(id)
);
CREATE TABLE IF NOT EXISTS segment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER,
    idx,
    content,
    start, end,
    FOREIGN KEY(chapter_id) REFERENCES chapter_transcript(id)
);
COMMIT;
"""

INSERT_TRANSCRIPT = """
INSERT INTO transcript (filename, title, stream, model, confidence, date)
VALUES (:filename, :title, :stream, :model, :confidence, :date) RETURNING ID;
"""

INSERT_CHAPTER_TRANSCRIPT = """
INSERT INTO chapter_transcript (idx, transcript_id, title, start, end)
VALUES (:idx, :transcript_id, :title, :start, :end) RETURNING id;
"""

INSERT_CHUNKS = """
INSERT INTO chunks (chapter_id, idx, tokens, start, end, segment_start, segment_end, nospeech_prob, temperature, logprob)
VALUES (:chapter_id, :idx, :tokens, :start, :end, :segment_start, :segment_end, :nospeech_prob, :temperature, :logprob);
"""

INSERT_SEGMENTS = """
INSERT INTO segment (chapter_id, idx, content, start, end)
VALUES (:chapter_id, :idx, :content, :start, :end);
"""

LIST_TRANSCRIPTS = """
SELECT * from transcript ORDER BY date DESC LIMIT 20;
"""

GET_TRANSCRIPT = """
select * from transcript where id=:id;
"""

GET_CHAPTERS = """
SELECT * FROM chapter_transcript WHERE transcript_id=:id ORDER BY idx ASC;
"""

GET_CHUNKS = """
SELECT * FROM chunk WHERE chapter_id=:id ORDER BY idx ASC;
"""

GET_CHUNKS = """
SELECT * FROM segment WHERE chapter_id=:id ORDER BY idx ASC;
"""


# adapter: list -> blob
sqlite3.register_adapter(list, pickle.dumps)
# converter: blob -> list
sqlite3.register_converter("PICKLE", pickle.loads)

sqlite3.register_adapter(datetime, lambda x: int(x.timestamp()))
sqlite3.register_converter("DATETIME", lambda x: datetime.fromtimestamp(int(x)))

class Cache:
    def __init__(self, database):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(schema)

    def list(self):
        cur = self.conn.cursor()
        transcripts = cur.execute(LIST_TRANSCRIPTS)
        print(transcripts.fetchall())

    def get(self, id):
        pass

    def put(self, stream, transcript, model):
        cur = self.conn.cursor()
        cur.execute(INSERT_TRANSCRIPT, dict(filename=stream.parent.path.name, title=stream.parent.title, stream=stream.idx,
                                            confidence=transcript.confidence, model=model, date=datetime.now()))
        transcript_id = cur.fetchall()
        cur.executemany(INSERT_CHAPTER_TRANSCRIPT, [dict(transcript_id=transcript_id[0]['id'], idx=i, **asdict(c))
                                                    for i, c in enumerate(transcript.chapters)])
        chapter_ids = cur.fetchall()
        if chapter_ids != len(transcript.chapters):
            print("len(chapter_ids) != len(chapters)")
            print(chapter_ids)

        for i, c in enumerate(transcript.chapters):
            cid = chapter_ids[i]['id']
            cur.executemany(INSERT_SEGMENTS, [dict(chapter_id=cid, idx=i, **asdict(s)) for i, s in enumerate(c.segments)])
            cur.executemany(INSERT_CHUNKS,   [dict(chapter_idcid, idx=i, **asdict(k)) for i, k in enumerate(c.chunks)])
        self.conn.commit()
