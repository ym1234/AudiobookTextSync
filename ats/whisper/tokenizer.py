import os
from functools import cached_property

import tokenizers

class Tokenizer:
    def __init__(self, path):
        self.tokenizer = tokenizers.Tokenizer.from_file(path=os.path.join(path, "tokenizer.json"))

    @cached_property
    def transcribe(self) -> int: return self.tokenizer.token_to_id("<|transcribe|>")
    @cached_property
    def sot(self) -> int: return self.tokenizer.token_to_id("<|startoftranscript|>")
    @cached_property
    def eot(self) -> int: return self.tokenizer.token_to_id("<|endoftext|>")
    @cached_property
    def no_timestamps(self) -> int: return self.tokenizer.token_to_id("<|notimestamps|>")
    @property
    def timestamp_begin(self) -> int: return self.no_timestamps + 1

    def token_to_id(self, token): return self.tokenizer.token_to_id(token)
    def id_to_token(self, id): return self.tokenizer.id_to_token(id)
    def language_code(self, language_id): return None if language_id is None else self.id_to_token(language_id)[2:-2] # TODO add guards for the token range
    def encode(self, text: str): return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens): return self.tokenizer.decode(tokens)
    def split_segments(self, tokens):
        segments, start = [], 0
        for i, t in enumerate(tokens):
            if t >= self.timestamp_begin and i > start:
                segments.append(tokens[start:i+1])
                start = i+1
        if start < len(tokens): segments.append(tokens[start:])
        return segments
