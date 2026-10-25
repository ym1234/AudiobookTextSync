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
    @cached_property
    def translate(self) -> int: return self.tokenizer.token_to_id("<|translate|>")
    @cached_property
    def no_speech(self) -> int:
        t = self.tokenizer.token_to_id("<|nospeech|>")
        return self.tokenizer.token_to_id("<|nocaptions|>") if t is None else t
    @property
    def timestamp_begin(self) -> int: return self.no_timestamps + 1
    @cached_property
    def language_tokens(self) -> list: return list(range(self.sot + 1, self.translate))

    @cached_property
    def non_speech_tokens(self) -> list:
        """Tokens that mark events/music/etc. rather than speech, ported from openai's whisper"""
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or a part of one, only the former can be suppressed
        miscellaneous = set("♩♪♫♬♭♮♯")

        result = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.encode(symbol), self.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
        return sorted(result)

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
