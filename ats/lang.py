import regex as re
import unicodedata
from functools import cache
import chinese_converter

class Language:
    def __init__(self, prepend, append, nopend): # Should each language have its own *pend defaults?
        self.translations = {}
        self.prepend, self.append, self.nopend = prepend, append, nopend

    def normalize(self, s): return unicodedata.normalize("NFKD", s)
    def translate(self, s): return s.translate(self.translations).lower()
    def clean(self, s): return self.translate(s)
    def fix(self, s, i): return i

class Japanese(Language):
    def __init__(self, prepend, append, nopend):
        self.ascii_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
        self.kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))

        kansuu = '一二三四五六七八九十〇零壱弐参肆伍陸漆捌玖拾'
        arabic = '１２３４５６７８９１００'
        self.kansuu_arabic = {ord(kansuu[i]): arabic[i%len(arabic)] for i in range(len(kansuu))}

        self.punc =  {ord(i): '。' for i in prepend}
        self.confused = {ord('は'): 'わ', ord('あ'): 'わ', ord('お'): 'を', ord('へ'): 'え'}

        self.translations = self.kata_hira | self.kansuu_arabic | self.ascii_wide | self.punc | self.confused

        self.r1 = re.compile(r'(?![。])[\p{C}\p{M}\p{P}\p{S}\p{Z}\sー々ゝ'+nopend+r']+')
        self.r2 = re.compile(r'(.)(?=\1+)')
        # self.r3 = re.compile(r'（.*?）')

    def clean(self, s):
        s = self.translate(s)
        s = self.r1.sub('', s)
        s = self.r2.sub('', s)
        return s
        # return self.r3.sub('', self.r2.sub('', s))

class English(Language):
    def __init__(self, prepend, append, nopend):
        self.translations = {ord(i): '.' for i in prepend}

class Chinese(Language):
    def translate(self, s): return chinese_converter.to_simplified(s)

_languages = {
        'ja': Japanese,
        'en': English,
        'zh': Chinese,
}

@cache
def get_lang(lang, prepend=None, append=None, nopend=None):
    return _languages.get(lang, _languages['en'])(prepend, append, nopend)
