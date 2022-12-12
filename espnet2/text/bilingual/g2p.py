import os
from typing import List, Iterable
from espnet2.text.bilingual import THCHSDict, CMUDict, thchs_phoneme, cmu_phoneme
from espnet2.text.bilingual import text2pinyin


class BilingualG2P():
    def __init__(self):
        thchsdict_path = os.path.abspath(r'../../../espnet2/text/bilingual/dictionary/thchs_tonebeep')
        self.thchsdict = THCHSDict(thchsdict_path)
        cmudict_path = os.path.abspath(r'../../../espnet2/text/bilingual/dictionary/cmudict-0.7b')
        self.cmudict = CMUDict(cmudict_path)
        self.text2pinyin = text2pinyin
        self.symbols = ["sp", "np", "lp"] + thchs_phoneme + cmu_phoneme + ["#1", "#2", "#3", "#4"]
        self.pause_punctuation = {
            "？": "np",
            "！": "np",
            "，": "np",
            "。": "np",
            "：": "np",
            "、": "np",
            "；": "np",
            ",": "np",
            ".": "np",
            "?": "np",
            "!": "np",
            "~": "sp",
        }

    def text2tokens(self, line: str, add_blank: bool=False) -> List[str]:
        pinyin = self.text2pinyin(line)
        phoneme = [self.get_phoneme(self.thchsdict, cn) for cn in pinyin.split(' ')]
        phoneme = ' '.join([self.get_arpabet(self.cmudict, en) for en in phoneme])
        phoneme = self.punctuation2silence(phoneme)
        if add_blank:
            phoneme = self.add_blank(' '.join(phoneme))
        # print(' '.join(phoneme))
        return phoneme

    def add_blank(self, line: str) -> List[str]:
        new_line = []
        for phoneme in line.split(' '):
            new_line.append(phoneme)
            # if phoneme[0]!='#' and phoneme[-1].isdigit():
            if phoneme not in ["sp", "np", "lp", "#1", "#2", "#3", "#4"]:
                new_line.append('<blank>')
        return new_line

    def should_keep_symbol(self, s):
        return s in self.symbols
    
    def get_arpabet(self, cmudict, word):
        arpabet = cmudict.lookup(word)
        return '%s' % arpabet[0] if arpabet is not None else word

    def get_phoneme(self, thchsdict, pinyin):
        phoneme = thchsdict.lookup(pinyin)
        return '%s' % phoneme if phoneme is not None else pinyin

    def punctuation2silence(self, phoneme):
        result = []
        for p in phoneme.split(' '):
            sil = self.pause_punctuation.get(p) 
            p = sil if sil is not None else p
            if self.should_keep_symbol(p):
                result.append(p)
        return result

    def get_symbols(self):
        return self.symbols

        
        
if __name__ == "__main__":
    g2p = BilingualG2P()
    print(g2p.text2tokens("厦门是个美丽的海滨城市，do you agree?"))