from espnet2.text.bilingual import thchs_phoneme, cmu_phoneme
from espnet2.text.minnan import minnan_phoneme


def get_pinyin_list():
    thchs_tonebeep = "../../../espnet2/text/bilingual/dictionary/thchs_tonebeep"
    pinyin_list = []
    with open(thchs_tonebeep, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            pinyin_list.append(line.split(' ')[0])
    pinyin_list = sorted(pinyin_list)
    return pinyin_list


class Symbols():
    def __init__(self):
        self.thchs_phoneme = thchs_phoneme
        self.cmu_phoneme = cmu_phoneme
        self.minnan_phoneme = minnan_phoneme
        self.punc_phoneme = ["sp", "np", "lp"]
        self.prosody = ["#1", "#2", "#3", "#4"]

    def get_symbols(self, language="bilingual"):
        if language == "bilingual":
            symbols = self.punc_phoneme + self.thchs_phoneme + self.cmu_phoneme + self.prosody
        elif language == "minnan":
            symbols = self.punc_phoneme + self.minnan_phoneme
        elif language == "multilingual":
            decorate_phoneme = lambda x: '@' + x
            minnan_phoneme = [decorate_phoneme(phoneme) for phoneme in self.minnan_phoneme]
            symbols = self.punc_phoneme + self.thchs_phoneme + self.cmu_phoneme + minnan_phoneme
        elif language == "pinyin":
            symbols = self.punc_phoneme + get_pinyin_list()
        else:
            raise RuntimeError(f"only surpport [bilingual minnan multilingual pinyin] format !")
        return symbols
