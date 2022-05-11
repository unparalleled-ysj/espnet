from espnet2.text.bilingual import thchs_phoneme, cmu_phoneme
from espnet2.text.minnan import minnan_phoneme

class Symbols():
    def __init__(self):
        self.thchs_phoneme = thchs_phoneme
        self.cmu_phoneme = cmu_phoneme
        self.minnan_phoneme = minnan_phoneme
        self.punc_phoneme = ["sp", "np", "lp"]

    def get_symbols(self, language="bilingual"):
        if language == "bilingual":
            symbols = self.punc_phoneme + self.thchs_phoneme + self.cmu_phoneme
        elif language == "minnan":
            symbols = self.punc_phoneme + self.minnan_phoneme
        elif language == "multilingual":
            decorate_phoneme = lambda x: '@' + x
            minnan_phoneme = [decorate_phoneme(phoneme) for phoneme in self.minnan_phoneme]
            symbols = self.punc_phoneme + self.thchs_phoneme + self.cmu_phoneme + minnan_phoneme
        else:
            raise RuntimeError(f"only surpport [bilingual minnan multilingual] format !")
        return symbols