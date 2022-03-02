# -*- coding: utf-8 -*-
# Copyright 2020 YSJ@TalentedSoft

import re

cmu_phoneme = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

upper_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

punctuation = ['！', '、', '‘', '’', '（', '）', '，', '。', '：', '；', '“', '”', '？', '《', '》', '-', '.', ',', '!', '?']

_valid_symbol_set = set(cmu_phoneme)


class CMUDict:
    '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''

    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='latin-1') as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        '''Returns list of ARPAbet pronunciations of the given word.'''
        result = self._entries.get(word.replace("‘", "'"))
        if result is not None:
            return result
        elif is_upper_alphabet(word):
            phoneme = ''
            for w in word:
                if w in punctuation:
                    phoneme += w + ' '
                else:
                    for i in self._entries.get(w):
                        phoneme += (i + ' ')
            return [phoneme[:-1]]
        else:
            return None


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
            parts = line.split('  ')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return ' '.join(parts)


def is_upper_alphabet(word):
    for c in word:
        if c not in upper_alphabet and c not in punctuation:
            return False
    return True
