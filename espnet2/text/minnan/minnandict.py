# -*- coding: utf-8 -*-
# Copyright 2020 YSJ@TalentedSoft

minnan_phoneme = [
    'a', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
    'ai', 'ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 
    'aiN1', 'aiN2', 'aiN3', 'aiN5', 'aiN6', 
    'aN0', 'aN1', 'aN2', 'aN3', 'aN4', 'aN5', 'aN6', 
    'aNh7', 
    'ao', 'ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 
    'aoN2', 'aoN3', 'aoN5', 'aoN6', 
    'b', 
    'bb', 
    'c', 
    'd', 
    'e', 'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 
    'ee', 'ee0', 'ee1', 'ee2', 'ee3', 'ee5', 'ee6', 
    'eeN1', 'eeN2', 'eeN3', 'eeN5', 'eeN6', 
    'eeNh7', 
    'eN1', 'eN2', 'eN3', 'eN5', 
    'eNh8', 
    'eo', 'eo1', 'eo2', 'eo3', 'eo4', 'eo6', 'eo8', 
    'eoN1', 
    'g', 
    'gg', 
    'h', 
    'h7', 'h8', 
    'i', 'i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 
    'ia', 'ia0', 'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 'ia6', 
    'iaN', 'iaN0', 'iaN1', 'iaN2', 'iaN3', 'iaN4', 'iaN5', 'iaN6', 'iaNh7', 
    'iao', 'iao0', 'iao1', 'iao2', 'iao3', 'iao4', 'iao5', 'iao6', 
    'iN0', 'iN1', 'iN2', 'iN3', 'iN4', 'iN5', 'iN6', 
    'iNh7', 'iNh8', 
    'io', 'io0', 'io1', 'io2', 'io3', 'io4', 'io5', 'io6', 
    'ioh8', 
    'ioo', 'ioo1', 'ioo2', 'ioo3', 'ioo5', 'ioo6', 
    'iooN0', 'iooN1', 'iooN2', 'iooN3', 'iooN5', 'iooN6', 
    'iu', 'iu1', 'iu2', 'iu3', 'iu4', 'iu5', 'iu6', 
    'iuN1', 'iuN2', 'iuN3', 'iuN4', 'iuN5', 'iuN6', 
    'k', 
    'k7', 'k8', 
    'l', 
    'm', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 
    'n', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 
    'ng', 'ng0', 'ng1', 'ng2', 'ng3', 'ng4', 'ng5', 'ng6', 
    'o', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 
    'oo', 'oo0', 'oo1', 'oo2', 'oo3', 'oo4', 'oo5', 'oo6', 
    'ooh', 
    'ooN1', 'ooN2', 'ooN3', 'ooN5', 'ooN6', 
    'ooNh7', 
    'p', 
    'p7', 'p8', 
    's', 
    't', 
    't7', 't8', 
    'u', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 
    'ua', 'ua0', 'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'ua6', 
    'uai1', 'uai2', 'uai3', 'uai5', 'uai6', 
    'uaiN1', 'uaiN2', 'uaiN3', 'uaiN5', 'uaiN6', 
    'uaN1', 'uaN2', 'uaN3', 'uaN4', 'uaN5', 'uaN6', 
    'ue', 'ue1', 'ue2', 'ue3', 'ue4', 'ue5', 'ue6', 
    'ui', 'ui0', 'ui1', 'ui2', 'ui3', 'ui4', 'ui5', 'ui6', 
    'uiN0', 'uiN1', 'uiN2', 'uiN3', 'uiN5', 'uiN6', 
    'uu1', 'uu2', 'uu3', 'uu4', 'uu6', 
    'z', 
    'zz',
    ]


class MinNanDcit:
    '''Thin wrapper around MinNanDcit data to convert chinese word to minnan phoneme'''
    def __init__(self, file):
        with open(file, encoding='utf-8')as f:
            entries = _parse_minnandict(f)
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        '''Returns list of  minnan phoneme of the given chinese word'''
        return self._entries.get(word)


def _parse_minnandict(file):
    minnandict = {}
    for line in file:
        content = line.strip().split()
        word = content[0]
        phoneme = ' '.join(content[1:])
        if word not in minnandict:
            minnandict[word] = phoneme
    return minnandict
