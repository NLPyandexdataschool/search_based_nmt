# -*- coding: utf-8 -*-
"""
translitertion from Herbrew letters to Latin letters.

*   EKTB as suggested by [Amnon Katz in 1987][AK].
    Characters are identified by the early Hebrew letters from which they developed.
*   PHONETIC
    based on similar sounding letters, except ט->U ש->W which are based on figure.
*   other exceptions and vowels were taken from wikipedia.org and the examples
"""

HEBREW = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ".decode('UTF-8'))

trans = {'EKTB': list("ABCDEFGHVIKLMNXOPZQRSTKMNPZ".decode('UTF-8')),
         'PHONETIC': list("ABCDEFGHVIKLMNXOPZQRSTKMNPZ".decode('UTF-8'))}

transtable = {'EKTB': dict(zip(HEBREW, trans['EKTB'])),
              'PHONETIC': dict(zip(HEBREW, trans['PHONETIC']))}
exception = {
    u'צ׳': u'CH',
    u'ל״': u'L',
    u'ב֭': u'BI',
    u'ח׳': u'H',
    u'ר֭': u'V',
    u'ב֥': u'E',
    u'ן֭': u'H',
    u'ה״': u'H',
    u'ז׳': u'JE',
    u'ת״': u'TH',
    u'ג׳': u'J',
    u'ד׳': u'TH',
    u'כ׳': u'CH',
    u'ע׳': u'_',
    u'י֭': u'E',
    u'ר֣': u'H',
    u'ײ': u'A',
    u'מ֥': u'I',
    u'ל׳': u'L',
    u'ה׳': u'DH',
    u'ו״': u'VL',
    u'ר֖': u'R',
    u'ט״': u'T',
    u'ש״': 'SH',
    u'ת׳': u'KH',
    u'ז״': u'L',
    u'ב״': u'V',
    u'ץ׳': u'TS',
    u'כ״': u'AT',
    u'פ֣': u'I',
    u'י׳': u'I',
    u'ת׳': u'TH',
    u'ײ': u'A',
    u'ז֑': u'V',
    u'נ֔': u'AN',
    u'ך״': u'AV',
    u'ש׳': u'TS',
    u'ו׳': u'F',
    u'ם׳': u'M',
    u'א֥': u'A',
    u'ו׳': u'WS',
    u'י״': u'IY',
    u'ה֭': u'H',
    u'מ֖': u'M',
    u'ס״': u'S',
    u'נ״': u'L',
    u'ע״': u'K',
    u'חײ': u'KH',
    u'ה֛': u'E',
    u'ת֭': u'L',
}

vowels = {
    u'ׁ': u'SH',
    u'ׂ': u'O',
    u'ׅ': u'I',
    u'\n': u'\n',
    u'ְ': u'A',
    u'ִ': u'E',
    u'ַ': u'A',
    u'ֶ': u'I',
    u'ֹ': u'O',
    u'ֻ': u'U',
    u'ֺ': u'A',
    u'ֽ': u'_',
    u'ּ': u'_',
    u'ֿ': u'_',
}


def transliterate(string, mode='PHONETIC', return_string=True):
    """
    Canonical trans
    :param string: hebrew string
    :param mode: string PHONETIC or EKTB
    :param return_string: boolean
    :return: sting or list of letters
    """
    trans_string = []
    previous = None
    for ch in string:
        if ch in HEBREW:
            trans_string.append(transtable[mode][ch])
            previous = ch
        else:
            if ch == u'־':
                trans_string.append('-')
            elif previous is None:
                continue
            elif previous + ch in exception.keys():
                trans_string[-1] = exception[previous + ch]
            elif ch in vowels.keys():
                trans_string.append(vowels[ch])
            else:
                trans_string.append('?')
    if return_string:
        return ''.join(trans_string)
    return trans_string
