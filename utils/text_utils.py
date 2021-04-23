import itertools
import re
import sys
from operator import methodcaller

import nltk
from nltk.corpus import cmudict

"""
Taken from https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
"""


def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article
