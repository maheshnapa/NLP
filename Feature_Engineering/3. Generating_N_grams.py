# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:44:24 2020

@author: MAHNAPA
"""

""" +++++++++++++ Generating the N Grams+++++++++++++++++++++

Disadvantages of Count Vec

1. It does not consider the previous and the next words, to see if that would give a proper and complete meaning to the words.
2. For example: consider the word “not bad.” If this is split into individual words, then it will lose out on conveying “good” – which is what this word
actually means.
3.As we saw, we might lose potential information or insight because a lot of words make sense once they are put together. This problem can be solved by N-grams.

N-Grams 

1. N-grams are the fusion of multiple letters or multiple words. They are ormed in such a way that even the previous and next words are captured.

    1. Unigrams are the unique words present in the sentence.
    2. Bigram is the combination of 2 words.
    3.Trigram is 3 words and so on.
    
For example:
    
“I am learning NLP”
Unigrams: “I”, “am”, “ learning”, “NLP”
Bigrams: “I am”, “am learning”, “learning NLP”
Trigrams: “I am learning”, “am learning NLP”

++++++++++++++++ Gebnerating the the N grams ++++++++++++++++++++++++

"""
from textblob import TextBlob

text = " i am learning NLP"

# for unigram : Use n=1

TextBlob(text).ngrams(1)

'''[WordList(['i']), WordList(['am']), WordList(['learning']), WordList(['NLP'])] '''

# fro bigram " use n=2

TextBlob(text).ngrams(2)
'''[WordList(['i', 'am']),
 WordList(['am', 'learning']),
 WordList(['learning', 'NLP'])]'''

from sklearn.feature_extraction.text import CountVectorizer

Text = ["i love NLP and i will Learn NLP in 1month using this repo"]

vec = CountVectorizer(ngram_range=(2,2))
vec.fit(Text)
vector= vec.transform(Text)

print(vec.vocabulary_)
'''{'love nlp': 4, 'nlp and': 5, 'and will': 1, 'will learn': 9, 'learn nlp': 3, 'nlp in': 6, 'in 1month': 2, '1month using': 0, 'using this': 8, 'this repo': 7}'''

print(vector.toarray())
'''[[1 1 1 1 1 1 1 1 1 1]]'''
