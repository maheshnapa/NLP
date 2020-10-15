# -*- coding: utf-8 -*-

import nltk
from nltk import ne_chunk
from nltk import word_tokenize

sent = "John is studying at Stanford University in California"

ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary=False)


import spacy
nlp = spacy.load('en')

# Read/create a sentence
doc = nlp(u'Apple is ready to launch new phone worth $10000 in New york time square ')

for ent in doc.ents:
   print(ent.text, ent.start_char, ent.end_char, ent.label_)