# -*- coding: utf-8 -*-

import gensim

# please download this file in from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

model = gensim.models.Word2Vec.load_word2vec_format('C:\\Users\\GoogleNews-vectors-negative300.bin', binary=True)

#Checking how similarity works.
print (model.similarity('this', 'is'))

print (model.similarity('post', 'book'))

# Finding the odd one out.
model.doesnt_match('breakfast cereal dinner lunch';.split())

# It is also finding the relations between words.
word_vectors.most_similar(positive=['woman', 'king'],
negative=['man'])