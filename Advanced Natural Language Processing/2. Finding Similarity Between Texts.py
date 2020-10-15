# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = (
"I like NLP",
"I am exploring NLP",
"I am a beginner in NLP",
"I want to learn NLP",
"I like advanced NLP"
)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_matrix.shape

cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)

"""cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
Out[37]: array([[1.        , 0.17682765, 0.14284054, 0.13489366, 0.68374784]]) """
