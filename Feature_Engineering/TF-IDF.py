"""
1. Let’s say a particular word is appearing in all the documents of the corpus, then it will achieve higher importance in our previous methods. That’s bad for our analysis.
2. The whole idea of having TF-IDF is to reflect on how important a word is to a document in a collection, and hence normalizing words appeared frequently in all the documents.



"""

Text = ["The quick brown fox jumped over the lazy dog.","The dog.","The fox"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorizer.fit(Text)

print(vectorizer.vocabulary_)
print(vectorizer.idf_)
