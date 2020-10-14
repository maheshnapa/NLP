""" 

+++++++++++ Count Vectorizing  ++++++++++++++

Disadvantages of OneHOtEncoding

1. it has a disadvantage. It does not take the frequency of the word occurring into consideration. 
2. If a particular wordis appearing multiple times, there is a chance of missing the information if it is not included in the analysis

Count Vectorzing

1. Count vectorizer is almost similar to One Hot encoding. 
2. The only difference is instead of checking whether the particular word is present or not, it will count the words that are present in the document.

"""

from sklearn.feature_extraction.text import CountVectorizer

text = ["I love NLP and i Will learn NLP in 2 months and apply in real world"]

vec = CountVectorizer()

vec.fit(text)

vector = vec.transform(text)

print(vec.vocabulary_)
print(vector.toarray())