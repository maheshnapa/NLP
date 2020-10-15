# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
#Example sentences
sentences = [['I', 'love', 'nlp'],['I', 'will', 'learn', 'nlp', 'in', '2','months'],['nlp', 'is', 'future'],[ 'nlp', 'saves', 'time', 'and', 'solves','lot', 'of', 'industry', 'problems'],['nlp', 'uses', 'machine', 'learning']]

cbow = Word2Vec(sentences, size =50, window = 3, min_count=1,sg = 1)
print(cbow)

print(cbow['nlp'])

cbow.save('cbow.bin')

cbow = Word2Vec.load('cbow.bin')

# T – SNE plot
X = cbow[cbow.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)


# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(cbow.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()