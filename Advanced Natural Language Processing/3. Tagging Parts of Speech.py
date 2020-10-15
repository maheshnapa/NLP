import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

Text = "I love NLP and I will learn NLP in 2 month"

tokens = sent_tokenize(Text)

for i in tokens:
    words = nltk.word_tokenize(i)
    words = [w for w in words if not w in stop_words]
    # POS-tagger.
    tags = nltk.pos_tag(words)
    
print(tags)
