
# coding: utf-8

# In[1]:

from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import nltk
from nltk.corpus import gutenberg
import pandas as pd
import urllib2

get_ipython().magic(u'matplotlib inline')

dict_word_score = {}
for line in urllib2.urlopen("http://ptrckprry.com/course/ssd/data/positive-words.txt"):
    if not (line.startswith(";") or line.startswith(" ")):
        dict_word_score[line.rstrip('\n')] = 1

for line in urllib2.urlopen("http://ptrckprry.com/course/ssd/data/negative-words.txt"):
    if not (line.startswith(";") or line.startswith(" ")):
        dict_word_score[line.rstrip('\n')] = -1
        
emma = nltk.Text(gutenberg.words('austen-emma.txt'))
persuasion = nltk.Text(gutenberg.words('austen-persuasion.txt'))

def sentiment_score(corpus):
    return np.sum([dict_word_score[x] for x in corpus if x in dict_word_score]) * 1. / len(corpus)

def partial_sentiment(corpus, positive = True):
    return np.sum([dict_word_score[x] for x in corpus if (x in dict_word_score) and (dict_word_score[x] * (2 * positive - 1) > 0)]) * 1. / len(corpus)

print(sentiment_score(emma))
print(sentiment_score(persuasion))


# In[2]:

for corpus in [emma, persuasion]:
    for pos in [True, False]:
        counts = Counter([x for x in corpus if (x in dict_word_score) and (dict_word_score[x] * (2 * pos - 1) > 0)]).most_common(20)
        dat = pd.DataFrame.from_dict(counts)
        dat.columns = ['word', 'frequency']
        dat.set_index('word', inplace = True)
        dat.plot(kind = 'bar', legend = False, title = '{} in {}'.format("Positives" if pos else "Negatives", "Emma" if corpus == emma else "Persuasion"))


# In[4]:

def plot_sentiment_flow(title):
    sents = gutenberg.sents(title)
    positive_flow = [partial_sentiment(x) for x in sents]
    negative_flow = [partial_sentiment(x, positive = False) for x in sents]
    plt.plot(range(len(sents)), positive_flow, label = 'Positive')
    plt.plot(range(len(sents)), negative_flow, label = 'Negative')
    plt.ylabel('Sentiment Score')
    plt.xlabel(title)
    plt.show()
    
plot_sentiment_flow('austen-emma.txt')
plot_sentiment_flow('austen-persuasion.txt')

