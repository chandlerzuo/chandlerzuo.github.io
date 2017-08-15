# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 19:03:35 2017
"""

import urllib
from bs4 import BeautifulSoup as bs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

search_url = "http://variety.com/2017/tv/news/doctor-who-jenna-coleman-1202510944/"
sock = urllib.urlopen(search_url)
htmlSource = sock.read()
sock.close()
soup = bs(htmlSource, 'html.parser')

paragraphs = soup.find_all(class_="variety-content-wrapper")[0].select('p')
text = ' '.join([x.get_text() for x in paragraphs]).replace(u"\n", "").replace(u"\u2014", "").replace(u"\u2018", "").replace(u"\u2019", "").replace(u"\u201c","").replace(u"\u201d", "")

tokens = [x.lower() for x in word_tokenize(text) if x.isalpha()]

stopwords = stopwords.words('english')
allWords = nltk.FreqDist(w.lower() for w in tokens if w not in stopwords)
top_common_words = dict(allWords.most_common(10)).keys()

text = nltk.Text(tokens)
len(set(text)) * 1. / len(text) #0.57758

text.dispersion_plot(top_common_words)

