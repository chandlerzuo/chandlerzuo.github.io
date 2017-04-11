
# coding: utf-8

# In[1]:

import codecs
import json
import logging
import nltk
import numpy as np
import os
import pandas as pd
import pipes
from multiprocessing import Pool
from matplotlib import pyplot as plt
import subprocess
import sys
from textstat.textstat import textstat
##from tsne import bh_sne

from sklearn import manifold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

from gensim import corpora, models

import nltk
from nltk.tree import ParentedTree
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from stanford_corenlp_pywrapper import CoreNLP

def setup_s3():
    global conn, s3bucket
    conn = S3Connection('awsuser', 'awskey')
    s3bucket = conn.get_bucket('bucket_name')
    
def setup_log():
    global logger
    # create logger
    logger = logging.getLogger('PARSE')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

def setup_path():
    global local_data_dir, s3_input_prefix, s3_output_prefix, s3_bucket
    local_data_dir = '...'
    s3_input_prefix = '...'
    s3_output_prefix = '...'
    s3_bucket = 'bucket_name'
    
def exists_in_s3(key):
    ls = list_in_s3(key)
    if len(ls) > 0:
        if ls[0] == key[(key.rfind('/') + 1):]:
            return True
    return False

def list_in_s3(prefix):
    command = "aws s3 ls s3://{}/{}".format(s3_bucket, prefix)
    ls = subprocess.Popen(command.split(' '), stdout = subprocess.PIPE).stdout.read().split('\n')
    if len(ls) > 0:
        ls = [x[(x.rfind(' ') + 1):] for x in ls]
        return ls
    else:
        return []
    
def text_tokenize(sentence):
    #stemmer = SnowballStemmer('english')
    lmtr = WordNetLemmatizer()
    tokens = [x.lower() for x in word_tokenize(sentence) if x.isalpha()]
    tokens_tagged = nltk.pos_tag(tokens)
    tokens_tagged = [(x, get_wordnet_pos(y)) for (x, y) in tokens_tagged if x not in stopwords.words('english')]
    return [lmtr.lemmatize(x, y) if y != '' else x for (x, y) in tokens_tagged]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
def build_lda_model(documents, num_topics = 10):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    lda_model = models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes=20)
    
    topic_dist = [lda_model.get_document_topics(dictionary.doc2bow(x)) for x in documents]
    lda_topic_embed = np.zeros((len(topic_dist), 8))
    for i in range(len(topic_dist)):
        for j, x in topic_dist[i]:
            lda_topic_embed[i, j] = x
    return dictionary, lda_model, lda_topic_embed

def build_nmf_model(documents, num_topics = 10):
    vectorizer = TfidfVectorizer(max_df = 1., min_df = 0.2, stop_words = 'english')
    dat_tfidf = vectorizer.fit_transform([' '.join(d) for d in documents])
    nmf_model = NMF(n_components=8, random_state=1, alpha=.1, l1_ratio=.5).fit(dat_tfidf)
    nmf_topic_embed = nmf_model.transform(dat_tfidf)
    return vectorizer.get_feature_names, nmf_model, nmf_topic_embed, dat_tfidf

## load syntactic parsing result from an json file
class syntactic_loader(object):

    def __init__(self, parsed):
        self.parsed = parsed

    def generate_average_depth(self):
        return np.mean([self.compute_tree_depth(s.get('parse')) for s in self.parsed.get('sentences')])
        
    def compute_tree_depth(self, parse):
        ptree = ParentedTree.fromstring(parse)
        ##depth = self.__get_tree_depth(ptree)
        return ptree.height()
                
class syntactic_parser(object):
    
    def __init__(self, key):
        filename = key[(key.rfind('/') + 1):]
        self.key = key
        self.filename = filename
        self.outfilename = self.filename[:self.filename.rfind('.')] + '_syntax.json'
        logger.info('Input: %s, output: %s', self.filename, self.outfilename)
        
    def __load_data(self):
        self.__download_file()
        self.__load_text()
        self.__delete_file()
        logger.info("%s is loaded.", self.filename)
        
    def parse(self):
        self.__load_data()
        self.__parse_text()
        logger.info('Text is parsed.')
        syntactic_processor = syntactic_loader(self.parsed)
        self.avg_constituency_depth = syntactic_processor.generate_average_depth()
        print 'Average constituency depth ', self.avg_constituency_depth
        self.__output()
        return {'avg_constituency_depth': self.avg_constituency_depth,
               'avg_sentence_length': self.avg_sentence_length,
               'avg_word_length': self.avg_word_length,
               'num_sentences': self.n_sentences,
                'flesch_reading_ease': self.flesch_reading_ease,
                'flesch_kincaid_grade': self.flesch_kincaid_grade,
               'year_president': self.key[(self.key.rfind('/') + 1): self.key.rfind('.')],
               'year': self.key[(self.key.rfind('/') + 1): self.key.rfind('-')],
               'president': self.key[(self.key.rfind('-') + 1): self.key.rfind('.')]}
    
    def print_parse(self):
        s = np.random.choice(self.parsed.get('sentences'), 1)
        print s[0].get('parse').replace('(', '[').replace(')', ']')
        ##ptree = ParentedTree.fromstring(s[0].get('parse'))
        ##ptree.pretty_print()
    
    def __download_file(self):
        ## download
        command = 'aws s3 cp {} {}/'.format(self.key, local_data_dir)
        status = subprocess.call(command.split(' '))
        if status != 0:
            logger.error('Error in downloading %s', self.key)
        else:
            logger.info('Downloaded file %s to local.', self.key)
        ##subprocess.check_call(['gzip', '-d', local_data_dir + self.filename + '.gz'])
    
    def __delete_file(self):
        os.remove(local_data_dir + self.filename)

    def __load_text(self):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        with codecs.open('{}/{}'.format(local_data_dir, self.filename), 'r', encoding = 'utf8', errors = 'ignore') as f:
            data = f.read()
        self.flesch_reading_ease = textstat.flesch_reading_ease(data)
        self.flesch_kincaid_grade = textstat.flesch_kincaid_grade(data)
        sentences = tokenizer.tokenize(data)
        self.n_sentences = textstat.sentence_count(data)
        self.avg_sentence_length = textstat.lexicon_count(data, True) * 1. / self.n_sentences
        self.avg_word_length = np.mean([len(w) for s in sentences for w in s.split(' ') if w not in stopwords.words('english')])
        print 'Parse ', len(sentences), ' sentences, average sentence length ', self.avg_sentence_length, ', average word length ', self.avg_word_length
        self.sentences = sentences
        self.tokens = []
        [self.tokens.extend(text_tokenize(sentence)) for sentence in sentences]
        
    def __parse_text(self):
        if exists_in_s3('{}/{}'.format(s3_output_prefix, self.outfilename)):
            self.__load_parse_result()
            return
        ss = CoreNLP('parse', corenlp_jars = ['/home/ec2-user/software/stanford-corenlp-full-2015-12-09/*'])
        self.parsed = ss.parse_doc(self.sentences)
        ss.cleanup()
        
    def __load_parse_result(self):
        ## download
        command = 'aws s3 cp s3://{}/{}/{} {}/'.format(s3_bucket, s3_output_prefix, self.outfilename, local_data_dir)
        print command
        status = subprocess.call(command.split(' '))
        if status != 0:
            logger.error('Error in downloading %s', self.outfilename)
        else:
            logger.info('Downloaded file %s to local.', self.outfilename)
        with open('{}/{}'.format(local_data_dir, self.outfilename), 'r') as f:
            self.parsed = json.load(f)
        
    def __output(self):
        with open(local_data_dir + self.outfilename, 'wb') as f:
            json.dump(self.parsed, f)
        logger.info('Parsed result written locally.')
        ## upload the result to Charlie
        command = "aws s3 cp {}/{} s3://{}/{}/{}".format(local_data_dir, self.outfilename, s3_bucket, s3_output_prefix, self.outfilename)
        status = subprocess.call(command.split(' '))
        if status == 1:
            logger.error('Parsed result upload error.')
        elif status == 0:
            logger.info('Parsed result uploaded.')
        ## delete from local
        os.remove(local_data_dir + self.outfilename)

setup_path()
setup_log()

ls = list_in_s3(s3_input_prefix + '/')

all_keys = list_in_s3('{}/'.format(s3_input_prefix))
all_keys = ['s3://{}/{}/{}'.format(s3_bucket, s3_input_prefix, x) for x in all_keys if x[(len(x) - 4):] == '.txt']
print len(all_keys), ' files to be processed in total.'

def process_one_file(key):
    file_parser = syntactic_parser(key)
    parsed = file_parser.parse()
    if key == all_keys[0] or key == all_keys[len(all_keys) - 1]:
        file_parser.print_parse()
    return parsed, file_parser.tokens

pool = Pool(16)
results = pool.map(process_one_file, reversed(all_keys))


# In[22]:

processed_data = [x for x, _ in results]
documents = [x for _, x in results]
               
processed_data = pd.DataFrame(processed_data)
processed_data.loc[:, 'year'] = processed_data.year.astype(int)
processed_data.loc[:, 'party'] = 'Republican'
processed_data.loc[processed_data.year < 1828, 'party'] = 'Other'
processed_data.loc[np.in1d(processed_data.year.tolist(), [1841, 1849]), 'party'] = 'Other'
processed_data.loc[np.in1d(processed_data.year.tolist(), [1845, 1885, 1893, 1977]), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1828) & (processed_data.year < 1841), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1852) & (processed_data.year < 1860), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1932) & (processed_data.year < 1952), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1960) & (processed_data.year < 1968), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1912) & (processed_data.year < 1920), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 1992) & (processed_data.year < 2000), 'party'] = 'Democrat'
processed_data.loc[(processed_data.year > 2008) & (processed_data.year < 2016), 'party'] = 'Democrat'

print processed_data.head()
print processed_data.party.value_counts()

## Fit NMF model
nmf_dictionary, nmf_model, nmf_topic_embed, tfidf_embed = build_nmf_model(documents)
##print_top_words(nmf_model, nmf_dictionary, 10)

## Fit LDA model
dictionary, lda_model, lda_topic_embed = build_lda_model(documents, num_topics = 8)
        
## compute embeddings
mds = manifold.MDS(n_components=2, metric = True, max_iter=100, eps=1e-9, random_state=1, dissimilarity="precomputed", n_jobs=1)
lda_mds_embed = mds.fit(pairwise_distances(lda_topic_embed, metric = 'cosine')).embedding_
nmf_mds_embed = mds.fit(pairwise_distances(nmf_topic_embed, metric = 'cosine')).embedding_
tfidf_mds_embed = mds.fit(pairwise_distances(tfidf_embed, metric = 'cosine')).embedding_

lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors = 10, method = 'modified', tol=1e-9, max_iter = 1000)
lda_lle_embed = lle.fit(lda_topic_embed).embedding_
nmf_lle_embed = lle.fit(nmf_topic_embed).embedding_
tfidf_lle_embed = lle.fit(tfidf_embed.toarray()).embedding_

processed_data.loc[:, 'lda_embed1'] = lda_lle_embed[:, 0]
processed_data.loc[:, 'lda_embed2'] = lda_lle_embed[:, 1]
processed_data.loc[:, 'nmf_embed1'] = nmf_lle_embed[:, 0]
processed_data.loc[:, 'nmf_embed2'] = nmf_lle_embed[:, 1]
processed_data.loc[:, 'tfidf_embed1'] = tfidf_lle_embed[:, 0]
processed_data.loc[:, 'tfidf_embed2'] = tfidf_lle_embed[:, 1]

processed_data.to_csv('{}/processed_data.csv'.format(local_data_dir))

command = "aws s3 cp {}/processed_data.csv s3://{}/{}/processed_data.csv".format(local_data_dir, s3_bucket, s3_output_prefix)
status = subprocess.call(command.split(' '))

##lda_tsne_embed = bh_sne(lda_topic_embed)
##nmf_tsne_embed = bh_sne(nmf_topic_embed)


# In[37]:

get_ipython().magic(u'matplotlib inline')
import pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def scatter_with_legend(col1, col2, title, position = 'upper left'):
    fig, ax = plt.subplots()
    parties = list(set(processed_data.party))
    colors = ['r', 'g', 'b']
    colormap = dict(zip(parties, colors))
    plots = []
    for party in parties:
        plot = ax.scatter(processed_data.loc[processed_data.party == party, col1].tolist(),                           processed_data.loc[processed_data.party == party, col2],
                          color = colormap[party])
        plots.append(plot)
    label = 'year_president' if col1 != 'year' else 'president'
    for i in range(processed_data.shape[0]):
        ax.annotate(processed_data.loc[i, label], (processed_data.loc[i, col1], processed_data.loc[i, col2]),
                   color = colormap[processed_data.party[i]])
    ax.legend(tuple(plots), tuple(parties), loc = position)
    ax.title.set_text(title)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    
scatter_with_legend('lda_embed1', 'lda_embed2', 'LDA Embedding', 'upper right')
scatter_with_legend('tfidf_embed1', 'tfidf_embed2', 'TF-idF Embedding')
scatter_with_legend('year', 'avg_word_length', 'Average Word Length', 'upper right')
scatter_with_legend('year', 'avg_sentence_length', 'Average Sentence Length', 'upper right')
scatter_with_legend('year', 'avg_constituency_depth', 'Average Parse Tree Depth', 'upper right')
scatter_with_legend('year', 'num_sentences', 'Number of Sentences')
scatter_with_legend('year', 'flesch_reading_ease', 'Flesch Reading Ease Score')
scatter_with_legend('year', 'flesch_kincaid_grade', 'Flesch Kincaid Grade Score', 'upper right')

