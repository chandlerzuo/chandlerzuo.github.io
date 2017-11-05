import logging
import os
import subprocess

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

def setup_log(tag = 'VOC_TOPICS'):
    global logger
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)

def setup_path(s3_prefix, data_dir):
    global local_data_dir, s3_input_prefix, s3_bucket
    s3_bucket = 's3://xxxxxx'
    local_data_dir = data_dir
    s3_input_prefix = s3_prefix

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

def download_file(key):
    ## download
    command = 'aws s3 cp s3://{}/{} {}/'.format(s3_bucket, key, local_data_dir)
    status = subprocess.call(command.split(' '))
    if status != 0:
        logger.error('Error in downloading %s', key)
    else:
        logger.info('Downloaded file %s to local.', key)
        import re
    return

def plot_tsne(data, perplexity = 30, n_iter = 200):
    tsne = TSNE(perplexity = perplexity, n_components = 2, init = 'pca', n_iter = n_iter)
    low_dim_embs = tsne.fit_transform(data)
    plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1])
