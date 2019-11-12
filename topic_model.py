import warnings
warnings.filterwarnings('always')
import numpy as np
import pandas as pd
import re
import math
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from pprint import pprint
import gensim
import pickle
import collections
#import psycopg2
import time
import spacy
from CountVect import *

"""
here we run lda on the extracted entities
"""

class LDATopicModel():
    def __init__(self):
        '''define the main path'''
        #self.nlp = spacy.load('en', disable=['parser', 'ner'])


    def get_score_dict(self, bow_corpus, lda_model_object):
        """
        get lda score for each document
        """
        all_lda_score = {}
        for i in range(len(bow_corpus)):
            lda_score ={}
            for index, score in sorted(lda_model_object[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                lda_score[index] = score
                od = collections.OrderedDict(sorted(lda_score.items()))
            all_lda_score[i] = od
        return all_lda_score


    def pickle_object(self, object, path_to_save):
        """
        pickle object
        """
        with open(path, 'wb') as handle:
            pickle.dump(object, handle, protocol = pickle.HIGHEST_PROTOCOL)

    # def make_bigrams(self, df):
    #     bigram_l = []
    #     for bigram_mod, text in zip(bigram_mods, df):
    #         bigrams = [bigram_mod[doc] for doc in text]
    #         bigram_l.append(bigrams)
    #     return bigram_l

    



    def get_lda_score(self, text, topics_numbers):

        #prepare = PrepareData()

        #text = get_liwc_text(365)
        c= Count_Vect() #initialize text preprocessing class
        
        text['text'] = text['text'].apply(lambda x: c.remove_noise(x))
        text['text'] = text['text'].apply(lambda x: c.lemmatization(x))

        clean_text = c.parallelize_dataframe(text, c.get_precocessed_text)
        clean_text['text'] = clean_text['text'].apply(lambda x: c.remove_single_letter(x))
        #clean_text['text'] = clean_text['text'].apply(lambda x: c.lemmatization(x)) #clean data with lemmatization

        clean_text['text'] = clean_text['text'].apply(lambda x: x.split())
        #clean_text['text'] = clean_text['text'].apply(lambda x: t.lemmatization(x))
        dictionary = gensim.corpora.Dictionary(clean_text['text']) #generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in clean_text['text']]

        print('running LDA...') 
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=10, workers=multiprocessing.cpu_count(), random_state = 300)

        #getting LDA score 
        lda_score_all = self.get_score_dict(bow_corpus, lda_model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)
        all_lda_score_dfT['userid'] = clean_text['userid']

        pprint(lda_model.print_topics())

        return all_lda_score_dfT, lda_model






# Form Bigrams
# data_words_bigrams = make_bigrams(data_words_nostops)

# # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# # python3 -m spacy download en
# nlp = spacy.load('en', disable=['parser', 'ner'])

# # Do lemmatization keeping only noun, adj, vb, adv
# data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
