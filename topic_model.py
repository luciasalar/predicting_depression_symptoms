import warnings
warnings.filterwarnings('always')
import numpy as np
import pandas as pd
import re
import math
#import multiprocessing
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
import tracemalloc
import datetime

tracemalloc.start()

"""
here we run lda on the extracted entities
"""

class LDATopicModel:
    def __init__(self):
        '''define the main path'''
        #self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'


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


        #text = get_liwc_text(365)
        c= Count_Vect() #initialize text preprocessing class
        
        text['text'] = text['text'].apply(lambda x: c.remove_single_letter(x))
        #clean_text['text'] = clean_text['text'].apply(lambda x: c.lemmatization(x)) #clean data with lemmatization

        text['text'] = text['text'].apply(lambda x: x.split())
    
        dictionary = gensim.corpora.Dictionary(text['text']) #generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        print('running LDA...') 
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=10,  update_every=1, random_state = 300)
        #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=2, workers=10, random_state = 300)

        #getting LDA score 
        lda_score_all = self.get_score_dict(bow_corpus, lda_model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)
        all_lda_score_dfT['userid'] = text['userid']

        pprint(lda_model.print_topics())
        all_lda_score_dfT.to_csv(self.path + 'ldaScores{}.csv'.format(str(datetime.datetime.now())))

        return all_lda_score_dfT, lda_model

# c= Count_Vect()
# path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
# text = pd.read_csv(path + 'status_sentiment.csv') 

# text = text.head(10000)
# text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
# text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
# text = c.get_precocessed_text(text)

# l = LDATopicModel()
# topics, model = l.get_lda_score(text, 30)


