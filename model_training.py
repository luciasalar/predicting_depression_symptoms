#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import sklearn
import nltk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from ruamel import yaml
import datetime
import matplotlib.pyplot as plt
import os
from CountVect import *
from topic_model import *
import time 
import logging
import csv
from sklearn.decomposition import TruncatedSVD
from construct_mood_feature import *


def load_experiment(path_to_experiment):
	#load experiment 
	data = yaml.safe_load(open(path_to_experiment))
	return data

class PrepareData():

	def __init__(self):
		'''define the main path'''
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
		self.timeRange = 365

	def liwc_preprocess(self, timeRange):
		'''aggregate text then process text with liwc'''
		#get posts within time range
		participants = pd.read_csv(self.path + 'participants_matched.csv') 
		sentiment_pre = pd.read_csv(self.path + 'status_sentiment.csv')  
		text = mood.get_relative_day(sentiment_pre, timeRange)
		#merge posts with matched participants
		text_merge = pd.merge(participants, text, on = 'userid') 
		text_fea = text_merge[['userid','text']]
		#aggregate text    
		'''DONT USE JOIN STRING!! USE STR.CAT OTHERWISE YOU WILL LOSE A LOT OF INFORMATION  '''
		text_fea3 = text_fea.groupby(['userid'])['text'].apply(lambda x: x.str.cat(sep=',')).reset_index()
		text_fea3 = text_fea3.drop_duplicates() #remove duplication
		text_fea3.to_csv(self.path + 'aggregrate_text_{}.csv'.format(timeRange))

		return text_fea3

	# def process_text(self, timeRange):
	# 	'''read text data, clean data then concatenate string according to userid'''
	# 	c= Count_Vect()
	# 	text = self.liwc_preprocess(timeRange) #retrieve text within time Range

	# 	text2 = c.prepare_text_data(text) #define time range for features
	# 	clean_text = c.parallelize_dataframe(text2, c.get_precocessed_text)

	# 	return clean_text


	def sentiment_data(self):
		'''generate sentiment features in time window X'''
		#read sentiment data
		mood = MoodFeature()
		sentiment_pre = pd.read_csv(self.path + 'status_sentiment.csv')
		sentiment = mood.get_relative_day(sentiment_pre, 365)

		sentiment = sentiment[['userid','time','positive','negative']]
		#compute sentiment sum on each post
		sentiment['sentiment_sum'] = sentiment['positive'] + sentiment['negative']

		#compute average sentiment
		s_mean = sentiment.groupby('userid')['sentiment_sum'].mean().to_frame().reset_index() 
		#compute sentiment sd
		s_sd = sentiment.groupby('userid')['sentiment_sum'].std().to_frame().reset_index() 
		#compute number of post
		p_count = sentiment.groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 
		p_count = p_count.rename(columns={"sentiment_sum": "post_c"})
		#count positive and negative sentiment post
		p_neg_count = sentiment.loc[sentiment['sentiment_sum'] < 0].groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 
		p_pos_count = sentiment.loc[sentiment['sentiment_sum'] >= 0].groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 

		p_total = pd.merge(p_count, p_pos_count, on = 'userid', how = 'left') 
		p_total = pd.merge(p_total, p_neg_count, on = 'userid', how = 'left')
		p_total = p_total.rename(columns={"sentiment_sum_x": "pos_count", "sentiment_sum_y": "neg_count"})
		p_total = p_total.fillna(0) #convert NA to 0
		#compute positive and negative ratio
		p_total['pos_per'] = p_total['pos_count']/p_total['post_c']
		p_total['neg_per'] = p_total['neg_count']/p_total['post_c']

		per_fea = p_total[['userid','post_c','pos_per','neg_per']]
		#return df with sentiment features
		mean_sd = pd.merge(s_sd, s_mean, on = 'userid', how = 'left')
		mean_sd = mean_sd.rename(columns={"sentiment_sum_x": "sent_sd", "sentiment_sum_y": "sent_mean"})

		sentiment_fea = pd.merge(mean_sd, per_fea, on = 'userid', how = 'left')
		#rename columns
		sentiment_fea.columns = [str(col) + '_sentiment' for col in sentiment_fea.columns]
		sentiment_fea  = sentiment_fea.rename(columns = {"userid_sentiment":"userid"})
		return sentiment_fea


	# def posting_frequency():
	# 	'''compute posting frequency in the past x days'''
	# 	pass
	def demographic(self):
		'''create demographic features '''
		demographics = pd.read_csv(self.path + 'participants_matched.csv')

	def topic_modeling(self, text, topic_number):
		'''return topic features'''
		t = LDATopicModel() 
		lda_score_df, lda_model = t.get_lda_score(text, topic_number) 
		return lda_score_df, lda_model



	def merge_data(self, moodFeatureFile):

	    '''merging features, LIWC, mood vectors'''

	    mood_feature = pd.read_csv(self.path + moodFeatureFile)
	    mood_feature.columns = [str(col) + '_mood' for col in mood_feature.columns]
	    mood_feature  = mood_feature.rename(columns = {"Unnamed: 0_mood":"userid"}) 
	    #select frequent users, here you need to change the matched user files if selection criteria changed
	    participants = pd.read_csv(self.path + 'participants_matched.csv')
	    participants  = participants[['userid','cesd_sum']]
	    #merge with text feature
	    c= Count_Vect()
	    text = self.liwc_preprocess(self.timeRange)

	    text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
	    text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
	    text = c.parallelize_dataframe(text, c.get_precocessed_text)
	    #text['text'] = text['text'].apply(lambda x: c.remove_single_letter(x))

	    all_features = pd.merge(text, mood_feature,on = 'userid')
	    #load  liwc (including WC)
	    liwc = pd.read_csv(self.path + 'liwc_scores_1year.csv')
	    liwc.columns = [str(col) + '_liwc' for col in liwc.columns]
	    liwc  = liwc.rename(columns = {"userid_liwc":"userid"})
	    #load sentiment feature (including post count)
	    sentiment = self.sentiment_data()
	   
	    #posting frequency


	    #topic ratio LDA
	    
	    # topic_text = all_features[['userid','text']]
	    # topics, ldamodel = self.topic_modeling(topic_text, 30) #topic number and time range for text
	    # topics.columns = [str(col) + '_topic' for col in topics.columns]
	    # topics  = liwc.rename(columns = {"userid_topic":"userid"})

	    #merge all features
	    all_features = pd.merge(liwc, all_features, on = 'userid')
	    all_features2 = pd.merge(all_features, sentiment, on = 'userid')
	    #all_features2 = pd.merge(all_features2, topics, on = 'userid')
	    feature_cesd = pd.merge(all_features2, participants, on = 'userid')

	    return feature_cesd

	

	def get_y(self, feature_df):
		'''get y '''
		y = feature_df['cesd_sum']
		return y

	def recode_y(self, y, threshold):
		'''recode y to binary according to a threshold'''
		new_labels = []
		for i in y:
			if i <= threshold:
				i = 0
			if i > threshold:
				i = 1
			new_labels.append(i)
		return new_labels

	def pre_train(self):
		'''merge data, get X, y and recode y '''
		f = self.merge_data('mood_vectors/mood_vector_frequent_user.csv')
		y_cesd = self.get_y(f)
		y_recode = self.recode_y(y_cesd, 22) 
		X = f.drop(columns=['userid', 'cesd_sum'])
		return X, y_recode

	def get_train_test_split(self):
	    '''split train test'''
	    X, y = self.pre_train()
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 300)
	    return X_train, X_test, y_train, y_test



class ColumnSelector(BaseEstimator, TransformerMixin):
	'''feature selector for pipline '''
	def __init__(self, columns):
		self.columns = columns

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, pd.DataFrame)

		try:
		    return X[self.columns]
		except KeyError:
		    cols_error = list(set(self.columns) - set(X.columns))
		    raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

	def get_feature_names(self):
		return self.columns.tolist
		    

class TrainingClassifiers(): 
	def __init__(self):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test= y_test
		self.parameters = parameters
		self.experiment = experiment
		self.tfidf_words = tfidf_words 

	def select_features(self,features_list):
		'''
		select columns with names in feature list then convert it to a transformer object 
		feature_list is in the dictionary 
		'''
		fea_list = []
		for fea in features_list: #select column names with keywords in dict
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)
		#flatten a list
		flat = [x for sublist in fea_list for x in sublist]
		#convert to transformer object
		#selected_features = FunctionTransformer(lambda x: x[flat], validate=False)

		return flat

	def get_other_feature_names(self, feature_list):
		fea_list = []
		for fea in feature_list: #select column names with keywords in dict
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)
		#flatten a list
		flat = [x for sublist in fea_list for x in sublist]
		#convert to transformer object
		return flat


	def setup_pipeline(self,features_list, classifier, tfidf_words):
		'''set up pipeline'''
		features_col = self.get_other_feature_names(features_list)


		pipeline = Pipeline([
			#ColumnSelector(columns = features_list),
		    
		    ('feats', FeatureUnion([
		  #   #generate count vect features
		        ('text', Pipeline([
		            ('selector', ColumnSelector(columns='text')),
		            #('cv', CountVectorizer()),
		            ('tfidf', TfidfVectorizer(max_features = tfidf_words, ngram_range = (1,3), stop_words ='english', max_df = 0.50, min_df = 0.0025)),
		           # ('svd', TruncatedSVD(algorithm='randomized', n_components=300))
		             ])),
		  # # select other features, feature sets are defines in the yaml file
		 

		 		('other_features', Pipeline([

		 			('selector',  ColumnSelector(columns = features_col)),
		 		])),

		     ])),


		       ('clf', Pipeline([  
		       ('scale', StandardScaler(with_mean=False)),  #scale features
		        ('classifier',  classifier),  #classifier
		   
		         ])),
		])
		return pipeline

	

	def training_models(self, pipeline):
		'''train models with grid search'''
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv = 5, scoring='accuracy')
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		return grid_search

	def test_model(self, path, classifier, features_list, tfidf_words):
		'''test model and save data'''
		start = time.time()
		#training model
		print('getting pipeline...')
		#the dictionary returns a list, here we extract the string from list use [0]
		pipeline = self.setup_pipeline(features_list, eval(classifier)(), tfidf_words)

		print('features', features_list)
		grid_search = self.training_models(pipeline)
		#make prediction
		print('prediction...')
		y_true, y_pred = self.y_test, grid_search.predict(self.X_test)
		precision,recall,fscore,support=precision_recall_fscore_support(y_true,y_pred,average='macro')
		#store prediction result
		y_pred_series = pd.DataFrame(y_pred)
		result = pd.concat([pd.Series(y_true).reset_index(drop=True), y_pred_series], axis = 1)
		result.columns = ['y_true', 'y_pred']
		result.to_csv(path + 'results/best_result2.csv' )
		end = time.time()
		print('running time:{}, fscore:{}'.format(end-start, fscore))

		return precision,recall,fscore,support,grid_search,pipeline

def get_liwc_text(self, timeRange):  
    '''run this to get text for liwc, you need to define the time range in days '''
    prepare = PrepareData()
    sen = prepare.sentiment_data()
    liwc_text = prepare.liwc_preprocess(timeRange)
    return liwc_text


#for debug 
# prepare = PrepareData()

# X_train, X_test, y_train, y_test = prepare.get_train_test_split()
# experiment = load_experiment(prepare.path + '../experiment/experiment.yaml')

# parameters = experiment['experiment']['sklearn.ensemble.RandomForestClassifier']
# features_list = experiment['features']['set3']

# tfidf_words = 2000
# training = TrainingClassifiers()
# precision,recall,fscore,support,grid_search,pipeline = training.test_model(prepare.path, 'sklearn.ensemble.RandomForestClassifier', features_list, tfidf_words)



# fea_list = []
# for fea in features_list: 
# 	f_list = [i for i in X_train.columns if fea in i]
# 	fea_list.append(f_list)
# #flatten a list
# flat = [x for sublist in fea_list for x in sublist]
# #convert to transformer object
# selected_features = FunctionTransformer(lambda x: x[flat], validate=False)

#pipeline = training.setup_pipeline(features_list, 'sklearn.ensemble.RandomForestClassifier', tfidf_words)
#grid_search = training.training_models(pipeline)
#training.test_model(features_list, 'sklearn.ensemble.RandomForestClassifier', tfidf_words)


# c = Count_Vect()
# text = prepare.liwc_preprocess(365)
# # text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
# # text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
# # text = c.parallelize_dataframe(text, c.get_precocessed_text)
# text.to_csv(path + 'process.csv')


# t = LDATopicModel()
# ldaText = prepare.liwc_preprocess(365)
# ldaText.to_csv(prepare.path + 'sample.csv')
# # lda_score_df, lda_model = t.get_lda_score(ldaText, 30)


# 		'''aggregate text then process text with liwc'''
# 		#get posts within time range
# path = prepare.path
# participants = pd.read_csv(path + 'participants_matched.csv') 
# sentiment_pre = pd.read_csv(path + 'status_sentiment.csv')  
# text = mood.get_relative_day(sentiment_pre, 365)
# #merge posts with matched participants
# text_merge = pd.merge(participants, text, on = 'userid') 
# text_fea = text_merge[['userid','text']]

# #aggregate text
# #text_fea = text_fea.groupby(['userid'])['text'].apply(lambda text: ''.join(text.to_string(index=False)))

# text2 = text_fea.groupby(['userid'])['text'].apply(lambda x: x.str.cat(sep=',')).reset_index()
# text2.to_csv(path + 'pre_process.csv')
# text_fea2 = text_fea.drop_duplicates() #remove duplication


	



prepare = PrepareData()

X_train, X_test, y_train, y_test = prepare.get_train_test_split()
experiment = load_experiment(prepare.path + '../experiment/experiment.yaml')

f = open(prepare.path + 'results/result.csv' , 'a')
writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
writer_top.writerow(['best_scores'] + ['best_parameters'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support'] +['time'] + ['model'] +['feature_set'] +['tfidf_words'])

for classifier in experiment['experiment']:

	for feature in experiment['features']: #loop feature sets
		for word_n in experiment['tfidf_features']['max_fea']: #loop tfidf features
			tfidf_words = word_n

			parameters = experiment['experiment'][classifier]
			print('parameters are:', parameters)
			training = TrainingClassifiers()
			
			precision, recall, fscore, support, grid_search, pipeline = training.test_model(prepare.path, classifier, feature, tfidf_words)

			print('printing fscore', fscore)
			result_row = [[grid_search.best_score_, grid_search.best_params_, precision,recall,fscore,support, str(datetime.datetime.now()), classifier, feature,tfidf_words]]

			writer_top.writerows(result_row)

	f.close
				
	
	










