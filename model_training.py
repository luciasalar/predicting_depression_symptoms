#!/usr/bin/env python
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import re
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from ruamel import yaml
import datetime
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
from CountVect import *
import time 
import logging
import csv

def load_experiment(path_to_experiment):
	
	data = yaml.safe_load(open(path_to_experiment))
	return data

class PrepareData():

	def __init__(self):
		'''define the main path'''
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

	def process_text(self):
		'''read text data, clean data then concatenate string according to userid'''
		c= Count_Vect()
		text = c.prepare_text_data()
		clean_text = c.parallelize_dataframe(text, c.get_precocessed_text)

		return clean_text

	def merge_data(self, moodFeatureFile):
	    '''merging features, LIWC, mood vectors'''
	    mood_feature = pd.read_csv(self.path + moodFeatureFile)
	    mood_feature.columns = [str(col) + '_mood' for col in mood_feature.columns]
	    mood_feature  = mood_feature.rename(columns = {"Unnamed: 0_mood":"userid"}) 
	    #select frequent users, here you need to change the matched user files if selection criteria changed
	    participants = pd.read_csv(self.path + 'participants_matched.csv')
	    participants  = participants[['userid','cesd_sum']]
	    #merge with text feature
	    text = self.process_text()
	    all_features = pd.merge(text, mood_feature,on = 'userid')
	    #merge with  liwc
	    #liwc = pd.read_csv(path + 'liwc_scores.csv')
	    #liwc.columns = [str(col) + '_liwc' for col in liwc.columns]
	    #merge with sentiment
	    feature_cesd = pd.merge(all_features, participants, on = 'userid')
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



class ItemSelectorText(BaseEstimator, TransformerMixin):
	'''feature selector for pipline '''
	def __init__(self, key):
	    self.key = key

	def fit(self, x, y=None):
	    return self

	def transform(self, data_dict):
	    return data_dict[self.key]


class TrainingClassifiers(): 
	def __init__(self):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test= y_test
		self.parameters = parameters

	def select_features(self,features_list):
		'''
		select columns with names in feature list then convert it to a transformer object 
		feature_list is in the dictionary 
		'''
		selected_features = FunctionTransformer(lambda x: x[[i for i in self.X_train.columns if features_list in i]], validate=False)

		return selected_features

	def setup_pipeline(self,features_list, classifier):
		'''set up pipeline'''
		
		pipeline = Pipeline([
		    
		    ('feats', FeatureUnion([
		    #generate count vect features
		        ('text', Pipeline([
		            ('selector', ItemSelectorText(key='text')),
		            #('cv', CountVectorizer()),
		            ('tfidf', TfidfVectorizer()),

		             ])),
		  #select other features, feature sets are defines in the yaml file
		     	('other_features', Pipeline([

		 			('selector',  self.select_features(features_list)),
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

	def test_model(self, path, classifier, features_list):
		'''test model and save data'''
		start = time.time()
		#training model
		print('getting pipeline...')
		#the dictionary returns a list, here we extract the string from list use [0]
		pipeline = c.setup_pipeline(features_list[0], eval(classifier)())

		print('training...')
		grid_search = c.training_models(pipeline)
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
		print('running time:', end-start)

		return precision,recall,fscore,support,grid_search


def loop_experiment():
	p = PrepareData()
	X_train, X_test, y_train, y_test = p.get_train_test_split()
	experiment = load_experiment(p.path + '../experiment/experiment.yaml')


	f = open(p.path + 'results/result.csv' , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support'] +['time'] + ['model'] +['feature_set'])

	

	for classifer in experiment['experiment']:
		for feature in experiment['features']:

			

			#pipeline = set_pipeline(experiment['features'][feature], eval(classifer)())
			parameters = experiment['experiment'][classifer]
			c = TrainingClassifiers()
			precision, recall, fscore, support,grid_search = c.test_model(p.path, classifer, feature)

			result_row = [[grid_search.best_score_, grid_search.best_params_, precision,recall,fscore,support, str(datetime.datetime.now()), classifer, feature]]


		
			writer_top.writerows(result_row)
			f.close
	f.close
	return grid_search

#preparing data

#parameters = experiment['experiment']['sklearn.ensemble.RandomForestClassifier']
#training classifiers

loop_experiment()

# start = time.time()
# grid_search = c.training_models(pipeline, parameters)
# end = time.time()
# print('running time:', end-start)



class plotting_results():

	def __init__(self):
		self.grid_search = grid_search
		self.path = path


	def get_feature_importance(self, X_train):
		importance = self.grid_search.best_estimator_.named_steps['clf'].steps[1][1].feature_importances_
		print(X_train.shape)
		feat_importances = pd.Series(importance, index=X_train.columns)
		feat_importances.nlargest(20).plot(kind='barh')
		plt.show()

		fea_df = pd.DataFrame(feat_importances)
		fea_df['features'] = fea_df.index
		fea_df.columns = ['importance','features']
		fea_df.to_csv(self.path + 'results/topFea_test.csv')

	def calculate_feature_importance_rf():
	    importances = self.grid_search.best_estimator_.named_steps['clf'].steps[1][1].feature_importances_
	    std_importances = np.std([tree.feature_importances_ for tree in grid_search.best_estimator_.named_steps['clf'].steps[1][1].estimators_],
	            axis=0)
	    indices = np.argsort(importances)[::-1]
	    return importances, std_importances, indices

def plot_feature_importance_rf(matrix, mean_importance, std_importance, indices, fig_size):
    tfidf_names = grid_search.best_estimator_.named_steps['feats'].transformer_list[0][1].named_steps['cv'].get_feature_names()
    feature_names = pd.concat([pd.Series(tfidf_names), pd.Series(matrix.columns.tolist)])
    sorted_predictors = [x for (y, x) in sorted(zip(mean_importance, feature_names[:-1]), reverse=True)]

    fig, ax = plt.subplots(figsize=fig_size)
    
    plt.bar(range(len(indices)), mean_importance[indices],
            color="m", yerr=std_importance[indices], align="center")
    plt.xticks(range(len(indices)), sorted_predictors, rotation=90)
    plt.xlim([-1, len(indices)])
    ##£
    plt.show()
   
    return fig, ax


#if __name__ == '__main__':

	






#get_feature_importance()






# parameters = [{
#             'estimator__clf__svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'estimator__clf__svc__gamma': [0.01, 0.001, 0.0001],
#             'estimator__clf__svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 10] , 'estimator__clf__svc__class_weight':['balanced']}]
# parameters = [{'estimator__clf__feature_selection__estimator__max_depth': [5,10,20], 'estimator__clf__feature_selection__estimator__max_leaf_nodes': [50, 100, 200],
#                'estimator__clf__log__C':[1.0, 2.0, 3.0], 'estimator__clf__log__class_weight': ['balanced'], 'estimator__clf__log__multi_class': ['ovr', 'multinomial']}]


# pred = pipe1.predict(X_test)
# print(classification_report(y_test, pred))










