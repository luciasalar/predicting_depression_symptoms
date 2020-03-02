import pandas as pd 
import collections 
import numpy as np
from collections import defaultdict
# import pymc3 as pm
# from theano import shared
# from pymc3.distributions.timeseries import GaussianRandomWalk
# from scipy import optimize
#from time import mktime as mktime
import datetime
import time
# from datetime import datetime
import GPy
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import os
import csv
from construct_mood_transition_feature import *

#other paths
#path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
#path = '/Users/lucia/phd_work/predicting_depression_symptoms/data/'

'''for each user generate the mood (category) in the past X days (x is the time window) '''


class SelectParticipants:
	def __init__(self):
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/' 
		#self.path = '/Users/lucia/hawksworth/predicting_depression_symptoms/data/'
		#self.path = '/home/lucia/phd_work/predicting_depression_symptoms/data/' 
		self.participants = pd.read_csv(self.path + 'participants_matched.csv')
		self.frequent_users = pd.read_csv(self.path + 'frequent_users.csv')
		self.annotate = pd.read_csv(self.path + 'annotate_data.csv')

	def process_participants(self):
		'''process participant files, select useful columns '''
		self.frequent_users.columns = ['rowname', 'userid','freq']
		participants  = self.participants[['userid','time_completed','cesd_sum']]

		#frequent participants (commment this one if not needed)
		participants = pd.merge(self.frequent_users, participants, on='userid')
		participants.drop('rowname', axis=1, inplace=True)
		participants.drop('freq', axis=1, inplace=True)

		return participants

	def processed_annotation(self):
		'''adjust labels to positive: 1, negative: -1, neutral: 0, mix: 0 '''
		participants = self.annotate[['userid','negative_ny','time', 'time_diff']]
		participants=participants.rename(columns = {'negative_ny':'sentiment_sum'})
		participants = participants.loc[participants['sentiment_sum'] != 5,] #remove non English
		participants = participants.replace([1, 2, 0, 4], [-1, 1, 0, 0]) #recode values


		return participants



class SentiFeature:

	def __init__(self, path, participants):
		self.path = path
		self.sentiment_status = pd.read_csv(self.path + 'status_sentiment.csv')
		self.participants = participants		


	def sum_sentiment(self, data):
		'''suming sentistregth score'''
		data['sentiment_sum'] = data['positive'] + data['negative']
		return data

	def get_relative_day(self, time_frame):
		'''this function returns date the post is written relatively to the day user complete the cesd '''
		#merge data and 
		sum_sentiment = self.sum_sentiment(self.sentiment_status)
		senti_part = pd.merge(self.participants, sum_sentiment, on = 'userid')
		senti_part['time_diff'] = pd.to_datetime(senti_part['time_completed']) - pd.to_datetime(senti_part['time'])
		#select posts before completed cesd
		senti_part['time_diff'] = senti_part['time_diff'].dt.days
		senti_sel = senti_part[(senti_part['time_diff'] >= 0) & (senti_part['time_diff'] < time_frame)]
		print('there are {} posts posted before user completed cesd and within {} days'.format(senti_sel.shape[0], time_frame))
		return senti_sel


	def SortTime(self,file):
		'''sort post by time'''
		file = file.sort_values(by=['userid','time_diff'],  ascending=True)
		return file

	def get_user_time_obj(self, sentiment_selected):
		'''for each user, there's a value vector for sentiment and value vector for time '''
		sorted_senti = senti.SortTime(sentiment_selected)
		
		mydict = lambda: defaultdict(mydict)
		userTime = mydict()
		preUser = None
		sentiment_all = []
		time_all = []

		for userid, time, sentiment_sum in zip(sorted_senti['userid'], sorted_senti['time'], sorted_senti['sentiment_sum']):
			if preUser == None:
				userTime[userid]['senti'] = sentiment_all
				userTime[userid]['postTime'] = time_all

			if userid == preUser:
				sentiment_all.append(sentiment_sum)
				time_all.append(time)
			
			else:
				userTime[preUser]['senti'] = sentiment_all
				userTime[preUser]['postTime'] = time_all

				sentiment_all = []
				time_all = []
			preUser = userid

		return userTime


	def get_user_mood_obj(self, moodVec):
		'''inser mood vector to a dictionary, for each user, there's a value vector for sentiment and value vector for time '''
		
		mydict = lambda: defaultdict(mydict)
		userTime = mydict()
		preUser = None
		sentiment_all = []
		time_all = []

		moodVector = moodVec.to_numpy()
		moodVec['userid'] = moodVec.index

		for userid, mv in zip(moodVec['userid'], moodVector):
			userTime[userid]['senti'] = mv
			userTime[userid]['time'] = np.arange(len(mv))

		return userTime



	def change_timestamp(self, sentiment_selected, timescale):
		'''convert time to timestamp (Epoch, also known as Unix timestamps, is the number of seconds (not milliseconds!) 
		that have elapsed since January 1, 1970 at 00:00:00 GMT, then divide timestamp by number of hours in a year'''
		userTime = self.get_user_time_obj(sentiment_selected)

		mydict = lambda: defaultdict(mydict)
		
		new_dict = mydict()
		for k, v in userTime.items():
			timestamp_hour = []
			for timestamp in v['postTime']:
				#print(type(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple()))
				time_num = time.mktime(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple())
				new_time = time_num/timescale #(60 min * 60 second)
				timestamp_hour.append(new_time)
			new_dict[k]['senti'] = v['senti']
			new_dict[k]['time'] = timestamp_hour

		return new_dict, timescale

	def change_timestamp_anno(self, sentiment_selected, timescale):
		'''convert time to timestamp (Epoch, also known as Unix timestamps, is the number of seconds (not milliseconds!) 
		that have elapsed since January 1, 1970 at 00:00:00 GMT, then divide timestamp by number of hours in a year'''
		userTime = self.get_user_time_obj(sentiment_selected)

		mydict = lambda: defaultdict(mydict)
		
		new_dict = mydict()
		for k, v in userTime.items():
			timestamp_hour = []
			for timestamp in v['postTime']:
				#print(type(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple()))
				time_num = time.mktime(datetime.datetime.strptime(timestamp, "%d/%m/%Y %H:%M").timetuple())
				new_time = time_num/timescale #(60 min * 60 second)
				timestamp_hour.append(new_time)
			new_dict[k]['senti'] = v['senti']
			new_dict[k]['time'] = timestamp_hour

		return new_dict, timescale


class GaussianProcess:
	def __init__(self, userTimeObj, path):
		self.userTimeObj = userTimeObj
		self.path = path 
	
	def GP_regression(self, X, Y, lengthscaleNum):
		'''the GP model '''
		# define the covariance kernel 
		kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=lengthscaleNum)
		m = GPy.models.GPRegression(X,Y,kernel)

		#optimization  
		#This selects random (drawn from N(0,1)) initializations for the parameter values, optimizes each, and sets the model to the best solution found.
		m.optimize_restarts(num_restarts = 10, robust=True)


		return m, lengthscaleNum


	def get_model_scores(self, lengthscaleNum):
		'''get smoothing score for each user  '''

		mydict = lambda: defaultdict(mydict)
		GP_models = mydict()
		count = 0 
		for user, v in self.userTimeObj.items():
			if user is not None:
				#train a model on each user
				GP_models[user]['model'], GP_models[user]['lengthscale'] = self.GP_regression(np.asarray(v['time']).reshape(-1,1),np.asarray(v['senti']).reshape(-1,1), lengthscaleNum)
			
				count = count +1 
				# if count == 10:
				# 	break
		return GP_models

	def save_results(self, timescale, lengthscale):
		'''save model results and plots '''
		models = self.get_model_scores(lengthscale) 

		file_exists = os.path.isfile(self.path + 'result/GPresults_annotation.csv')
		f = open(self.path + 'results/GPresults_annotation.csv' , 'a')
		writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)

		if not file_exists:
			writer_top.writerow(['userid'] + ['model_para'] + ['lengthscale_init'] +['timescale'])
		for k, m in models.items():
			result_row = [[k, m['model'], m['lengthscale'], timescale]]
			writer_top.writerows(result_row)
			
			fig = m['model'].plot()
			for i in fig: 
				plt.xlabel('Time (time window = 14, step = 3)', fontsize=18)
				plt.legend(loc=2, prop={'size': 12})
				plt.ylabel('Mood of Every 14 Days', fontsize=18)
				plt.xticks(fontsize=18)
				plt.yticks(fontsize=18)
				plt.savefig(path +'results/plots/GP/gp_process_{}'.format(k))
				#plt.show()
			#plt.close()



	#read sentiment file
# sp = SelectParticipants()
# path = sp.path
#participants = sp.process_participants()

# #here you define the number of days you want to use as feature and the time window for mood
#senti = SentiFeature(path = path, participants = participants)

#sentiment_selected = senti.get_relative_day(365) #data in number of days use for training
# userTime, timescale = senti.change_timestamp(sentiment_selected, 3600) #set time scale


# g = GaussianProcess(userTimeObj = userTime, path = path)
# g.save_results(timescale, 168) #set length scale 24*7

#using annotated data or mood vector 
sp = SelectParticipants()
path = sp.path
annotate = sp.processed_annotation() #get annotation data

senti = SentiFeature(path = path, participants = annotate)

mood_vector_feature, windowSzie = mood.get_mood_in_timewindow(365, 14, 3)
mood_vector_feature = mood_vector_feature.fillna(mood_vector_feature.mean()) 
mvObj = senti.get_user_mood_obj(mood_vector_feature)
g = GaussianProcess(userTimeObj = mvObj, path = path)
g.save_results(0, 168) #set length scale 24*7
