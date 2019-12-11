import pandas as pd 
import collections 
import numpy as np
from collections import defaultdict
# import pymc3 as pm
# from theano import shared
# from pymc3.distributions.timeseries import GaussianRandomWalk
# from scipy import optimize
#from time import mktime as mktime
import time
from datetime import datetime
import GPy
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import os
import csv

#other paths
#path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
#path = '/Users/lucia/phd_work/predicting_depression_symptoms/data/'

'''for each user generate the mood (category) in the past X days (x is the time window) '''


class SelectParticipants():
	def __init__(self):
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/' 
		#self.path = '/home/lucia/phd_work/predicting_depression_symptoms/data/' 
		self.participants = pd.read_csv(self.path + 'participants_matched.csv')
		self.frequent_users = pd.read_csv(self.path + 'frequent_users.csv')

	def process_participants(self):
		'''process participant files, select useful columns '''
		self.frequent_users.columns = ['rowname', 'userid','freq']
		participants  = self.participants[['userid','time_completed','cesd_sum']]

		#frequent participants (commment this one if not needed)
		participants = pd.merge(self.frequent_users, participants, on='userid')
		participants.drop('rowname', axis=1, inplace=True)
		participants.drop('freq', axis=1, inplace=True)

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


	def change_timestamp(self, sentiment_selected):
		'''convert time to timestamp (Epoch, also known as Unix timestamps, is the number of seconds (not milliseconds!) 
		that have elapsed since January 1, 1970 at 00:00:00 GMT, then divide timestamp by number of hours in a year'''
		userTime = self.get_user_time_obj(sentiment_selected)

		mydict = lambda: defaultdict(mydict)
		
		new_dict = mydict()
		for k, v in userTime.items():
			timestamp_hour = []
			for timestamp in v['postTime']:
				#print(type(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple()))
				time_num = time.mktime(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple())
				new_time = round(time_num/8760,2) #(24 hours * 365 days)
				timestamp_hour.append(new_time)
			new_dict[k]['senti'] = v['senti']
			new_dict[k]['time'] = timestamp_hour

		return new_dict


class GaussianProcess:
	def __init__(self, userTimeObj, path):
		self.userTimeObj = userTimeObj
		self.path = path 
	
	def GP_regression(self, X, Y):
		'''the GP model '''
		# define the covariance kernel 
		kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
		m = GPy.models.GPRegression(X,Y,kernel)

		#optimization  
		#This selects random (drawn from N(0,1)) initializations for the parameter values, optimizes each, and sets the model to the best solution found.
		m.optimize_restarts(num_restarts = 10)


		return m


	def get_model_scores(self):
		'''get smoothing score for each user  '''

		mydict = lambda: defaultdict(mydict)
		GP_models = mydict()
		count = 0 
		for user, v in self.userTimeObj.items():
			if user is not None:
				#train a model on each user
				#print(len(np.asarray(v['time'])),len(np.asarray(v['senti'])))
				GP_models[user]['model'] = self.GP_regression(np.asarray(v['time']).reshape(-1,1),np.asarray(v['senti']).reshape(-1,1))
				#fig = m.plot()
				#fig = GP_models[user]['model'].plot()
				#plt.show(fig)
				#plt.savefig(self.path +'plots/gp/gp_process{}'.format(user))
			
				count = count +1 
				if count == 100:
					break
		return GP_models

	def save_results(self):
		'''save model results and plots '''
		models = self.get_model_scores()

		file_exists = os.path.isfile(self.path + 'result/GPresults.csv')
		f = open(self.path + 'results/GPresults.csv' , 'a')
		writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)

		if not file_exists:
			writer_top.writerow(['userid'] + ['model_para'])
		for k, m in models.items():
			result_row = [[k, m['model']]]
			writer_top.writerows(result_row)
			
			fig = m['model'].plot()
			for i in fig: 
				#plt.show()
				plt.savefig(path +'results/plots/GP/gp_process{}'.format(k))



#read sentiment file
sp = SelectParticipants()
path = sp.path
participants = sp.process_participants()

#here you define the number of days you want to use as feature and the time window for mood
senti = SentiFeature(path = path, participants = participants)

sentiment_selected = senti.get_relative_day(365)
userTime = senti.change_timestamp(sentiment_selected)

g = GaussianProcess(userTimeObj = userTime, path = path)
g.save_results()
#models = g.get_model_scores()

# sorted_senti = senti.SortTime(sentiment_selected)
# Y1 = sorted_senti['sentiment_sum'].values.reshape(-1,1)
# Y1 = Y1[1:100]
# #X1 = sorted_senti['time'] 


		#print(type(fig))
	#plt.plot(m['model'])
	
	
	

	#print(m['model'])
	

