import pandas as pd 
import datetime
import collections 
import numpy as np
from collections import defaultdict
# import pymc3 as pm
# from theano import shared
# from pymc3.distributions.timeseries import GaussianRandomWalk
# from scipy import optimize
import time
import datetime
import GPy
import matplotlib.pyplot as plt


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


class GaussianSmoothing:
	def __init__(self, userTimeObj):
		self.userTimeObj = userTimeObj
		self.smoothing = 0.5 
		
	def infer_z(self, y):
		'''smoothing model '''
		model = pm.Model()
		LARGE_NUMBER = 1e5
		with model:
		    smoothing_param = shared(0.9)
		    mu = pm.Normal("mu", sigma=LARGE_NUMBER)
		    tau = pm.Exponential("tau", 1.0/LARGE_NUMBER)
		    z = GaussianRandomWalk("z",
		                           mu=mu,
		                           tau=tau / (1.0 - smoothing_param),
		                           shape=y.shape)
		    obs = pm.Normal("obs",
		                    mu=z,
		                    tau=tau / smoothing_param,
		                    observed=y)

		with model:
			smoothing_param.set_value(self.smoothing)
			res = pm.find_MAP(vars=[z], fmin=optimize.fmin_l_bfgs_b)
			return res['z']

	def GP_regression(self, X, y):
		x = np.atleast_2d(np.linspace(0, 10, 365)).T

		kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
		gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

		gp.fit(X, y)
		y_pred, sigma = gp.predict(x, return_std=True)
		return y_pred


	def get_model_scores(self):
		'''get smoothing score for each user  '''

		mydict = lambda: defaultdict(mydict)
		smoothing_dict = mydict()
		count = 0 
		for user, v in self.userTimeObj.items():
			if user is not None:
				#get smoothing scores, here mood becomes a continous concept 
				smoothing_dict[user]['smoothing'] = self.GP_regression(np.asarray(v['senti']))
				smoothing_dict[user]['time'] = v['postTime']
				#print(len(smoothing_dict[user]), len(v['senti']))
				count = count +1 
				if count == 3:
					break
		return smoothing_dict
		
class GuassianModel:
	'''create one gaussian model for the sample '''
	def __init__(self, sorted_sentiment):
		self.path = path
		self.sort_senti = sorted_sentiment
		



#read sentiment file
sp = SelectParticipants()
path = sp.path
participants = sp.process_participants()

#here you define the number of days you want to use as feature and the time window for mood
senti = SentiFeature(path = path, participants = participants)

sentiment_selected = senti.get_relative_day(365)
userTime = senti.get_user_time_obj(sentiment_selected)

sorted_senti = senti.SortTime(sentiment_selected)
Y1 = sorted_senti['sentiment_sum'].values.reshape(-1,1)
Y1 = Y1[1:100]
#X1 = sorted_senti['time']


X1= np.random.uniform(0, 1., (99, 1))
#plt.plot(X1, Y1, 'ok', markersize=10)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X1,Y1,kernel, noise_var=1e-10)

testX = np.linspace(0, 1, 100).reshape(-1, 1)

# g = GaussianSmoothing(userTimeObj = userTime)
# score = g.get_model_scores()

# Instantiate a Gaussian Process model

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X1, Y1)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(testX, return_std=True)

plt.plot(testX, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([testX, testX[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# for k, v in userTime.items():
# 	time = time.mktime(datetime.datetime.strptime(v, "%Y-%m-%d %H-%M-%S").timetuple())
# 	print(time)
#sorted_senti = senti.SortTime(sentiment_selected)

	