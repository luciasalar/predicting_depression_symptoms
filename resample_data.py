import pandas as pd 
import random

''' here we resample the data so that high symptom would be 1/3 of low symptom'''
'''don't select frequent users because we don't have enough sample for low symptom class'''

class Resample:
	def __init__(self):
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

	def selection(self):
		'''select users so that the proportion of high symptom and low symptom is 1:3'''
		all_users = pd.read_csv(self.path + 'participants_matched.csv')
		#frequent users
		# frequent_users = pd.read_csv(self.path + 'frequent_users.csv')
		# frequent_users.columns =['rn','userid','freq']
		
		cesd = all_users[['userid','cesd_sum']]
		#cesd_freq = pd.merge(frequent_users, cesd, on ='userid')


		#convert labels
		high = all_users.loc[all_users['cesd_sum'] >=22, ] 
		low = all_users.loc[all_users['cesd_sum'] < 22, ] 

		random.seed(333)
		high = high.sample(n=294) #randomly select 100 cases
		selected = pd.concat([high, low])
		print(high.shape, low.shape)

		selected = selected[['userid','q10']]
		selected.to_csv(self.path + 'adjusted_sample2.csv')

		return selected

r = Resample()
selected = r.selection()












































