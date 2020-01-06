import pandas as pd 
from scipy import stats

class GPstats:
	def __init__(self):
		self.path = path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/' 

	def read_results(self):
		''' read result file '''
		result = pd.read_csv(self.path + 'results/GPresults_14_3.csv')
		participants = pd.read_csv(self.path + 'participants_matched.csv')
		participants = participants[['userid','cesd_sum']]
		#merge with frequent users
		# frequent = pd.read_csv(self.path + 'frequent_users.csv')
		# frequent.columns = ['num','userid', 'freq']
		# merge with cesd data
		data = pd.merge(result, participants, on = 'userid')
		#data = pd.merge(data, frequent, on = 'userid')
		return data

	def recode(self, y, threshold):
		'''recode y to binary according to a threshold'''
		new_labels = []
		for i in y:
			if i <= threshold:
				i = 0
			if i > threshold:
				i = 1
			new_labels.append(i)
		return new_labels

	def recode_cesd(self):
		'''recode y '''
		
		data = s.read_results()
		data['cesd_sum'] = self.recode(data['cesd_sum'], 16) 
		data.to_csv(self.path + 'temp.csv')
		
		return data


	def averaged_lengthscale(self):
		###get averaged score of length scale in different groups
		high = data.loc[data['cesd_sum'] == 1]
		low = data.loc[data['cesd_sum'] == 0]
		#get t test
		t_test = stats.mannwhitneyu(high['lengthscale'].values, low['lengthscale'].values)

		return high['lengthscale'].median(), low['lengthscale'].median(), t_test

	def averaged_variance(self):
		###get averaged score of length scale in different groups
		high = data.loc[data['cesd_sum'] == 1]
		low = data.loc[data['cesd_sum'] == 0]
		#get t-test
		t_test = stats.mannwhitneyu(high['variance'].values, low['variance'].values)

		return high['variance'].median(), low['variance'].median(), t_test




s = GPstats()
data = s.recode_cesd()
high_m, low_m, p1 = s.averaged_lengthscale()
high_v, low_v, p2 = s.averaged_variance()