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
		more_than_10 = pd.read_csv(self.path + 'more_than_10.csv')
		# merge with users more than 10 posts
		#merge with frequent users
		# frequent = pd.read_csv(self.path + 'frequent_users.csv')
		# frequent.columns = ['num','userid', 'freq']
		# merge with cesd data
		clean = pd.merge(more_than_10, participants, on = 'userid')
		data = pd.merge(result, clean, on = 'userid')
		#data = pd.merge(data, frequent, on = 'userid')
		return data

	def recode(self, y, threshold1, threshold2):
		'''recode y to binary according to a threshold'''
		new_labels = []
		for i in y:
			if i <= threshold1:
				i = 0
			if i > threshold1 and i <= threshold2:
				i = 1
			if i > threshold2:
				i = 2
			new_labels.append(i)
		return new_labels

	def recode_cesd(self):
		'''recode y '''

		data = s.read_results()
		data['cesd_sum'] = self.recode(data['cesd_sum'], 16, 22) 
		data.to_csv(self.path + 'temp.csv')
		
		return data


	def averaged_lengthscale(self):
		###get averaged score of length scale in different groups
		high = data.loc[data['cesd_sum'] == 2]
		moderate = data.loc[data['cesd_sum'] == 1]
		low = data.loc[data['cesd_sum'] == 0]
		#get t test
		t_test1 = stats.mannwhitneyu(high['lengthscale'].values, low['lengthscale'].values)
		t_test2 = stats.mannwhitneyu(moderate['lengthscale'].values, low['lengthscale'].values)
		t_test3 = stats.mannwhitneyu(moderate['lengthscale'].values, high['lengthscale'].values)

		return high['lengthscale'].median(), low['lengthscale'].median(), moderate['lengthscale'].median(), t_test1, t_test2, t_test3

	def averaged_variance(self):
		###get averaged score of length scale in different groups
		high = data.loc[data['cesd_sum'] == 2]
		moderate = data.loc[data['cesd_sum'] == 1]
		low = data.loc[data['cesd_sum'] == 0]
		#get t-test
		t_test1 = stats.mannwhitneyu(high['variance'].values, low['variance'].values)
		t_test2 = stats.mannwhitneyu(moderate['variance'].values, low['variance'].values)
		t_test3 = stats.mannwhitneyu(moderate['variance'].values, high['variance'].values)

		return high['variance'].median(), low['variance'].median(), moderate['variance'].median(), t_test1, t_test2, t_test3




s = GPstats()
data = s.recode_cesd()
high_m, low_m, moderate_n, p1, p2, p3 = s.averaged_lengthscale()
high_v, low_v, moderate_v, t1, t2, t3 = s.averaged_variance()