import pandas as pd 
import datetime
import collections 
import numpy as np

def map_sentiment(data):
	#mapping sentiment score (positive: 2, negative:1, neutral:4)
	'''if sum positive, negaitve = 0 -> neutral   > 0 positive, < 0 negative '''
	data['sentiment_sum'] = data['positive'] + data['negative']
	data['class'] = data['sentiment_sum'].apply(lambda x: sentiment_groups(x))
	return data

def get_relative_day(adjusted_sentiment,time_frame):
	'''this function returns date the post is written relatively to the day user complete the cesd '''
	#merge data and 
	senti_part = pd.merge(participants, adjusted_sentiment, on = 'userid')
	senti_part['time_diff'] = pd.to_datetime(senti_part['time_completed']) - pd.to_datetime(senti_part['time'])
	#select posts before completed cesd
	senti_part['time_diff'] = senti_part['time_diff'].dt.days
	senti_sel = senti_part[(senti_part['time_diff'] >= 0) & (senti_part['time_diff'] < time_frame)]
	print('there are {} posts posted before user completed cesd and within {} days'.format(senti_sel.shape[0], time_frame))
	return senti_sel


def sentiment_groups(series):
	'''recode sentiment to categorical '''
	if series > 0:
	    return 2
	elif  series < 0:
	    return 1
	elif  series == 0:
	    return 4


def SortTime(file):
	'''sort post by time'''
	file = file.sort_values(by=['userid','time_diff'],  ascending=True)
	return file


def user_obj(dfUserid, length):
	'''#create user vector of x(length) days'''
	users = {}
	for user in dfUserid:
	   # print(user)
	    if user not in users:
	        users[user] = [-1]*length
	return users

def getValenceVector(userObject, df):
	''' parse mood condition to a day framework, return a dictionary of user: day1 mood, day2 mood...'''
	preDay = None
	preUser = None
	preValence = None
	curdayPosts = []
	i = 0
	for valence, day, user in zip(df['sentiment_sum'], df['time_diff'], df['userid']):
	    #rint(user, day)
	    #posVal = 0
	    if preUser is None:
	        curdayPosts = [valence]
	    elif day == preDay and user == preUser:# and valence != preValence:
	        curdayPosts.append(valence)
	    else:
	        dayvalence = getValenceFromMultipleDays(curdayPosts)
	        curdayPosts = [valence]
	        userObject[user][int(day)-1] = dayvalence
	    preDay = day
	    preUser = user
	    i +=1
	return userObject

def getValenceFromMultipleDays(curdayPosts):
	'''get mood of the day'''
	positive_count = 0
	neg_count = 0
	neu_count = 0
	for post in curdayPosts:
	    if post < 0:
	        neg_count += 1
	    elif post == 0:
	        neu_count += 1
	    else:
	        positive_count +=1

	#  define mood conditions
	if (neg_count !=0 and (neg_count > positive_count or neg_count > neu_count )):
	    return 1
	if (positive_count !=0 and (positive_count > neg_count or positive_count > neu_count)):
	    return 2
	elif (neu_count !=0 and (neu_count >= positive_count or neu_count >= neg_count)):
	    return 0
	else: #silence days
	    return -1


def get_mood_dict(timeRange, windowSzie, moodVector):
	'''construct mood temporal feature, in the past X day, the dominant mood is ? The window moves +1 day in each iteration'''
	#count = 0
	mood_vect_window_dict={}
	for k, v in moodVector.items():
		mood_vect_window  = []
		for start in range(0,timeRange): #define data range
			end = start+windowSzie #window size
			#print(start, end)
			mini_vect = v[start:end]
			#mood_vect_window = []
			freq = collections.Counter(mini_vect)
			#if silence days are more than 50% then mood is slience 
			if freq[-1] > len(mini_vect)*0.6:
				mood_vect_window.append(-1)
				#print(freq[-1])
			else: #otherwise remove silence and count the most frequent one 
				lst = [i for i in mini_vect if i != -1]
				freq2 = collections.Counter(lst).most_common(1)[0][0]
				mood_vect_window.append(freq2)

		mood_vect_window_dict[k] = mood_vect_window 
	return mood_vect_window_dict

def get_mood_vector(timeRange, sentiment_status):
	'''return a daily mood vector in certain time range '''
	#map sentiment score 
	adjusted_sentiment = map_sentiment(sentiment_status)
	#get relative day
	sentiment_selected = get_relative_day(adjusted_sentiment, timeRange)
	#get mood score for each day , construct mood vector

	# sort posts according to time
	sorted_senti = SortTime(sentiment_selected)

	# get mood vector for X days
	users = user_obj(sorted_senti['userid'], timeRange)
	moodVector = getValenceVector(users, sorted_senti)
	return moodVector

def get_mood_in_timewindow(timeRange, windowSzie):
	# get mood vector with 
	moodVector = get_mood_vector(1095, sentiment_status)
	mood_dict = get_mood_dict(timeRange, windowSzie, moodVector) #paramenter: number of days used as features, time window
	mood_vect_df = pd.DataFrame.from_dict(mood_dict).T
	#change column names

	mood_vect_df.columns = [str(col) + '_mood' for col in mood_vect_df.columns]
	#mood_vect_df.rename(columns={'0_mood':'userid'}, inplace=True)
	#mood_vect_df['userid'] = mood_vect_df.index
	mood_vect_df.to_csv(path + './mood_vectors/mood_vector_frequent_user_window_{}.csv'.format(windowSzie)) #feature matrx for prediction 
	return mood_vect_df, windowSzie


#read sentiment file
path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
#path = '/Users/lucia/phd_work/predicting_depression_symptoms/data/'
sentiment_status = pd.read_csv(path + 'status_sentiment.csv')
participants = pd.read_csv(path + 'participants_matched.csv')
frequent_users = pd.read_csv(path + 'frequent_users.csv')
frequent_users.columns = ['rowname', 'userid','freq']
participants  = participants[['userid','time_completed','cesd_sum']]
#frequent participants (commment this one if not needed)
participants = pd.merge(frequent_users, participants, on='userid')
participants.drop('rowname', axis=1, inplace=True)
participants.drop('freq', axis=1, inplace=True)


mood_vector_feature, windowSzie = get_mood_in_timewindow(365, 3)


