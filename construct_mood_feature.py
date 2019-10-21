import pandas as pd 
import datetime

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
	file = file.sort_values(by=['userid','time_diff'],  ascending=False)
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

	#     compute mood
	if (neg_count !=0 and (neg_count > positive_count or neg_count > neu_count )):
	    return 1
	if (positive_count !=0 and (positive_count > neg_count or positive_count > neu_count)):
	    return 2
	elif (neu_count !=0 and (neu_count >= positive_count or neu_count >= neg_count)):
	    return 0
	else: #silence days
	    return -1


#read sentiment file
path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
sentiment_status = pd.read_csv(path + 'status_sentiment.csv')
participants = pd.read_csv(path + 'participants_matched.csv')
participants  = participants[['userid','time_completed','cesd_sum']]

#map sentiment score 
adjusted_sentiment = map_sentiment(sentiment_status)
#get relative day
sentiment_selected = get_relative_day(adjusted_sentiment, 1095)
#get mood score for each day , construct mood vector

# sort posts according to time
sorted_senti = SortTime(sentiment_selected)

# get mood vector for X days
users = user_obj(sorted_senti['userid'], 1095)
moodVector = getValenceVector(users, sorted_senti)

#construct mood temporal feature, in the past X day, the dominant mood is , X increase in each loop. If we have 100 days, 
#the length of this feature is X/100
#problem silence will always be the most frequent
import collections 

count = 0
mood_vect_window_dict={}
for k, v in moodVector.items():
	window = 10
	mini_vect = v[0:window]
	mood_vect_window = []
	freq = collections.Counter(mini_vect)
	#if silence day more than half 50% then mood is slience 
	if freq[-1] > window/2:
		mood_vect_window.append(-1)
	else: #otherwise remove silence and count the most frequent one 
		mini_vect.remove(-1)
		print(mini_vect.remove(-1))
		#freq2 = collections.Counter(no_silence).most_common(1)[0][0]
		#mood_vect_window.append(freq2)

		print(mini_vect)
	mood_vect_window_dict[k] = mood_vect_window 
	count+= 1
	if count == 10:
		break

# print('save objects')
# savePath = path + '/newScripts/moodVector/moodVectorsData/MoodVecDes1.pickle'
# with open(savePath, 'wb') as handle:
#     pickle.dump(users2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# savePath2 = path + '/newScripts/moodVector/moodVectorsData/MoodVec.csv'
# saveCSV(users2, savePath2)

# print('compute transition states probability')

# TransitionStates = getUserTransitions(users2)  



