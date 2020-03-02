import pandas as pd 
import datetime
import collections 
import numpy as np
import statistics
#other paths
#path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
#path = '/Users/lucia/phd_work/predicting_depression_symptoms/data/'

'''for each user generate the mood (category) in the past X days (x is the time window) '''


class SelectParticipants():
    def __init__(self):
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/' 
        #self.path = '/Users/lucia/hawksworth/predicting_depression_symptoms/data/'
        self.participants = pd.read_csv(self.path + 'participants_matched.csv')
        #self.frequent_users = pd.read_csv(self.path + 'frequent_users.csv')
        #self.frequent_users = pd.read_csv(self.path + 'only2011_users.csv')
        self.frequent_users = pd.read_csv(self.path + 'adjusted_sample3.csv')

    def process_participants(self):
        '''process participant files, select useful columns '''
        self.frequent_users.columns = ['rowname', 'userid','freq']
        participants  = self.participants[['userid','time_completed','cesd_sum']]

        #frequent participants (commment this one if not needed)
        participants = pd.merge(self.frequent_users, participants, on='userid')
        participants.drop('rowname', axis=1, inplace=True)
        participants.drop('freq', axis=1, inplace=True)

        return participants


class MoodFeature:

    def __init__(self, path, participants):
        self.path = path
        self.sentiment_status = pd.read_csv(self.path + 'status_sentiment.csv')
        self.participants = participants        

    def sentiment_groups(self, series):
        '''recode sentiment to categorical '''
        if series > 0:
            return 2
        elif  series < 0:
            return 1
        elif  series == 0:
            return 4

    def map_sentiment(self, data):
        '''mapping sentiment score (positive: 2, negative:1, neutral:4)'''
        '''if sum positive, negaitve = 0 -> neutral   > 0 positive, < 0 negative '''
        data['sentiment_sum'] = data['positive'] + data['negative']
        #data['sentiment_sum'] = data['sentiment_sum'].apply(lambda x: self.sentiment_groups(x))
        return data

    
    def get_relative_day(self, adjusted_sentiment,time_frame):
        '''this function returns date the post is written relatively to the day user complete the cesd '''
        #merge data and 
        senti_part = pd.merge(self.participants, adjusted_sentiment, on='userid')
        senti_part['time_diff'] = pd.to_datetime(senti_part['time_completed']) - pd.to_datetime(senti_part['time'])
        #select posts before completed cesd
        senti_part['time_diff'] = senti_part['time_diff'].dt.days
        senti_sel = senti_part[(senti_part['time_diff'] >= 0) & (senti_part['time_diff'] < time_frame)]
        print('there are {} posts posted before user completed cesd and within {} days'.format(senti_sel.shape[0], time_frame))
        return senti_sel


    def SortTime(self,file):
        '''sort post by time'''
        file = file.sort_values(by=['userid','time_diff'],  ascending=True)
        file.to_csv(self.path +'check_time.csv')
        return file


    def user_obj(self, dfUserid, length):
        '''#create user vector of x(length) days'''
        users = {}
        for user in dfUserid:
           # print(user)
            if user not in users:
                users[user] = [np.nan]*length
        return users

    def getValenceVector(self, userObject, df):
        ''' parse mood condition to a day framework, return a dictionary of user: day1 mood, day2 mood...'''
        preDay = np.nan
        preUser = np.nan
        preValence = np.nan
        curdayPosts = []
        i = 0
        for valence, day, user in zip(df['sentiment_sum'], df['time_diff'], df['userid']):
            #rint(user, day)
            #posVal = 0
            if preUser is np.nan:
                curdayPosts = [valence]
            elif day == preDay and user == preUser:# and valence != preValence:
                curdayPosts.append(valence)
            else:
                dayvalence = self.getValenceFromMultiplePosts(curdayPosts)
                curdayPosts = [valence]
                userObject[user][int(day)-1] = dayvalence
            preDay = day
            preUser = user
            i +=1
        return userObject

    def getValenceFromMultiplePosts(self, curdayPosts):
        '''get mood'''
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
        if (neg_count !=0 and (neg_count > positive_count or neg_count > neu_count)):
            return -1
        if (positive_count !=0 and (positive_count > neg_count or positive_count > neu_count)):
            return 1
        elif (neu_count !=0 and (neu_count >= positive_count or neu_count >= neg_count)):
            return 0
        else: #silence days
            return np.nan

    def getAveragedValence(self, curdayPosts):
        '''get average mood of the day'''
        valence_sum = 0
        post_num = 1
        for post in curdayPosts:
            valence_sum = valence_sum + post
            post_num = post_num + 1

        valence_average = valence_sum/post_num
        return valence_average
        
    def getAveragedValenceVector(self, userObject, df):
        ''' parse mood condition to a day framework, return a dictionary of user: day1 mood, day2 mood...'''
        preDay = np.nan
        preUser = np.nan
        preValence = np.nan
        curdayPosts = []
        i = 0
        for valence, day, user in zip(df['sentiment_sum'], df['time_diff'], df['userid']):
            #rint(user, day)
            #posVal = 0
            if preUser is np.nan:
                curdayPosts = [valence]
            elif day == preDay and user == preUser:# and valence != preValence:
                curdayPosts.append(valence)
            else:
                dayvalence = self.getAveragedValence(curdayPosts)
                curdayPosts = [valence]
                userObject[user][int(day)-1] = dayvalence
            preDay = day
            preUser = user
            i +=1
        return userObject


    def get_mood_dict(self, timeRange, windowSzie, moodVector, step):
        '''construct mood temporal feature, in the past X day, the dominant mood is ? The window moves +1 day in each iteration'''
        #count = 0
        mood_vect_window_dict={}
        for k, v in moodVector.items():
            mood_vect_window  = []
            for start in range(0,timeRange, step): #define data range
                end = start+windowSzie #window size
                #print(start, end)
                mini_vect = v[start:end]
                #mood_vect_window = []
                freq = collections.Counter(mini_vect)
                #if silence days are more than 90% then mood is slience 
                if freq[np.nan] > len(mini_vect)*0.9:
                    mood_vect_window.append(np.nan)
                    #print(freq[-1])
                else: #otherwise remove silence and count the most frequent one 
                    lst = [i for i in mini_vect if i != np.nan]
                    freq2 = collections.Counter(lst).most_common(1)[0][0]
                    mood_vect_window.append(freq2)

            mood_vect_window_dict[k] = mood_vect_window 
        return mood_vect_window_dict

    def get_mood_vector(self, timeRange):
        '''return a daily mood vector in certain time range '''
        #map sentiment score 
        adjusted_sentiment = self.map_sentiment(self.sentiment_status)
        #get relative day
        sentiment_selected = self.get_relative_day(adjusted_sentiment, timeRange)
        #get mood score for each day , construct mood vector

        # sort posts according to time, in ascending order
        sorted_senti = self.SortTime(sentiment_selected)

        # get mood vector for X days
        users = self.user_obj(sorted_senti['userid'], timeRange)
        moodVector = self.getValenceVector(users, sorted_senti)
        return moodVector

    def get_mood_continous(self, timeRange):
        '''return a daily averaged mood  in certain time range '''
        #get sentiment sum
        adjusted_sentiment = self.map_sentiment(self.sentiment_status)
        #get relative day
        sentiment_selected = self.get_relative_day(adjusted_sentiment, timeRange)
        #get mood score for each day , construct mood vector

        # sort posts according to time
        sorted_senti = self.SortTime(sentiment_selected)

        # get mood vector for X days
        users = self.user_obj(sorted_senti['userid'], timeRange)
        moodVector = self.getAveragedValenceVector(users, sorted_senti)
        return moodVector

    def get_mood_change_dict(self, timeRange, windowSzie, moodVector, step):
        '''construct mood temporal feature, how much does the mood changes every X day? X is the time window'''
        #count = 0
        mood_vect_window_dict= {}
        for k, v in moodVector.items():
            mood_vect_window  = []
            preWin_mean = 0 
             
            for start in range(0,timeRange, step): #define data range
                end = start+windowSzie #window size
                #print(start, end)
                mini_vect = v[start:end]
                #mood_vect_window = []
                win_mean = np.nanmean(mini_vect)
                
                #get the change between two windows
                mood_vect_window.append(win_mean - preWin_mean)
                #print(win_mean)
                preWin_mean = win_mean

            mood_vect_window_dict[k] = mood_vect_window 
        return mood_vect_window_dict

    def get_mood_continous_dict(self, timeRange, windowSzie, moodVector, step):
        '''construct mood temporal feature, how much does the mood changes every X day? X is the time window'''
        #count = 0
        mood_vect_window_dict= {}
        for k, v in moodVector.items():
            mood_vect_window  = []
            preWin_mean = 0 
             
            for start in range(0,timeRange, step): #define data range
                end = start+windowSzie #window size
                #print(start, end)
                mini_vect = v[start:end]
                #mood_vect_window = []
                win_mean = np.nanmean(mini_vect)
                #get the change between two windows
                mood_vect_window.append(win_mean)

            mood_vect_window_dict[k] = mood_vect_window 
        return mood_vect_window_dict

    def get_mood_in_timewindow(self, timeRange, windowSzie, step):
        # get mood vector with 
        moodVector = self.get_mood_vector(timeRange)
        mood_dict = self.get_mood_dict(timeRange, windowSzie, moodVector, step) #paramenter: number of days used as features, time window
        mood_vect_df = pd.DataFrame.from_dict(mood_dict).T
        #change column names

        #mood_vect_df.columns = [str(col) + '_mood' for col in mood_vect_df.columns]
        #mood_vect_df.rename(columns={'0_mood':'userid'}, inplace=True)
        #mood_vect_df['userid'] = mood_vect_df.index

        mood_vect_df.to_csv(self.path + './mood_vectors/mood_vector_frequent_user_window_{}_timeRange{}_step{}.csv'.format(windowSzie, timeRange, step)) #feature matrx for prediction 
        return mood_vect_df, windowSzie

    def get_mood_change_in_timewindow(self, timeRange, windowSzie, step):
        # get mood vector with 
        moodVector = self.get_mood_continous(timeRange)
        mood_dict = self.get_mood_change_dict(timeRange, windowSzie, moodVector, step) #paramenter: number of days used as features, time window
        mood_vect_df = pd.DataFrame.from_dict(mood_dict).T

        mood_vect_df.to_csv(self.path + './mood_vectors/mood_change_frequent_user_window_{}_timeRange{}_step{}.csv'.format(windowSzie, timeRange, step)) #feature matrx for prediction 
        return mood_vect_df, windowSzie

    def get_mood_continous_in_timewindow(self, timeRange, windowSzie, step):
        # get mood vector with 
        moodVector = self.get_mood_continous(timeRange)
        mood_dict = self.get_mood_continous_dict(timeRange, windowSzie, moodVector, step) #paramenter: number of days used as features, time window
        mood_vect_con_df = pd.DataFrame.from_dict(mood_dict).T

        mood_vect_con_df.to_csv(self.path + './mood_vectors/mood_change_continous_user_window_{}_timeRange{}_step{}.csv'.format(windowSzie, timeRange, step)) #feature matrx for prediction 
        return mood_vect_con_df, windowSzie

#read sentiment file
sp = SelectParticipants()
path = sp.path
participants = sp.process_participants()

#here you define the number of days you want to use as feature and the time window for mood
mood = MoodFeature(path = path, participants = participants)
#get mood feature in time window (category)
# mood_vector_feature, windowSzie = mood.get_mood_in_timewindow(365, 30, 3)

# #get mood feature in time window (continues)
# mood_vector_con_feature, windowSzie = mood.get_mood_continous_in_timewindow(365, 30, 3)

# mood_vector_feature, windowSzie = mood.get_mood_change_in_timewindow(365, 30, 3)


