import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
import datetime
#from datetime import datetime
import pickle


def SortTime(file):
	#select date
	file['time'] = file['time'].apply(lambda x: x.split()[0])
	#convert to time series
	file['time'] = file['time'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
	#sort values according to userid, time and valence
	file = file.sort_values(by=['userid','time_diff','negative_ny'],  ascending=False)
	return file

#create user vector
def user_obj(dfUserid):
    users = {}
    for user in dfUserid:
       # print(user)
        if user not in users:
            users[user] = [-1]*60 
    return users

#get the which valence is dominate in one day
def getValenceFromMultipleDays(curdayPosts):
    positive_count = 0
    neg_count = 0
    neu_count = 0
    mix_count = 0
    for post in curdayPosts:
        if post == 1:
            neg_count += 1
        elif post == 3:
            mix_count += 1
        elif post == 4:
            neu_count += 1
        else:
            positive_count +=1
#     print(mix_count < positive_count or mix_count < neg_count)
#     print('neg_count{}, pos_count{}, neu_count{}, mix_count{}'. format(neg_count,positive_count,neu_count,mix_count))
    if (neg_count !=0 and (neg_count > positive_count or neg_count > neu_count or neg_count > mix_count )):
        return 1
    #give more weights to mix
    if (mix_count !=0 and (mix_count >= positive_count or mix_count >= neg_count or neg_count == positive_count)):
        return 3
    #here we give positive emotions more weights
    if (positive_count !=0 and (positive_count > neg_count or positive_count > neu_count or positive_count > mix_count)):
        return 2
    elif (neu_count !=0 and (neu_count >= positive_count or neu_count >= neg_count or neu_count > mix_count)):
        return 4
    else:
        return -1

def getValenceVector(userObject, df):
    preDay = None
    preUser = None
    preValence = None
    curdayPosts = []
    i = 0
    for valence, day, user in zip(df['negative_ny'], df['time_diff'], df['userid']):
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

def saveCSV(userObj,file):
    data = pd.DataFrame.from_dict(userObj)
    data = data.T
    data.to_csv(file)

def getTransitions(ValenceObject):
    emptyTran = 0
    negaTran = 0
    posiTran = 0
    mixTran = 0
    neuTran = 0
    PosAndNeg = 0
    MixAndPos = 0
    MixAndNeg = 0
    MixAndNeu = 0    
    NeuAndPos = 0
    NeuAndNeg = 0
    EmptyAndPos = 0
    EmptyAndNeg = 0
    EmptyAndMix = 0
    EmptyAndNeu = 0
    preValence = 0
    for valence in ValenceObject:
    #these are self transition states
        if valence == -1 and preValence == -1:
            emptyTran = emptyTran + 1
        elif valence == 1 and preValence == 1:
            negaTran = negaTran + 1
        elif valence == 2 and preValence == 2:
            posiTran = posiTran + 1
        elif valence == 3 and preValence == 3:
            mixTran = mixTran + 1
        elif valence == 4 and preValence == 4:
            neuTran = neuTran + 1
    #positive and negative transition:
        if (valence == 1 and preValence == 2) or (valence == 2 and preValence == 1) :
            PosAndNeg = PosAndNeg + 1
    #mix and positive transition
        if (valence == 3 and preValence == 2) or (valence == 2 and preValence == 3) :
            MixAndPos = MixAndPos + 1
    #mix and negative transition
        if (valence == 3 and preValence == 1) or (valence == 1 and preValence == 3) :
            MixAndNeg = MixAndNeg + 1
    #mix and neutral transition
        if (valence == 3 and preValence == 4) or (valence == 4 and preValence == 3) :
            MixAndNeu = MixAndNeu + 1
    #neutral and postive transition
        if (valence == 4 and preValence == 2) or (valence == 2 and preValence == 4) :
            NeuAndPos = NeuAndPos + 1
    #neutral and negative transition 
        if (valence == 4 and preValence == 1) or (valence == 1 and preValence == 4) :
            NeuAndNeg = NeuAndNeg + 1
    #Empty and positive transition
        if (valence == -1 and preValence == 2) or (valence == 2 and preValence == -1) :
            EmptyAndPos = EmptyAndPos + 1
    
    #Empty and negative transition
        if (valence == -1 and preValence == 1) or (valence == 1 and preValence == -1) :
            EmptyAndNeg = EmptyAndNeg + 1
    
    #Empty and mix transition
        if (valence == -1 and preValence == 3) or (valence == 3 and preValence == -1) :
            EmptyAndMix = EmptyAndMix + 1
    
    #Empty and neutral transition
        if (valence == 4 and preValence == -1) or (valence == -1 and preValence == 4) :
            EmptyAndNeu = EmptyAndNeu + 1
            
            
        preValence = valence
    return [emptyTran, negaTran, posiTran, mixTran, neuTran, PosAndNeg, MixAndPos, MixAndNeg, MixAndNeu, NeuAndPos, NeuAndNeg, EmptyAndPos, EmptyAndNeg, EmptyAndMix, EmptyAndNeu]

def getUserTransitions(valencVec):
    result = {}
    for item in valencVec:
        result[item] = getTransitions(valencVec[item])
#         print(result)
    return result

#compute transition states
def computeTrans(savePath2):
	file = pd.read_csv(savePath2)
	file.columns = ['userid','emptyTran', 'negaTran', 'posiTran', 'mixTran', 'neuTran', 'PosAndNeg', 'MixAndPos', 'MixAndNeg', 'MixAndNeu', 'NeuAndPos', 'NeuAndNeg', 'EmptyAndPos', 'EmptyAndNeg', 'EmptyAndMix', 'EmptyAndNeu']
	file['allPosts'] = file.sum(axis=1) 

	#we compute the pobability by dividing the transition with days: 59
	file.index = file['userid']
	file = file.drop(['userid'], axis=1)
	Tranprob = file.apply(lambda x: x/59)
	Tranprob.to_csv(savePath2)
	return Tranprob

#here we get correlation matrix of valenceVec and feature variables
def getCorMatrix(savePath, savePath2, alldata, transitionMatrix):
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	transitionMatrix['userid'] = transitionMatrix.index
	compare = pd.merge(transitionMatrix, Var, on ='userid', how = 'left')
	corMatrix = compare.corr()
	corMatrix.to_csv(savePath)
	compare.to_csv(savePath2)

#here we get frequency matrix for the valence vector
def getFrequencyCor(valenceVec,savePath1,savePath2,alldata):
	negativeD = []
	positiveD = []
	neutralD = []
	mixed = []
	empty = []
	useridL = []
	for userid in valenceVec:
	    negativeD.append(valenceVec[userid].count(1))
	    positiveD.append(valenceVec[userid].count(2))
	    neutralD.append(valenceVec[userid].count(4))
	    mixed.append(valenceVec[userid].count(3))
	    empty.append(valenceVec[userid].count(0))
	    useridL.append(userid)
	#merge all the lists to data frame
	df = pd.DataFrame(np.array(negativeD).reshape(len(negativeD),1), columns=['NegativePosts'])
	df['PositivePosts'] = positiveD
	df['NeutralPosts'] = neutralD
	df['MixedPosts'] = mixed
	df['EmptyPosts'] = empty
	df['userid'] = useridL
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	compareFreq = pd.merge(df, Var, on ='userid', how = 'left')
	compareFreq.to_csv(savePath1)
	corMatrix = compareFreq.corr()
	corMatrix.to_csv(savePath2)

if __name__ == '__main__':
    path = '/Users/lucia/phd_work/cognitive_distortion'
    #this file contain users with 80% of posts retained after cleaning foreign language
    time = pd.read_csv(path + '/data/important_data/cleanLabelsReverse.csv')
    ids = pd.read_csv(path + '/data/important_data/FinalSampleUsers.csv')
    #remove underage and ppl with less than 80% of posts retained
    time  = time[time['userid'].isin(ids['userid'])]
    # sort posts according to time
    time = SortTime(time)

    print('get valence vector')
    users = user_obj(time['userid'])
    users2 = getValenceVector(users, time)


    print('save objects')
    savePath = path + '/newScripts/moodVector/moodVectorsData/MoodVecDes1.pickle'
    with open(savePath, 'wb') as handle:
        pickle.dump(users2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    savePath2 = path + '/newScripts/moodVector/moodVectorsData/MoodVec.csv'
    saveCSV(users2, savePath2)

    print('compute transition states probability')

    TransitionStates = getUserTransitions(users2)  

    print('save transition state objects')
    savePath = path + '/newScripts/moodVector/moodVectorsData/MoodTrans.pickle'
    with open(savePath, 'wb') as handle:
        pickle.dump(TransitionStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

    savePath2 = path + '/newScripts/moodVector/moodVectorsData/MoodTrans.csv'
    saveCSV(TransitionStates, savePath2) 

    Tranprob = computeTrans(savePath2) 
    print('get correlation corMatrix')
    savePath3 = path + '/newScripts/moodVector/moodVectorsData/MoodVecCor.csv'
    savePath4 = path + '/newScripts/moodVector/moodVectorsData/MoodVecAllVar.csv'
    allData = pd.read_csv( path + '/data/important_data/user_scale_post_time2.csv')
    getCorMatrix(savePath3, savePath4, allData, Tranprob)









