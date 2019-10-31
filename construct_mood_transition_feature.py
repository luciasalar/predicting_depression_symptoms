from construct_mood_feature import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_transitions_count(ValenceObject):
	#count number of transition
    emptyTran = 0
    negaTran = 0
    posiTran = 0
    neuTran = 0
    PosAndNeg = 0  
    NeuAndPos = 0
    NeuAndNeg = 0
    EmptyAndPos = 0
    EmptyAndNeg = 0
    EmptyAndNeu = 0
    preValence = 0
    for valence in ValenceObject:
    #these are self transition states
        if valence == 1 and preValence == 1:
            negaTran = negaTran + 1
        elif valence == 2 and preValence == 2:
            posiTran = posiTran + 1
        elif valence == 0 and preValence == 0:
            neuTran = neuTran + 1
    #positive and negative transition:
        elif (valence == 1 and preValence == 2) or (valence == 2 and preValence == 1) :
            PosAndNeg = PosAndNeg + 1
    #neutral and postive transition
        elif (valence == 0 and preValence == 2) or (valence == 2 and preValence == 0) :
            NeuAndPos = NeuAndPos + 1
    #neutral and negative transition 
        elif (valence == 0 and preValence == 1) or (valence == 1 and preValence == 0) :
            NeuAndNeg = NeuAndNeg + 1
    #Empty and positive transition
        elif (valence == -1 and preValence == 2) or (valence == 2 and preValence == -1) :
            EmptyAndPos = EmptyAndPos + 1
    
    #Empty and negative transition
        elif (valence == -1 and preValence == 1) or (valence == 1 and preValence == -1) :
            EmptyAndNeg = EmptyAndNeg + 1
    
    #Empty and neutral transition
        elif (valence == 0 and preValence == -1) or (valence == -1 and preValence == 0) :
            EmptyAndNeu = EmptyAndNeu + 1
            
        preValence = valence
    return [emptyTran, negaTran, posiTran, neuTran, PosAndNeg, NeuAndPos, NeuAndNeg, EmptyAndPos, EmptyAndNeg, EmptyAndNeu]


def get_transitions(ValenceObject):
	'''create a list that shows the transition states postive - negative:3, neu - positive: 4, neu - positive: 5'''
	#count number of transition
	transitions =[]
	preValence = 0
	for valence in ValenceObject:
	#these are self transition states
	    if valence == 1 and preValence == 1:
	        transitions.append(1)
	    elif valence == 2 and preValence == 2:
	        transitions.append(2)
	    elif valence == 0 and preValence == 0:
	        transitions.append(0)
	#positive and negative transition:
	    elif (valence == 1 and preValence == 2) or (valence == 2 and preValence == 1) :
	        transitions.append(3)
	#neutral and postive transition
	    elif (valence == 0 and preValence == 2) or (valence == 2 and preValence == 0) :
	        transitions.append(4)
	#neutral and negative transition 
	    elif (valence == 0 and preValence == 1) or (valence == 1 and preValence == 0) :
	        transitions.append(5)
	#Empty and positive transition
	    elif (valence == -1 and preValence == 2) or (valence == 2 and preValence == -1) :
	        transitions.append(6)

	#Empty and negative transition
	    elif (valence == -1 and preValence == 1) or (valence == 1 and preValence == -1) :
	        transitions.append(7)

	#Empty and neutral transition
	    elif (valence == 0 and preValence == -1) or (valence == -1 and preValence == 0) :
	        transitions.append(8)

	#Empty and empty
	    elif valence == -1 and preValence == -1 :
	        transitions.append(-1)
	        
	    preValence = valence
	return transitions


def get_mood_transitions(valencVec):
    result = {}
    for item in valencVec:
        result[item] = get_transitions(valencVec[item])
#         print(result)
    return result


def get_mood_transitions(valencVec):
	result = {}
	for index, row in valencVec.iterrows():
	    result[index] = get_transitions(row)
	    #print(row)
	return result


TransitionStates = get_mood_transitions(mood_vector_feature) 

#

def get_transitions_df(path, windowSzie):
	TransitionStates = get_mood_transitions(mood_vector_feature)  
	TransitionStates_df = pd.DataFrame.from_dict(TransitionStates).T
	TransitionStates_df.to_csv(path + './mood_vectors/mood_transition_frequent_user_window_{}.csv'.format(windowSzie)) #feature matrx for prediction 
	return TransitionStates_df

TransitionStates = get_transitions_df(path, windowSzie)


onehoc_X = OneHotEncoder(handle_unknown='ignore')
TransitionStates[:,1:] = onehoc_X.fit_transform(TransitionStates.iloc[:,1:]).toarray()










