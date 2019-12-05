from construct_mood_feature import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder





class TransitionMatrix:
    def __init__(self, ValenceObject):
        '''ValenceObject is a dictionary, key as userid, value as a vector to represent mood each day '''
        self.ValenceObject = ValenceObject
        self.windowSzie = 7 #window size of the transition state

    def get_transitions_count(self):
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
        for valence in self.ValenceObject:
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


    def get_transitions(self, valenceVec):
        '''create a list that shows the transition states postive - negative:3, neu - positive: 4, neu - positive: 5'''
        #count number of transition
        transitions =[]
        preValence = 0
        for valence in valenceVec:
    	#these are self transition states
            if valence == -1 and preValence == -1:
                transitions.append(1)
            elif valence == 1 and preValence == 1:
                transitions.append(2)
            elif valence == 0 and preValence == 0:
                transitions.append(0)
            #positive to negative transition:
            elif valence == 1 and preValence == -1:
                transitions.append(3)
            #negative to positive transition:
            elif valence == -1 and preValence == 1 :
                transitions.append(4)
            #neutral to postive transition
            elif valence == 0 and preValence == 1 :
                transitions.append(5)
            #positive to neutral transition
            elif valence == 1 and preValence == 0:
                transitions.append(6)
            #neutral to negative transition 
            elif valence == 0 and preValence == -1:
                transitions.append(7)
            #negative to neutral transition 
            elif (valence == -1 and preValence == 0) :
                transitions.append(8)
            #Empty to positive transition
            elif (valence == -1 and preValence == 1):
                transitions.append(9)
            #positive to empty transition
            elif (valence == 1 and preValence == -1) :
                transitions.append(10)
            #Empty to negative transition
            elif (valence == None and preValence == -1):
                transitions.append(11)
            # negative to empty transition
            elif (valence == -1 and preValence == None) :
                transitions.append(12)
            #Empty to neutral transition
            elif (valence == None and preValence == 0) :
                transitions.append(13)
            #neutral to empty transition
            elif (valence == 0 and preValence == None):
                transitions.append(14)
            #Empty and empty
            elif valence == None and preValence == None :
                transitions.append(15)
                
            preValence = valence
        return transitions


    def get_mood_transitions(self):
        '''get transtions of mood dict'''
        result = {}
        for k, v in self.ValenceObject.items():
            result[k] = self.get_transitions(v)
        return result


    def transitionPro(self, transitionSlide):
        ''' get trainsition probabililty'''
        # t = self.get_mood_transitions()
        transition_dict = {}
        #transition states  
        posTOPos = 0 
        PosTNeg = 0
        PosTNeu = 0 
        PosTSil = 0

        negaToNega = 0
        NegTPos = 0
        NegTNeu = 0
        NegTSil = 0

        neuTNeu = 0
        NeuTPos = 0
        NeuTNeg = 0
        NeuSil = 0
        

        silToSil = 0
        silTNeg = 0
        silTNeu = 0
        silTPos = 0

        probabilities = []
        for transition in transitionSlide:
            
            if transition == 2:
                posTOPos = posTOPos + 1
            if transition == 3:
                PosTNeg = PosTNeg + 1
            if transition == 6:
                PosTNeu = PosTNeu + 1
            if transition == 10:
                PosTSil = PosTSil+ 1

            if transition == 1:
                negaToNega = negaToNega + 1
            if transition == 4:
                NegTPos = NegTPos + 1
            if transition == 8:
                NegTNeu = NegTNeu+ 1
            if transition == 12:
                NegTSil = NegTSil + 1

            if transition == 0:
                neuTNeu = neuTNeu + 1
            if transition == 5:
                NeuTPos = NeuTPos + 1
            if transition == 7:
                NeuTNeg = NeuTNeg + 1
            if transition == 14:
                NeuSil = NeuSil + 1

            if transition == 15:
                silToSil = silToSil + 1
            if transition == 11:
                silTNeg = silTNeg + 1
            if transition == 13:
                silTNeu = silTNeu + 1
            if transition == 9:
                silTPos = silTPos + 1
           

        #get probabilities
        probabilities.append(posTOPos/len(transitionSlide))
        probabilities.append(PosTNeg/len(transitionSlide))
        probabilities.append(PosTNeu/len(transitionSlide))
        probabilities.append(PosTSil/len(transitionSlide))

        probabilities.append(negaToNega/len(transitionSlide))
        probabilities.append(NegTPos/len(transitionSlide))
        probabilities.append(NegTNeu/len(transitionSlide))
        probabilities.append(NegTSil/len(transitionSlide))

        probabilities.append(neuTNeu/len(transitionSlide))
        probabilities.append(NeuTPos/len(transitionSlide))
        probabilities.append(NeuTNeg/len(transitionSlide))
        probabilities.append(NeuSil/len(transitionSlide))

        probabilities.append(silToSil/len(transitionSlide))
        probabilities.append(silTNeg/len(transitionSlide))
        probabilities.append(silTNeu/len(transitionSlide))
        probabilities.append(silTPos/len(transitionSlide))
     
        # #append probabilities to dictionary 
        # transition_dict[k] = probabilities

        return probabilities
 
  
    def slideWindows(self, windowSize):
        '''a slide window that run across the mood vector array, return vector shows the probability of each transition state'''
        probabilities_dict = {}
        windowSize = 30
        t = self.get_mood_transitions()
        #count = 0
        for k, TransArray in t.items():
            #value in the dictionary is an array with all the transition states
            start = 0
            probabilities_array = []
            
            for i in range(0, len(TransArray)-1, windowSize):   
            
                end = start+windowSize
                slide = TransArray[start:end]
                start = end
                #compute transition probability of a slide
                probabilities = self.transitionPro(slide)
                probabilities_array.append(probabilities)
          
            # append transition probability to dictionary  
            probabilities_dict[k] = probabilities_array
            #print(len(probabilities_array))
        return probabilities_dict



    def transMat_(self, transitionArray):
        ''' get transition matrix'''
        
        #transition states  
        posTOPos = 0 
        PosTNeg = 0
        PosTNeu = 0 
        PosTSil = 0

        negaToNega = 0
        NegTPos = 0
        NegTNeu = 0
        NegTSil = 0

        neuTNeu = 0
        NeuTPos = 0
        NeuTNeg = 0
        NeuSil = 0
        

        silToSil = 0
        silTNeg = 0
        silTNeu = 0
        silTPos = 0

        transMat = []
        positive_tran = []
        negative_tran =[]
        neutral_tran =[]
        silence_tran=[]

        transMat_pos = []
        transMat_neg =[]
        transMat_neu =[]
        transMat_sil =[]
        for transition in transitionArray:
            
            if transition == 2:
                posTOPos = posTOPos + 1
            if transition == 3:
                PosTNeg = PosTNeg + 1
            if transition == 6:
                PosTNeu = PosTNeu + 1
            if transition == 10:
                PosTSil = PosTSil+ 1

            if transition == 1:
                negaToNega = negaToNega + 1
            if transition == 4:
                NegTPos = NegTPos + 1
            if transition == 8:
                NegTNeu = NegTNeu+ 1
            if transition == 12:
                NegTSil = NegTSil + 1

            if transition == 0:
                neuTNeu = neuTNeu + 1
            if transition == 5:
                NeuTPos = NeuTPos + 1
            if transition == 7:
                NeuTNeg = NeuTNeg + 1
            if transition == 14:
                NeuSil = NeuSil + 1

            if transition == 15:
                silToSil = silToSil + 1
            if transition == 11:
                silTNeg = silTNeg + 1
            if transition == 13:
                silTNeu = silTNeu + 1
            if transition == 9:
                silTPos = silTPos + 1

        positive_tran.append(posTOPos)
        positive_tran.append(PosTNeg)
        positive_tran.append(PosTNeu)
        positive_tran.append(PosTSil)

        negative_tran.append(negaToNega)
        negative_tran.append(NegTPos)
        negative_tran.append(NegTNeu)
        negative_tran.append(NegTSil)

        neutral_tran.append(neuTNeu)
        neutral_tran.append(NeuTPos)
        neutral_tran.append(NeuTNeg)
        neutral_tran.append(NeuSil)

        silence_tran.append(silToSil)
        silence_tran.append(silTNeg)
        silence_tran.append(silTNeu)
        silence_tran.append(silTPos)

        if sum(positive_tran) != 0:
            transMat_pos.append(posTOPos/sum(positive_tran))
            transMat_pos.append(PosTNeg/sum(positive_tran))
            transMat_pos.append(PosTNeu/sum(positive_tran))
            transMat_pos.append(PosTSil/sum(positive_tran))
        else: 
            transMat_pos.append([0,0,0,0])

        if sum(negative_tran) != 0:
            transMat_neg.append(negaToNega/sum(negative_tran))
            transMat_neg.append(NegTPos/sum(negative_tran))
            transMat_neg.append(NegTNeu/sum(negative_tran))
            transMat_neg.append(NegTSil/sum(negative_tran))
        else: 
            transMat_neg.append([0,0,0,0])

        if sum(neutral_tran) != 0:
            transMat_neu.append(neuTNeu/sum(neutral_tran))
            transMat_neu.append(NeuTPos/sum(neutral_tran))
            transMat_neu.append(NeuTNeg/sum(neutral_tran))
            transMat_neu.append(NeuSil/sum(neutral_tran))
        else: 
            transMat_neu.append([0,0,0,0])

        if sum(silence_tran) != 0:
            transMat_sil.append(silToSil/sum(silence_tran))
            transMat_sil.append(silTNeg/sum(silence_tran))
            transMat_sil.append(silTNeu/sum(silence_tran))
            transMat_sil.append(silTPos/sum(silence_tran))
        else: 
            transMat_sil.append([0,0,0,0])

        transMat.append(transMat_pos)
        transMat.append(transMat_neg)
        transMat.append(transMat_neu)
        transMat.append(transMat_sil)

        return transMat

    def get_transMat(self):
        '''a slide window that run across the mood vector array, return vector shows the probability of each transition state'''
        transMat_user_dict = {}
        
        t = self.get_mood_transitions()
        #count = 0
        for k, TransArray in t.items():
            probabilities = self.transMat_(TransArray)
            transMat_user_dict[k] = probabilities
        return transMat_user_dict
    # def get_transitions_df(self, path):
    #     '''convert transition of mood dict to df'''
    #     TransitionStates = get_mood_transitions(mood_vector_feature)  
    #     TransitionStates_df = pd.DataFrame.from_dict(TransitionStates).T
    #     TransitionStates_df.to_csv(path + './mood_vectors/mood_transition_frequent_user_window_{}.csv'.format(self.windowSzie)) #feature matrx for prediction 
    #     return TransitionStates_df

    # def get_transition_oneHoc(self, path):
    #     TransitionStates = get_transitions_df(path, windowSzie)
    #     #convert transition df to one hot
    #     onehoc_X = OneHotEncoder(handle_unknown='ignore')
    #     TransitionStatesOneHot = onehoc_X.fit_transform(TransitionStates.iloc[:,0:]).toarray()

    #     df = pd.DataFrame(TransitionStatesOneHot)
    #     df.columns = [str(col) + '_transitions' for col in df.columns]
    #     df.index = TransitionStates.index
    #     df.to_csv(path + './mood_vectors/mood_transition_one_hoc_frequent_user_window_{}.csv'.format(self.windowSzie))
    #     return df

#TransitionStatesOneHot = get_transition_oneHoc(path, windowSzie)

moodOb = MoodFeature(path = path, participants = participants)
ValenceObject = moodOb.get_mood_vector(365)

transition = TransitionMatrix(ValenceObject = ValenceObject)

#t = transition.slideWindows(30)   




