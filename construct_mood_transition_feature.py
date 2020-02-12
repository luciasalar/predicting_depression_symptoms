from construct_mood_feature import *
import itertools
import pandas as pd
import numpy as np


class TransitionMatrix:
    """compute transition matrix."""

    def __init__(self, ValenceObject, windowSize):
        """ValenceObject is a dictionary, key as userid, value as a vector to represent mood each day"""
        self.ValenceObject = ValenceObject
        self.windowSize = windowSize  # window size of the transition state

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
        # these are self transition states
            if valence == 1 and preValence == 1:
                negaTran = negaTran + 1
            elif valence == 2 and preValence == 2:
                posiTran = posiTran + 1
            elif valence == 0 and preValence == 0:
                neuTran = neuTran + 1
        # positive and negative transition:
            elif (valence == 1 and preValence == 2) or (valence == 2 and preValence == 1):
                PosAndNeg = PosAndNeg + 1
        # neutral and postive transition
            elif (valence == 0 and preValence == 2) or (valence == 2 and preValence == 0):
                NeuAndPos = NeuAndPos + 1
        # neutral and negative transition 
            elif (valence == 0 and preValence == 1) or (valence == 1 and preValence == 0):
                NeuAndNeg = NeuAndNeg + 1
        # Empty and positive transition
            elif (valence == -1 and preValence == 2) or (valence == 2 and preValence == -1):
                EmptyAndPos = EmptyAndPos + 1
        
        # Empty and negative transition
            elif (valence == -1 and preValence == 1) or (valence == 1 and preValence == -1):
                EmptyAndNeg = EmptyAndNeg + 1
        
        # Empty and neutral transition
            elif (valence == 0 and preValence == -1) or (valence == -1 and preValence == 0):
                EmptyAndNeu = EmptyAndNeu + 1
                
            preValence = valence
        return [emptyTran, negaTran, posiTran, neuTran, PosAndNeg, NeuAndPos, NeuAndNeg, EmptyAndPos, EmptyAndNeg, EmptyAndNeu]


    def get_transitions(self, valenceVec):
        """create a list that shows the transition states postive - negative:3,
        neu - positive: 4, neu - positive: 5"""
        # count number of transition
        transitions = []
        preValence = 0
        for valence in valenceVec:
            # these are self transition states
            if valence == -1 and preValence == -1:
                transitions.append(1)
            elif valence == 1 and preValence == 1:
                transitions.append(2)
            elif valence == 0 and preValence == 0:
                transitions.append(0)
            # positive to negative transition:
            elif valence == 1 and preValence == -1:
                transitions.append(3)
            # negative to positive transition:
            elif valence == -1 and preValence == 1:
                transitions.append(4)
            # neutral to postive transition
            elif valence == 0 and preValence == 1:
                transitions.append(5)
            # positive to neutral transition
            elif valence == 1 and preValence == 0:
                transitions.append(6)
            # neutral to negative transition
            elif valence == 0 and preValence == -1:
                transitions.append(7)
            # negative to neutral transition
            elif valence == -1 and preValence == 0:
                transitions.append(8)
            # Empty to positive transition
            elif valence == 4 and preValence == 1:
                transitions.append(9)
            # positive to empty transition
            elif valence == 1 and preValence == 4:
                transitions.append(10)
            # Empty to negative transition
            elif valence == 4 and preValence == -1:
                transitions.append(11)
            # negative to empty transition
            elif valence == -1 and preValence == 4:
                transitions.append(12)
            # Empty to neutral transition
            elif valence == 4 and preValence == 0:
                transitions.append(13)
            # neutral to empty transition
            elif valence == 0 and preValence == 4:
                transitions.append(14)
            # Empty and empty
            elif valence == 4 and preValence == 4:
                transitions.append(15)
            preValence = valence
        return transitions


    def get_mood_transitions(self):
        '''get transtions of mood dict'''
        result = {}
        for k, v in self.ValenceObject.items():
            b = np.where(np.isnan(v), 4, v)
            result[k] = self.get_transitions(b)
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
        # get probabilities
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
 
  
    def slideWindows(self):
        '''a slide window that run across the mood vector array, return vector 
        shows the probability of each transition state'''
        probabilities_dict = {}
        
        t = self.get_mood_transitions()
        #count = 0
        for k, TransArray in t.items():
            #value in the dictionary is an array with all the transition states
            start = 0
            probabilities_array = []
            pre_pro = None
            
            for i in range(0, len(TransArray)-1, self.windowSize):   
            
                end = start+self.windowSize
                slide = TransArray[start:end]
                start = end
                #compute transition probability of a slide
                probabilities = self.transitionPro(slide)
                if pre_pro is not None:
                    probabilities_array.append(pre_pro)
                pre_pro = probabilities #here we get rid of result from the last window because it might contain only a few days
              
            # append transition probability to dictionary  
            probabilities_dict[k] = probabilities_array
            #print(len(probabilities_array))-
        return probabilities_dict


    def slideWindows2(self):
        '''a slide window that run across the mood vector array, return vector shows the probability of each transition state'''
        probabilities_dict = {}
        
        t = self.get_mood_transitions()

        for k, TransArray in t.items():
            # value in the dictionary is an array with all the transition states
            start = 0
            probabilities_array = []
            prev_pro = None
            
            for i in range(0, len(TransArray) - 1, self.windowSize):
            # here the step equals to window size
                end = start + self.windowSize
                slide = TransArray[start:end]
                start = end
                # compute transition probability of a slide
                probabilities = self.transitionPro(slide)
                if prev_pro is not None:
                    result_pro = np.asarray(probabilities) - np.asarray(prev_pro)
                
                    probabilities_array.append(result_pro)

                prev_pro = probabilities
            # append transition probability to dictionary
            probabilities_dict[k] = list(itertools.chain(*probabilities_array))

        return probabilities_dict


    def get_mood_transitions_pro(self):
        '''get mood transition feature dict '''
        mood_tran = self.slideWindows() 
        mood_tran_dict = {}
        for k, v in mood_tran.items():
            mood_tran_dict[k] = list(itertools.chain(*v))

        TransitionStates_df = pd.DataFrame.from_dict(mood_tran_dict).T
        return TransitionStates_df

    def get_transitions_momentum(self):
        '''get mood transition and transition momentum features '''
        mood_tran_df = self.get_mood_transitions_pro()
        mood_t_momentum = self.slideWindows2()
        mood_t_momentum_df = pd.DataFrame.from_dict(mood_t_momentum).T

        return mood_t_momentum_df, mood_tran_df

# construct a daily mood vector


moodOb = MoodFeature(path=path, participants=participants)
# here returns a function with mood on 365 days
ValenceObject = moodOb.get_mood_vector(365)

# now convert the mood vector to transition features
transition = TransitionMatrix(ValenceObject=ValenceObject, windowSize = 30)
a, b = transition.get_transitions_momentum()





