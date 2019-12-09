# Project design

In this project, we want to examine does mood or transitions of mood indicated in social media data can be used to infer depression symptoms


## Literature:

* We will first introduce how researchers use social media data to predict mental illnesses symptoms

* Then we will discuss the concerns of using social media data to predict mental illnesses (Methodological Gaps in Predicting Mental Health States from Social Media: Triangulating Diagnostic Signals)

1. Do these diagnostic signals measure what they claim to measure? (mismatched between predictive features in the proxy classifiers and those of the patient model, classifier capture support seeking behaviors, interest in others’ lived experiences of the illness, self-reported accounts of stigma and inhibition)

2. Is what is being measured by a diagnostic signal itself valid?
'a lack of clinical grounding in the diagnostic information (individual’s clinical mental health state) that these signals intend to measure. Instead, what these signals presume as diagnostic information are essentially behavioral patterns associated with the appropriation of social media by a wide variety of stakeholders, not necessarily patients, in relation to the illness. '

'a lack of theoretical underpinning in the ways the diagnostic signals were identified.'


### To address the questions above, we examine using mood pattern to infer depression symptoms. 

*Maria's note: You cannot address those questions with your data set, because all you have is the behavioural pattern, and the CES-D is NOT a diagnostic instrument. I know that Ernala et al. also work just from the social media data, but I think that their approach of diagnosis by Twitter, even though it's done by clinical experts, is deeply flawed.* 

Explain mood and emotions and depression.  Clinical studies suggest mood is an important indicator of depression. 

Explain the current approach of using sentiment as one of the features to predict depression symptoms in the social media context. Their approach didn't capture the temporal component of mood and mood transitions.

### Contribution 
1. By looking at whether social media mood predicts depression symptoms, we gain insights on whether it's possible to build a ML model that is more algin with clinical theories. 
2. Existing literature use sentiment score within a specific period of time as feature, therefore, the temporal element is not included in the model prediction. Here we construct temporal mood features and we explore using transitions of mood as features.

*Maria's note: I think that 2 is your main contribution.*

### Limitation 
social media mood is different from real life mood

### Research questions
1. Can mood patterns and mood transition patterns indicated by social media text predict depression symptoms?


# Here's our design:
A lot of the prediction models involving time would use time series approach. A typical example is stock market prediction https://github.com/gdroguski/GaussianProcesses These models use the previous states to infer the probability of the next state. Such type of model is good at predicting mood, in which the previous mood state somewhat determine the next mood state. Here we use mood states to infer depression symptoms. However, we are not using mood only to infer depression symptoms, we also include other types of feature, e.g. mentioning of certain keywords, demographic info so on, combined with mood to infer the symptoms. In the case that time feature combine with non-time feature in a predicition model, mood time pair is used to represent time features. 

# Data: Here I would suggest not to use the frequent user criteria anymore, because, 1. we don't have enough negative case for training. Second, we also modeled silence days, it's probably not a good idea to use a cutoff point to justify why we are not modelling people with slightly less posts. 

*most of the papers used predict depression symptoms has a sample of 2:3 or 1:1 with positive and negative classes. These proportion do not represent the actual positive and negative class in real life. However, studies that look at social media users' self-reported depression symptom score often attracted users who tend to have higher symptoms. In our case, this proportion is 1.7:1 (positive: negative). This proportion is similar to xxx's paper (cite the nature paper and list other papers). We need to be aware that these models are trained and tested on samples that are not exactly the same as random real life samples, rather, they are trained and tested on social media users who are interested to evaluate their depression symptoms. It's likely that some of these users already suspect themselves have high symptom levels. In this study, We train models on different proportion of positive and negative classes, recall and precision of positive class is better as we increase the proportion of the negative class.* 

* frequent users: frequent_users.csv, users have at least 1 post in half of the weeks in the past 1 year.
The proportion of positive and negative in this sample is 

* all users: participants_match.csv
The proportion of positive and negative in this sample is 

* adjusted sample: adjusted sample.csv 
The proportion of positive and negative in this sample is 2:3, here I selected all the negative sample from all users and randomly select sample from positive users to adjust the propotion. Therefore, this sample contains both frequent and non-frequent users.

* adjusted_sample2.csv The proportion of positive and negative in this sample is 1:1

Here I train all the models with frequent users, all users, and adjusted sample users are divided in into train/test set before model training
---

# Part 1 HMM model mood changes

The purpose of buiding the HMM model is not to make prediction, it is to model the mood and observe the emission probabilities to different observations states. To evalutate whether the model performance, we also fit it on a new set of data.

* two hidden states: 0, 1   In reality, users who have high depression symptom is a relatively stable variable, we can't say user transfer from having high symptom to low symptom within 1 day, therefore, we can only assume the two states are somewhat related to depression symptoms

* four observations: positive, negative, neutral and silence

** method: I use day level observation (mood) (365 days before user completed CES-D) as input of the model. HMM compute the emission and transition matrix. Emission matrix is stored in data/results/HMMresult.csv. Then we evaluate the model by using the model to predict hidden states on a set of holdout users**

###initialize emmission prob 

			 	#positive negative neutral    silence
		#low       0.2      0.3      0.1          0.4
		#high      0.3      0.1      0.2          0.4

###trained emission prob: data/results/HMMresult.csv.
		low		[[0.08694357 0.05289536 0.04695897 0.81320211]

 		high 	[0.0321883  0.13017332 0.07177855 0.76585982]]
 
* Here we can see the emission probability makes a lot of sense, people with low symptoms have more silence days

* result: The emission and transition probabilities didnt change even though I tried different initial numbers for the emission and transition probabilities. This shows that the model is quite stable. Next, I use the trained model to predict the hidden states of a holdout set. For the holdout set, I use day level observation (mood) (365 days before user completed CES-D) as input of the. We obtain 365 hidden states (0: low symptom, 1: high symptom). The classification report is stored in data/results/HMMclassification_report.csv. 

To evaluate the model, we need to decide a theshold to label users as having high symptoms in general. We decide the threshold as: in the past X days, if users have y days in 1. 

Here's an example: data/results/HMMclassification_report.csv
From the 1:1 trainning sample, user have 7 days in the past 14 days have 1. In reality, users who have high depression symptom is a relatively stable variable, thh symptom states are labeled as high symptom in general 

                 low       high  accuracy   macro avg  weighted avg

f1-score    0.680672   0.208333   0.54491    0.444503      0.440260

precision   0.519231   0.909091   0.54491    0.714161      0.717663

recall      0.987805   0.117647   0.54491    0.552726      0.544910

support    82.000000  85.000000   0.54491  167.000000    167.000000


Develop a prediction model on depression symptoms:
### ML models
**Models are specified in the experiment/experiment.yaml file** 
*features in yaml file indicate the type of model*

* mood: mood within a time window X: Mood is categorical in this feature, silence day is filled by np.nan. Mood within a time window X, this time window move across the mood vector from day 1 to day 365, if time window = 3, we'll have mood from day 1-3, 4-6...  Since sklearn classifiers can't handle nan values, we impute the nan values in the mood feature with mean of the vector (This is not ideal, that's why we want to introduce guassian process in here)

* mood change: mood change feature within a time window X: Here the mood value is the average sentistrength value of the day (not categorical anymore). If time window is 3, we have average mood from (day6 to day 4) - (average mood from day 3 to day 1)

* mood transition: number of mood transitions in the last X days. X is usually 30 days


* Results: the ML models are still in training, we also need to test the size of the sliding window, in the result.csv file, all the window size is 3 so far. We can see that the precision score of the positive class increase a lot once we added the mood features. For example:

No 14 (row number) is a relatively good model

                   0          1  accuracy  macro avg  weighted avg
f1-score    0.717949   0.400000  0.616279   0.558974      0.581157
precision   0.617647   0.611111  0.616279   0.614379      0.614835
recall      0.857143   0.297297  0.616279   0.577220      0.616279
support    49.000000  37.000000  0.616279  86.000000     86.000000




* model 1: This model use a set of features including, averaged sentiment score in 1 year, LIWC, demographic, part-of-speech, readability score asn so on (baseline) (also check again to see if there's a paper predicting depression using this dataset)
* model 2: Same set of features as model 1 but we replace the averaged sentiment score in 1 year with temporal mood feature
* model 3: Same set of features as model 1 but we replace the averaged sentiment score in 1 year with temporal mood transition feature

*Maria's note: Good idea to integrate posting patterns like this.*


 
-------------------------------------
* model 4 (extension): neural network

## machine annotated sentiment

* Tool: sentistrength

* Valence labels: positive, negative, mix, neutral

Here are some examples from SentiStrength

The text 'I love you but hate the current political climate.'
has trinary result -1.

Approximate classification rationale: I love[3] you but hate[-4] the current political climate.[sentence: 3,-4] [overall result = -1 as pos<-neg] 


The text 'There is a bike there'
has positive strength 1 and negative strength -1

Approximate classification rationale: There is a bike there [sentence: 1,-1] [result: max + and - of any sentence][trinary result = 0 as pos=1 neg=-1] 

---------------------------------
we can define document with postive > 3, negative < -3 as mixed. Here we can change the mapping according to mannual judegment

*Maria's note: Make sure that you select the threshold on a "training set", and then validate it on a held out data set with equal proportions of each category.*

## traditional machine learning Approach

### mood vectors: 
Mood vector:  {P, N, Neu, S}  four categories, S: silence days

We use a sliding window to define the period of time for mood, it can be 1 day, 3 days, 1 week ...

We use the vectors to infer depression:

* step 1:  select data of X day   
* step 2: assign dominant valence as mood for the window 

mood vector, window 1 day:
[P, N, Neu, S, P, N.... ]  
Vd1, Vd2, Vd3.... di    i= X


* step 3: Build temporal features:
$Design 1: We can compute the mood in the past X days, in each step we move X+1 days. For example, feature 1 = mood from day 0 -day 7, feature 2 = mood from day 1-day 8 ...


### mood transition vector
* step 1: same as mood vector

* step 2: 
Mood transition vector: {P-N, P-Neu, N-Neu, N-S, S-P, S-Neu, P-M, M-Neu, M-N}, window 2 days

* step 3: same as mood vector 

Let's try the traditional approach first, this is for extension
------------------------------


## neural network approach
neural network approach gives us more flexibility in designing the features

* step 1: same as mood vector


* step 2: define mood vector of window L, the unit of L is day 

mood vector, window L days: 
[P, N, Neu, S, P, N.... ] 
[Vw1, Vw2, Vw3.....wi]    i = X/L



* step 3: create probabilty matrix of mood (stochastic matrix)

five states =  {P, N, Neu, S, M}, in each window, we have probabily for each mood, all the probabilities sum up to 1
[[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.2, 0.3, 0.3] ...] 


* step 4: predict depression symptoms

* step 5: Here we can have a mood embedding for depression 
















