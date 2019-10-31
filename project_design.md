# Project design

In this project, we want to examine does mood or transitions of mood indicated in social media data can be used to infer depression symptoms


## Literature:

* We will first introduce how researchers use social media data to predict mental illnesses symptoms

* Then we will discuss the concerns of using social media data to predict mental illnesses (Methodological Gaps in Predicting Mental Health States from Social Media: Triangulating Diagnostic Signals)

This paper raise a concern that the machine learning models capture signals related to support seeking behaviors, interest in others’ lived experiences of the illness, self-reported accounts of stigma and inhibition and so on, rather than behavioral patterns that a clinician would observe.


'a lack of theoretical underpinning in the ways the symptom signals were identified.'


### To address the concern above, we examine using mood pattern to infer depression symptoms. 

*Maria's note: You cannot address those questions with your data set, because all you have is the behavioural pattern, and the CES-D is NOT a diagnostic instrument. I know that Ernala et al. also work just from the social media data, but I think that their approach of diagnosis by Twitter, even though it's done by clinical experts, is deeply flawed.* 

*I have rephrased the concern raised in the paper*

Explain mood and emotions and depression.  Clinical studies suggest mood is an important indicator of depression. 

Explain the current approach of using sentiment as one of the features to predict depression symptoms in the social media context. Their approach didn't capture the temporal component of mood and mood transitions.

### Contribution 

1. Existing literature use sentiment score within a specific period of time as feature, therefore, the temporal element is not included in the model prediction. Here we construct temporal mood features and we explore using transitions of mood as features.
2. By using features that aligns with behavioral symptoms documented in the clinical literature, there is a potential to build assistive tools for clinicals to observe patients' affective changes over time.

*Maria's note: I think that 2 is your main contribution.*

### Limitation 
social media mood is different from real life mood

### Research questions
1. Can mood patterns and mood transition patterns indicated by social media text predict depression symptoms?


# Here's our design:


Develop a prediction model on depression symptoms:
### discuss: should we build one model with all users and one model with frequent users only? I selected a set of users who posted at least once in 25 weeks each year. However, I don't think selecting frequent users is a good idea, the model is suppose to learn whether frequent/non frequent posting behavior predicts depression

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
Mood vector:  {P, N, Neu, S, M}  five categories, S: silence days

We use a sliding window to define the period of time for mood, it can be 1 day, 3 days, 1 week ...

We use the vectors to infer depression:

* step 1:  select data of X day   
* step 2: assign dominant valence as mood for the window 

mood vector, window 1 day:
[P, N, Neu, S, P, N.... ]  
Vd1, Vd2, Vd3.... di    i= X


* step 3: Build temporal features:
$Design 1: We can compute the mood in the past X days, in each step we move X+1 days. For example, feature 1 = mood from day 0 -day 7, feature 2 = mood from day 1-day 8 ...

### original content

original content: {0, 1, 0, 1, 1}
We can either use a sliding window to define original content or just use the counts


### mood transition vector (I don't think we need this one anymore because the mood vector already capture the mood changes, we can add transtion states count as feature)

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
















