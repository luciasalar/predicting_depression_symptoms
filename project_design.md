# Project design

In this project, we want to examine does mood or transitions of mood indicated in social media data can be used to infer depression

## Literature:

We will first introduce:

Mood and emotions and depression
Mood and emotions on social media, how researchers use them to predict depression

We move on to discuss these models do not look at the transtion and do not consider mood as a temporal feature, and why transtions are important

Then we will discuss the problem of:
using social media data to predict mental illnesses (cite review )
and then we explain Why we still use social media data to infer depression symptoms

1.  Clinical studies suggest mood is an important indicator of depression, there's a lot of controversies about whether social media sentiment reflects real life mood. By looking at whether social media mood predicts depression symptoms, we gain insights on the degree to which social media mood can reflect real life mood.
2. Existing literature use sentiment score within a specific period of time as feature, therefore, the temporal element is not included in the model prediction. Here we construct temporal mood features and we explore using transitions of mood as features.
3.  


# Here's our design:
## Our goal

Develop a prediction model on depression symptoms:
model 1: This model use a set of features including, averaged sentiment score in 1 year, LIWC, demographic, part-of-speech, readability score asn so on (baseline) (also check again to see if there's a paper predicting depression using this dataset)
model 2: Same set of features as model 1 but we replace the averaged sentiment score in 1 year with temporal mood feature
model 3: Same set of features as model 1 but we replace the averaged sentiment score in 1 year with temporal mood transition feature
--- 
model 4 (extension): neural network

## machine annotated sentiment

Tool: sentistrength

Valence labels: positive, negative, mix, neutral

Here are some examples from SentiStrength

The text 'I love you but hate the current political climate.'
has trinary result -1.

Approximate classification rationale: I love[3] you but hate[-4] the current political climate.[sentence: 3,-4] [overall result = -1 as pos<-neg] 


The text 'There is a bike there'
has positive strength 1 and negative strength -1

Approximate classification rationale: There is a bike there [sentence: 1,-1] [result: max + and - of any sentence][trinary result = 0 as pos=1 neg=-1] 

$ we can define document with postive > 3, negative < -3 as mixed. Here we can change the mapping according to mannual judegment

## traditional machine learning Approach

### Mood vectors: 
Mood vector:  {P, N, Neu, S, M}  five categories, S: silence days

We use a sliding window to define the period of time for mood, it can be 1 day, 3 days, 1 week ...

We use the vectors to infer depression:

step 1:  select data of X day   
step 2: assign dominant valence as mood for the window 

mood vector, window 1 day:
[P, N, Neu, S, P, N.... ]  
Vd1, Vd2, Vd3.... di    i= X


step 3: Build temporal features:
$Design 1: We can compute the mood in the past X days, X equal to a string of numbers with an interval of [i], i is meaningful in psychology theory, here we can select i = 7
X= [7, 14, 21....]


### mood transition vector
step 1: same as mood vector

step 2: 
Mood transition vector: {P-N, P-Neu, N-Neu, N-S, S-P, S-Neu, P-M, M-Neu, M-N}, window 2 days

step 3: same as mood vector 

-------Let's try the traditional approach first, this is for extension-----------------------

## neural network approach
neural network approach gives us more flexibility in designing the features

step 1: same as mood vector


step 2: define mood vector of window L, the unit of L is day 

mood vector, window L days: 
[P, N, Neu, S, P, N.... ] 
[Vw1, Vw2, Vw3.....wi]    i = X/L



step 3: create probabilty matrix of mood (stochastic matrix)

five states =  {P, N, Neu, S, M}, in each window, we have probabily for each mood, all the probabilities sum up to 1
[[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.2, 0.3, 0.3] ...] 


step 4: predict depression 

step 5: Here we can have a mood embedding for depression 

















