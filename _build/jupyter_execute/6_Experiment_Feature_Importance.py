#!/usr/bin/env python
# coding: utf-8

# # 6. Experiment Feature Importance

# When an experiment is paused - regardless of whether we have statisticall conclusive or inconclusive results  - we may want to do some more digging into the results from our experiment. Luckily, in the era of big data and digital analytics there is a plethora of information that we may have at our disposal in relation to your experimentation data. This chapter will breakdown some basic machine learning techniques, using a Random Forest model. We will then use feature importance - using out of the box methods from Sklearn and using [SHAP (SHapley Additive exPlanations)](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d) techniques, respectively. The aim of this chapter is to determine whether:
# 
# A.) We can build a dependent machine learning model for predicting performance during our experiment period
# 
# B.) Of the feature variables used in modeling, which had the highest reletive importance in predicting high or low outcomes
# 
# As machine learning is such a wide and varied field, with many naunces - this chapter will not focus on computing an advance or highly tuned Random Forest model, but you can read some of my previous work on [Random Forest Modeling here](https://towardsdatascience.com/building-predictive-models-with-myanimelist-and-sklearn-54edc6c9fff3).
# 
# This chapter is primarily concerned with performing feature importance and analysing individual predictions, as a method to supplement and/or support your experimentation analysis.
# 
# We will work with a variety of feature variables - as well as our experiment/variant labels - in order to determine their significance in predicting our experiment metric (page views per brower/pvs_pb). For this example we will work on an experiment called Keanu 2.0 where we have ascertained that the new imagery of Keanu is valid for production but we want now to determine the optimal positioning of the widget:
# 
# 
# ![](keanu_position_8.png)
# 
# As we are focused on analysis subsequent to our primary assessment of the experiment results, I will note here that the result of the experiment was flat; therefore, positioning of the widget had no statistically significant effect on page views per browser. So it is our job now to dig a bit deeper and look at other user data in relation to pvs_pb.
# 
# Firstly, import all necessary packages for analysis:

# In[1]:


get_ipython().system('pip install --no-progress-bar pymc3')
get_ipython().system('pip install --no-progress-bar numba')
get_ipython().system('pip install --no-progress-bar shap')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sb
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import shap as shap


# Import our Keanu 2.0 experiment data, which has our additional user features. We will include the variables we want to use in modeling:

# In[4]:


local = 'keanu_features.csv'
df = pd.read_csv(local, encoding='unicode_escape')
df.loc[(df.user_experience == 0),'user_experience']= 'Position_1'
df.loc[(df.user_experience == 1),'user_experience']= 'Position_2'
df.loc[(df.user_experience == 2),'user_experience']= 'Position_3'
df['pvs_pb'] = df.exp_metric
df = df[['date','user_experience','pvs_pb','browser','operatingSystem','cityId']]
df.head(100)


# ## Encode feature variables

# We can now encode all feature variables to be used for subsequent modelling; this ensures that variables will be in an appropriate format for predictions. Feature encoding is beyond the scope of this tutorial, but you can [read more about it here](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/).

# In[5]:


enconder = LabelEncoder()
user_experience_labels = enconder.fit_transform(df['user_experience'])
user_experience_mappings = {index: label for index, label in 
                  enumerate(enconder.classes_)}

browser_labels = enconder.fit_transform(df['browser'])
browser_mappings = {index: label for index, label in
                   enumerate(enconder.classes_)}

operatingSystem_labels = enconder.fit_transform(df['operatingSystem'])
operatingSystem_mappings = {index: label for index, label in
                   enumerate(enconder.classes_)}

cityId_labels = enconder.fit_transform(df['cityId'])
cityId_mappings = {index: label for index, label in
                   enumerate(enconder.classes_)}

print(user_experience_labels, browser_labels, operatingSystem_labels, cityId_labels)


# As you can see the encoded variables for each user in the experiment have been appended to the original data-frame, which will be used for predictive analysis:

# In[6]:


df['user_experience_labels'] = user_experience_labels
df['browser_labels'] = browser_labels
df['operatingSystem_labels'] = operatingSystem_labels
df['cityId_labels'] = cityId_labels
df.head(100)


# ## Random forest modelling

# As mentioned in the intro, we are going to fit a basic Random Forest model, with no [hyper-parameter tuning](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) performed; so results may vary if you were to tune your model to perform optimially. As we have a vast amount of data, running Sklearn and SHAP functions locally can be computaionally exhaustive, so for proof of concept I am working with a subset of our total data for modeling:

# In[7]:


_model_sample = df.sample(frac = 0.01)


# Now, input feature variables (X/predictors) and target variable (y/pvs_pb) and make a test (25%) and training (75%) data-set: 

# In[8]:


## Feature variables

features = _model_sample[[ 'user_experience_labels',
                'browser_labels',
                'operatingSystem_labels',
                'cityId_labels'
]]

## Target variale

target = _model_sample.pvs_pb

## Test-train split (%)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25) ## Split into test and training set


# We can now fit our Random Forest model and view the first few predictions and the average prediction of pvs_pb made per observation:

# In[9]:


rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)

y_predictions = rfr.predict(X_test)

print(y_predictions)
print("The average prediction of the Random Forest was",y_predictions.mean(),"pvs_pb")


# We can now compute the R^2 value of our model. The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0:

# In[10]:


r_2 = rfr.score(X_test, y_test)
print("The r^2 value for the random forest regressor is:",round(r_2,4),)


# ## Feature importance

# We can now drill down into our feature variables and their relative importance in predictin our pvs_pb metric. The following code will loop through our feature variables and our Random Forest model and show us this information:

# In[11]:


for name, importance in zip(features, rfr.feature_importances_):
    print(name, "=", importance)


# We can present this visually as such:

# In[12]:


plt.figure(figsize=(15,10))
feat_importances = pd.Series(rfr.feature_importances_, index=features.columns)
feat_importances.nlargest().plot(kind='barh', color = 'purple')


# We can see above that by far the greatest predictor in our model was the cityId_labels, with the other feature variables expressing parity in their relative contributions. Although the predictive capabiltiies does not mean causality exists, we can conclude that only the cityId_labels were relevant to model prediction, seeing as we know the result from the experiment was flat (user_experience_labels). We now move onto SHAP methods, which allow us to drill down into individual predictions and relative impacts on model output.

# ## SHAP Features

# "Think about this: If you ask me to swallow a black pill without telling me what’s in it, I certainly don’t want to swallow it. The interpretability of a model is like a label on a drug bottle. We need to make our effective pill transparent for easy adoption" - I took this quote from a [piece by Dr. Dataman](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d) (one can only assume this is his/her real name!) I took this quote because it summarises why this form of analysis is so important (and this is not a segway from experimentation/A/B Testing). We can create a model to make predictions of which variables are strong predictors of our performance metric, but it's important to know which features have informed individual observations as well as the wholse test data. 
# 
# This is where we can use SHAP to look into overall model impact and individual observations. SHAP (SHapley Additive exPlanations) uses a game theoretic approach to explain the output of a given machine learning model. Furthermore, it unifies optimal credit allocation with local explanations using the [classic Shapley values](https://www.investopedia.com/terms/s/shapley-value.asp#:~:text=Key%20Takeaways-,In%20game%20theory%2C%20the%20Shapley%20value%20is%20a%20solution%20concept,other%20to%20obtain%20the%20payoff.). A Shapley value is computed by averaging out the marginal contributions of feature variables across all possible permutations, and for this reason can be computationally exhaustive (so be ready to wait for these functions to run). "In game theory, a game can be a set of circumstances whereby two or more players or decision-makers contribute to an outcome. The strategy is the gameplan that a player implements while the payoff is the gain achieved for arriving at the desired outcome." In the example of our experiment, the desired outcome is the correct prediction of our performance/dependent variable (pvs_pb).
# 
# 
# Here is a schematic of how the SHAP package works with our feature variables to predict our performance metric:
# 
# ![](ab_shap_overview_9.png)
# 
# This section will help us determine overall relative importance to our model and individual observations in our model.
# 
# First create an explainer variable and generate our SHAP values for all of our features:

# In[13]:


explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(features)
shap.initjs()


# A feature variable importance plot lists the most significant variables in descending order. The top feature variable contributed more to the model than the bottom ones and thus yield greater predictive capabilities. SHAP feature importance offer an alternative to regular derivations (as seen above) and actually provide us with differing information. The most significant difference between importance measures is that permutation feature importance is predicated on the decrease in overall model performance; whilst SHAP is based on magnitude of feature attributions (this is why results may vary from that above):
# 

# In[14]:


shap.initjs()
shap.summary_plot(shap_values, features, plot_type="bar")


# The below figure exhibits features each contributing to push the model output from the base value (which is the average model output over the training dataset we passed through) to the model output. The feature variables pushing the prediction higher are shown in red, those pushing the prediction lower are in blue. Here we can see which feature variables pushed the prediction above baseline for a positive prediction. You can use .iloc to locate positions within your test data to observe these contributions:

# In[20]:


shap.force_plot(explainer.expected_value, shap_values[5, :], X_test.iloc[20, :])


# And for a negative prediction:

# In[15]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[100, :], X_test.iloc[1, :])


# This chapter has elucidated how some basic machine learning methods can supplement your experimentation analysis. Using feature importance can be a useful extension to your regular significance testing if you have obtained a flat test result or wish to dig deeper into an observed change. Furthermore, identifying such changes/effects can help you re-hypothesise and generate new test ideas.
