#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
import seaborn as sns


# In[3]:


#Reading csv file

df= pd.read_csv('train.csv')
df


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


#Checking missing values

df.isnull().sum()


# In[30]:


#Age-Missing Values

print('Percent of missing "Age" records is %.2f%%' 
      %((df['Age'].isnull().sum()/df.shape[0])*100))

ax = df["Age"].hist(bins=15, density=True, stacked=True, color='teal', 
                    alpha=0.6)
df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[26]:


#Cabin missing values

print('Percent of missing Cabin records is %.2f%%' %
      ((df['Cabin'].isnull().sum()/df.shape[0])*100))


# In[9]:


#Embarked missing values

print('Percent of missing "Embarked" records is %.2f%%' 
      %((df['Embarked'].isnull().sum()/df.shape[0])*100))

print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=df, palette='Set2')
plt.show()


# In[10]:


train_data = df.copy()
train_data["Age"].fillna(df["Age"].median(skipna=True), inplace=True)

train_data["Embarked"].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# In[11]:


train_data.isnull().sum()


# In[12]:


plt.figure(figsize=(15,8))
ax = df["Age"].hist(bins=15, density=True, stacked=True, color='red', alpha=0.6)
df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='black', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[13]:


# Create categorical variable for traveling alone
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)


# In[14]:


#create categorical variables and drop some variables

training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()


# In[15]:


#EDA

plt.figure(figsize=(15,8))
ax = sns.kdeplot(train_data["Age"][train_data.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(train_data["Age"][train_data.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[16]:



plt.figure(figsize=(20,8))
avg_survival_byage = train_data[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="Pink")
plt.show()


# In[17]:


#Exploration of Passenger class

sns.barplot('Pclass', 'Survived', data=df, color="darkturquoise")
plt.show()


# In[18]:


#Exploration of Embarked Port

sns.barplot('Embarked', 'Survived', data=df, color="teal")
plt.show()


# In[19]:


#Exploration of Traveling Alone vs. With Family

sns.barplot('TravelAlone', 'Survived', data=train_data, color="mediumturquoise")
plt.show()


# In[20]:


#Exploration of Gender Variable

sns.barplot('Sex', 'Survived', data=df, color="aquamarine")
plt.show()


# In[21]:


#Logistic Regression and Results

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male"]
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 8)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[22]:


#Feature ranking with recursive feature elimination and cross-validation

from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))


# In[23]:


# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:





# In[ ]:




