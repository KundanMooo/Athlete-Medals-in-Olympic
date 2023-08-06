#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv('D:/DS/resume projects/athlete seaborn/athlete_events.csv')


# In[3]:


df.shape


# In[4]:


df


# In[5]:


df.size


# In[6]:


df


# In[7]:


# athelete who won gold from Team US
c=len(df[(df['Team']=='United States') & (df['Medal']=='Gold')])
c


# In[8]:


#Q1 analyse hight data by removing None
sns.displot(x=df.Height.dropna(),bins=30,kde=True)


# In[9]:


#Q1 analyse weight data by removing None
sns.displot(x=df.Weight.dropna(),bins=40,kde=True)


# In[10]:


plt.figure(figsize=(10,5))
sns.displot(x=df.Weight.dropna(),bins=65)


# In[11]:


sns.kdeplot(x=df.Weight.dropna())


# In[12]:


sns.histplot(x=df.Weight.dropna(),bins=65,color='green')


# In[13]:


sns.kdeplot(x=df.Weight,color='red',fill='red')


# In[14]:


sns.kdeplot(x=df.Height.dropna(),label='Height',fill='red')
sns.kdeplot(x=df.Weight.dropna(),label='Weight',fill='red')
plt.legend()


# In[15]:


sns.jointplot(y=df.Height.dropna(), x=df.Weight.dropna(), kind='reg')


# In[16]:


sns.pairplot(df,kind='reg')


# In[17]:


sns.jointplot(x=df.Weight.dropna(),y=df.Height.dropna(),xlim=(20,180),ylim=(135,205))


# In[22]:


sns.pairplot(df,hue='Medal',palette=['gold','black','brown'])


# In[16]:


df


# In[18]:


sns.pairplot(df)


# In[24]:


sns.swarmplot(x=df.Medal,y=df.ID)


# In[26]:


import seaborn as sns
import pandas as pd

df = pd.read_csv('D:/DS/resume projects/athlete seaborn/athlete_events.csv')

sns.violinplot(x='Medal', y='ID', data=df)


# In[28]:


sns.pairplot(df[['Height','Weight','Age','Medal']].dropna(),hue='Medal')


# In[35]:


sns.heatmap(df.dropna().corr(),annot=True,cmap='cool')


# In[18]:


sns.heatmap(df.dropna().corr(),annot=True,cmap='cool',
            x_var=['Age','Height','Weight','Year'],y_var=['Age','Height','Weight','Year'])


# In[19]:


sns.heatmap(df[['Age','Height','Weight']].dropna().corr(),annot=True,cmap='cool')


# In[17]:


#medalistcount by top 20 country
sns.histplot(x=df.Team,y=df.ID)
plt.show()


# In[20]:


df


# In[21]:


df1=df.groupby('Team').count()['Medal'].sort_values(ascending=False)
df1=df1[0:20]
df1=df1.reset_index()


# In[22]:


plt.figure(figsize=(30,10))
sns.barplot(x=df1.Team,y=df1.Medal)
plt.show()


# In[23]:


dfg=df[df['Medal']=='Gold'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(10)
dfs=df[df['Medal']=='Silver'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(10)
dfb=df[df['Medal']=='Bronze'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(10)


# In[24]:


sns.barplot(x=dfg.Team.dropna(),y=dfg.Medal.dropna())
plt.show()
sns.barplot(x=dfs.Team.dropna(),y=dfs.Medal.dropna())
plt.show()
sns.barplot(x=dfb.Team.dropna(),y=dfb.Medal.dropna())
plt.legend()
plt.show()


# In[25]:


plt.bar(x=dfg.Team.dropna(),y=dfg.Medal.dropna(),width=0.5)
sns.barplot(x=dfs.Team.dropna(),y=dfs.Medal.dropna())
sns.barplot(x=dfb.Team.dropna(),y=dfb.Medal.dropna())
plt.show()


# In[173]:


plt.bar(x=dfg.Team.dropna(),y=dfg.Medal.dropna(),width=5)


# In[26]:


plt.figure(figsize=(20,7))
plt.bar(x=dfg.Team, height=dfg.Medal, width=0.5,alpha=0.4,color='Gold',label='Gold')
plt.bar(x=dfs.Team,height=dfs.Medal,width=0.7,alpha=0.5,color='Silver',label='silver')
plt.bar(x=dfb.Team,height=dfb.Medal,width=0.9,alpha=0.2,color='brown',label='Bronze')
plt.legend()
plt.xticks(rotation=90)
plt.show()


# In[27]:


plt.bar(x=dfg.Team.dropna(),y=dfg.Medal.dropna(),width=0.5)


# In[28]:


g=np.arange(len(dfg))-.2
s=np.arange(len(dfg))
b=np.arange(len(dfg))+.2


# In[29]:


g


# In[30]:


b


# In[31]:


plt.figure(figsize=(20,7))
plt.bar(x=g, height=dfg.Medal, width=0.2,alpha=0.4,color='Gold',label='Gold')
plt.bar(x=dfs.Team,height=dfs.Medal,width=0.2,alpha=0.9,color='Silver',label='silver')
plt.bar(x=b,height=dfb.Medal,width=0.2,alpha=0.7,color='brown',label='Bronze')
plt.legend()
plt.xticks(dfs.Team,rotation=90)
plt.show()


# In[ ]:





# In[32]:


sns.barplot(x=dfg.Team.dropna(),y=dfg.Medal.dropna())
plt.show()
sns.barplot(x=dfs.Team.dropna(),y=dfs.Medal.dropna())
plt.show()
sns.barplot(x=dfb.Team.dropna(),y=dfb.Medal.dropna())
plt.legend()
plt.show()


# In[33]:


s=np.arange(len(dfg))
g=s-.2
b=s+.2


# In[34]:


s,g,b


# In[35]:


plt.figure(figsize=(20,7))
a=np.arange(len(dfg))
plt.bar(x=g, height=dfg.Medal, width=0.2,alpha=0.4,color='Gold',label='Gold')
plt.bar(x=s,height=dfs.Medal,width=0.2,alpha=0.9,color='Silver',label='silver')
plt.bar(x=b,height=dfb.Medal,width=0.2,alpha=0.7,color='brown',label='Bronze')
plt.legend()
plt.xticks(dfs.Team,rotation=90)
plt.show()


# In[36]:


a+.2


# In[37]:


dfgg=df[df['Medal']=='Gold'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(20)


# In[38]:


dfgg


# In[39]:


dfg=df[df['Medal']=='Gold'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(25)
dfs=df[df['Medal']=='Silver'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(25)
dfb=df[df['Medal']=='Bronze'].groupby('Team').count()['Medal'].sort_values(ascending=False).reset_index().head(25)


# In[40]:


dff=pd.merge(dfg, dfs, how='left', on='Team')


# In[41]:


dff=pd.merge(dff,dfb,how='left')


# In[42]:


dff=dff.rename(columns={'Medal_x':'Gold','Medal_y':'Silver','Medal':'Bronze'})


# In[43]:


dff


# In[44]:


dff.set_index('Team')


# In[45]:


dff.plot(kind='bar',stacked=True)
plt.xlabel('Team')
plt.show()


# In[46]:


pd.pivot_table(df,index='Team',aggfunc={df.Medal=='Gold':'count',df.Medal=='Silver':'count'})


# In[47]:


k=pd.pivot_table(df, index='Team', aggfunc={'Medal': [['Gold', lambda x: sum(x == 'Gold')],['Silver', lambda x: sum(x == 'Silver')],['Bronze', lambda x: sum(x == 'Bronze')]]})#.sort_values(by=('Medal'),ascending=False)
k=k.sort_values(by=('Medal','Gold'),ascending=False)
k


# In[48]:


pivot_table = pd.pivot_table(df, index='Team', aggfunc={'Medal': ['count', lambda x: sum(x == 'Gold'), lambda x: sum(x == 'Silver'), lambda x: sum(x == 'Bronze')]})
sorted_pivot_table = pivot_table.sort_values(by=('Medal', 'count'), ascending=False)
sorted_pivot_table


# In[49]:


df=pd.read_csv('D:/DS/resume projects/athlete seaborn/athlete_events.csv')


# In[50]:


df1=df.groupby('Team').count().sort_values('Medal',ascending=False).head(20)


# In[51]:


df1=df1.sort_values('Medal',ascending=False).reset_index()


# In[52]:


df1['Team']


# In[53]:


dfb=df[(df['Medal']=='Bronze') & (df['Team'].isin(df1['Team']))].sort_values('Medal',ascending=False)
dfg=df[(df['Medal']=='Gold') & (df['Team'].isin(df1['Team']))].sort_values('Medal',ascending=False)
dfs=df[(df['Medal']=='Silver') & (df['Team'].isin(df1['Team']))].sort_values('Medal',ascending=False)


# In[54]:


dfg


# In[55]:


dfg=dfg.groupby('Team').count()['Medal'].reset_index()
dfs=dfs.groupby('Team').count()['Medal'].reset_index()
dfb=dfb.groupby('Team').count()['Medal'].reset_index()


# In[56]:


dfg


# In[57]:


dfg=dfg.sort_values('Medal',ascending=False).reset_index()
dfs=dfs.sort_values('Medal',ascending=False).reset_index()
dfb=dfb.sort_values('Medal',ascending=False).reset_index()


# In[58]:


dfs


# In[59]:


dfb


# In[60]:


dfg


# In[61]:


dt=pd.DataFrame({'Team':df1['Team'],'Gold':dfg['Medal'],'Silver':dfs['Medal'],'Bronze':dfb['Medal']})


# In[62]:


dt


# In[63]:


dt.sort_values(by='Gold',ascending=False)


# In[64]:


dt=dt.set_index('Team')
dt.plot(kind='bar',stacked=True)
plt.show()


# In[65]:


dt


# In[66]:


l=df.groupby('Team').count()['Medal'].sort_values(ascending=False).head(5).reset_index()


# In[67]:


ll=df[(df['Team'].isin(l.Team)) & (df.Season=='Summer')]


# In[68]:


l1=ll.groupby(['Team','Year']).count()['Medal'].sort_values(ascending=True).reset_index()


# In[69]:


l1


# In[70]:


sns.lineplot(x=l1.Year,y=l1.Medal,hue=l1.Team)
plt.show()


# In[107]:


df


# In[72]:


dfi=df[df['Team']=='India']


# In[73]:


dfi=dfi.groupby(['Year']).count()['Medal'].sort_values(ascending=True).reset_index()


# In[74]:


plt.figure(figsize=(10,5))
sns.barplot(x=dfi.Year,y=dfi.Medal,color='red')
plt.xticks(rotation=90)
plt.show()


# In[4]:


df=pd.read_csv('D:/DS/resume projects/athlete seaborn/athlete_events.csv')


# In[5]:


df2=df


# In[6]:


df2.isna().sum().sum()


# In[7]:


l=[]
for i in df2.columns:
    if df2[i].isna().sum()>0:
        l.append(i)


# In[8]:


l


# In[9]:


df2[l].isna().sum().plot(kind='bar')


# In[10]:


df['Medal'] = df['Medal'].replace({'Gold': 1, 'Silver': 1, 'Bronze': 1})
df['Medal'].fillna(0, inplace=True)


# In[11]:


df['Medal'].value_counts()


# In[55]:


df


# In[12]:


df2['Age'].plot(kind='kde')
plt.xlim(10, 50)
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age Distribution')
plt.show()


# In[139]:


df2['Weight'].plot(kind='kde')
plt.xlim(30, 130)
plt.xlabel('Weight')
plt.ylabel('Density')
plt.title('Weight Distribution')
plt.show()


# In[140]:


df2['Height'].plot(kind='kde')
plt.xlim(140, 210)
plt.xlabel('Height')
plt.ylabel('Density')
plt.title('Height Distribution')
plt.show()


# In[13]:


for i in l:
    df2[i] = df2[i].fillna(df[i].mode().iloc[0])


# In[14]:


df2.isna().sum().sum()


# In[15]:


df2


# In[16]:


df=df2


# In[17]:


X = df.drop(['Medal','ID','Name','City','Games','Year'], axis=1)
y = df['Medal']


# # Label Encoding 

# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Team'] = le.fit_transform(X['Team'])
X['Sport'] = le.fit_transform(X['Sport'])
X['Event'] = le.fit_transform(X['Event'])
X['NOC'] = le.fit_transform(X['NOC'])


# # One Hot Encoding

# In[19]:


X = pd.get_dummies(X, drop_first=True, sparse=True)


# # Train Test Split

# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Create the DecisionTreeClassifier instance with balanced class weights
dt = DecisionTreeClassifier(class_weight='balanced')

# Define the parameter grid with reduced ranges
param_dist = {
    'max_depth': [5, 10, 15, None],  # A smaller range for max_depth
    'min_samples_split': np.arange(2, 8, 2),  # Fewer values for min_samples_split
    'min_samples_leaf': np.arange(2, 5),  # Fewer values for min_samples_leaf
    'max_features': [0.2, 0.3, 0.5, 0.7],
}

# Create the RandomizedSearchCV instance
random_search = RandomizedSearchCV(dt, param_distributions=param_dist, cv=5, n_iter=20, n_jobs=-1)

# Perform the random search on your dataset
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


# # 1.Hyper Parameter Tunning
# 

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the updated parameter grid with more options
param_dist = {
    'max_depth': [None, 5, 10, 15, 20],  # More options for max_depth
    'min_samples_split': np.arange(2, 10, 2),  # Expanded range for min_samples_split
    'min_samples_leaf': np.arange(2, 6),  # Expanded range for min_samples_leaf
    'max_features': [0.2, 0.3, 0.5, 0.7, 0.9],  # More options for max_features
}

# Create the DecisionTreeClassifier instance with balanced class weights
dt = DecisionTreeClassifier(class_weight='balanced')

# Create the RandomizedSearchCV instance with more iterations
random_search = RandomizedSearchCV(dt, param_distributions=param_dist, cv=5, n_iter=50, n_jobs=-1)

# Perform the random search on your dataset
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the random search
best_model = random_search.best_estimator_

# Predict on the test set (assuming you have a separate test set)
y_pred = best_model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # 2.Hyperparameter tuning( Computationally expensive)

# In[24]:


from sklearn.ensemble import RandomForestClassifier

# Create the RandomForestClassifier instance with balanced class weights
rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

# Define a smaller parameter grid with reduced options
param_dist = {
    #'n_estimators': [20, 25],
    'max_depth': [20, 50, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'max_features': [0.5, 0.7],
}

# Create the RandomizedSearchCV instance with fewer iterations
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=5, n_iter=10, n_jobs=-1)

# Perform the random search on a subset of your dataset
subset_size = 5000  # Adjust this size as needed
random_search.fit(X_train[:subset_size], y_train[:subset_size])

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the random search
best_model = random_search.best_estimator_

# Now fit the best model on the entire training set (optional)
best_model.fit(X_train, y_train)

# Predict on the test set (assuming you have a separate test set)
y_pred = best_model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # 3. Hyperparameter tuning( Computationally expensive code)

# In[25]:


# from sklearn.ensemble import RandomForestClassifier

# # Create the RandomForestClassifier instance with balanced class weights
# rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

# # Define the parameter grid for RandomizedSearchCV
# param_dist = {
#     'n_estimators': [10, 20],  # Number of trees in the forest
#     'max_depth': [10, 20, 30, None],  # Regularization: More options for max_depth
#     'min_samples_split': np.arange(2, 10, 2),  # Regularization: Expanded range for min_samples_split
#     'min_samples_leaf': np.arange(1, 6),  # Regularization: Expanded range for min_samples_leaf
#     'max_features': ['auto', 'sqrt', 0.2, 0.5, 0.7],  # Different options for max_features
# }

# # Create the RandomizedSearchCV instance with more iterations and folds
# random_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=10, n_iter=50, n_jobs=-1)

# # Perform the random search on your dataset
# random_search.fit(X_train, y_train)

# # Print the best hyperparameters and corresponding score
# print("Best Hyperparameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)

# # Get the best model from the random search
# best_model = random_search.best_estimator_

# # Predict on the test set (assuming you have a separate test set)
# y_pred = best_model.predict(X_test)

# # Create the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix heatmap
# labels = np.unique(y_test)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()


# # Random Forest 

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Create the RandomForestClassifier instance with balanced class weights
rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Initial Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # Hyperparameter In RF

# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Create the RandomForestClassifier instance with balanced class weights
rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 6),
    'max_features': [0.5, 0.6, 0.7, 0.8],
}

# Create the RandomizedSearchCV instance with a reasonable number of iterations
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=5, n_iter=20, n_jobs=-1)

# Perform the random search on the full dataset
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the random search
best_model = random_search.best_estimator_

# Predict on the test set (assuming you have a separate test set)
y_pred = best_model.predict(X_test)

# Calculate the accuracy score for the best model
accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # Logistic regression 

# In[28]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Create the LogisticRegression instance
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Initial Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[29]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Create the LogisticRegression instance
logreg = LogisticRegression()

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'C': np.logspace(-3, 3, 7),  # Vary C from 0.001 to 1000
}

# Create the RandomizedSearchCV instance with a reasonable number of iterations
random_search = RandomizedSearchCV(logreg, param_distributions=param_dist, cv=5, n_iter=20, n_jobs=-1)

# Perform the random search on the full dataset
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the random search
best_model = random_search.best_estimator_

# Predict on the test set (assuming you have a separate test set)
y_pred = best_model.predict(X_test)

# Calculate the accuracy score for the best model
accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




