#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import zscore

# Hva vil du modellen skal gjøre:
# Basert på tags og genres skal modellen finne hva slags filmer en bruker har 
# sett på, og foreslå filmer med like tags og genres med høy rating


# In[2]:



genomeS = pd.read_csv("genome-scores.csv")
genomeT = pd.read_csv("genome-tags.csv")
link = pd.read_csv("links.csv")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")


# In[3]:


print(tags.shape, genomeS.shape, genomeT.shape, link.shape, movies.shape, ratings.shape)


# In[4]:


genres = []
m = movies["genres"]
for movie in m:
    for genre in movie.split("|"):
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    newColumn = []
    for movie in m:
        if genre in movie.split("|"):
            newColumn.append(1)
        else:
            newColumn.append(0)
    movies[genre] = newColumn

    
newMovies = movies.drop(columns = "genres")
df = tags.merge(newMovies, on="movieId")
newRatings = ratings.drop(columns = "timestamp")
df.drop(columns = ["timestamp", "tag", "title"], inplace = True)
df = df.sort_values("userId")
df


# In[28]:



#splitte opp dataframen min til #userId dataframes og lage
df = df.reset_index(drop = True)


def mm(df):
    models = []
    users = df["userId"]
    pred = []
    score = []
    for i in range(df["userId"].max()):
        if df[df["userId"] == i].shape[0] != 0:
            X = df[df["userId"] == i].drop(columns = ["movieId"])
            y = df[df["userId"] == i]["movieId"]
            X = X.to_numpy()
            y = y.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            
            model = RandomForestClassifier()
            models.append((i, model.fit(X_train, y_train)))
            tempPred = model.predict(X_test)
            pred.append((i, tempPred))
            score.append((i, accuracy_score(y_test, tempPred)))

    return model, pred, score
# X = df[df["userId"] == 14].drop(columns = ["movieId"])
# y = df[df["userId"] == 14]["movieId"]
# X = X.to_numpy()
# y = y.to_numpy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# score = accuracy_score(y_test, pred)
# score
model, pred, score = mm(df)


# In[ ]:





# In[ ]:





# In[ ]:




