#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


# In[4]:


print(tags.shape, genomeS.shape, genomeT.shape, link.shape, movies.shape, ratings.shape)
movies


# In[13]:


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
df
        


# In[ ]:




