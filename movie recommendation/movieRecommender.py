# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:38:59 2019

@author: Nagul
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
##Helper functions
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

#Read CSV File
df = pd.read_csv("movie_dataset.csv")
print(df.columns)

#Select Features
features = ['keywords','cast','genres','director']

#Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')
    
def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print('Error:',row)

df["combined_features"] = df.apply(combine_features,axis=1)
print("Combined features:",df["combined_features"].head())   
     
#Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])


#Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

#Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

#Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse=True)

#Print titles of first 50 movies

i=0
print("Similar movies like "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break