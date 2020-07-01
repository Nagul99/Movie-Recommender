# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:38:59 2019

@author: Nagul
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London","Paris Paris London"]

cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
print(count_matrix.toarray())

similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)
