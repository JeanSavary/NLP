# -*- coding: utf-8 -*-
'''
TF_IDF.py
by Jean Savary
Date : 22/08/2018
'''

import os
import pandas as pd
import numpy as np
import math
import string

os.chdir("/Users/jeansavary/Desktop/Codes/DataBases")
database = pd.read_csv("articles.csv")

#Storing every articles' title into a 1 column dataframe
articles_title = database["title"]

#Defining our main function
'''
description : The main function which will suggest a consistent article suggestion according to the query
inputs : query (String), corpus (DataFrame of Strings)
output : suggestion (String)
'''
def query_retrieval(query, corpus):

    words_in_query = query.lower().split() #Create a list containing all the word of the query (non considering spaces)
    frequencies_of_word_in_query = {}
    tf_idf_values_for_word = {}

    for word in words_in_query :

        frequencies_of_word_in_query[word] = [] #Creating a dictionnary containing the frequencies of appearances of the word in each title of the corpus
        tf_idf_values_for_word[word] = [] #Creating a dictionnary containing the tf_idf values of the word for each title of the corpus
        number_of_word_in_corpus = 0

        for title in corpus :
            
            splited_title = title.split()
            words_in_title = [w.rstrip(string.punctuation).lower() for w in splited_title] #We convert every title's character to lowercase and erase punctuation
            number_of_word_in_corpus += words_in_title.count(word) #We keep updating the cumulative sum of the occurence of word in the whole corpus
            frequencies_of_word_in_query[word].append(words_in_title.count(word)/len(words_in_title)) #We keep updating the values of the frequencies in the dictionnary for each word of the queries

        for index in range(len(corpus)):

            tf_idf_values_for_word[word].append(frequencies_of_word_in_query[word][index]*math.log10(len(corpus)/number_of_word_in_corpus))

    calculation_matrix = np.array([list_of_tf_idf for list_of_tf_idf in tf_idf_values_for_word.values()]) #We built a matrix composed of the list of tf_idf values (number of columns) for each word of the query (number of rows)
    sum_column_tf_idf = np.sum(calculation_matrix, axis=0)
    index_max_tf_idf = max(xrange(len(sum_column_tf_idf)), key=values.__getitem__)

    return corpus[index_max_tf_idf]

