# -*- coding: utf-8 -*-
'''
TF_IDF.py
by Jean Savary
Date : 23/08/2018
'''

import os
import pandas as pd
import numpy as np
import string
import math

os.chdir("/Users/jeansavary/Desktop/Codes/DataBases")
database = pd.read_csv("articles.csv")

# -- Storing every articles' title into a 1 column dataframe -- #

articles_title = database["title"]
list_of_words_in_titles = []
translation_table = str.maketrans(dict.fromkeys(string.punctuation + '“‘”🤖👶🔮🔨'))

# -- Pre-processing database, collecting pre-processed data in a dictionnary {film_id : [words]} -- #

def pre_processing(database):

    global list_of_words_in_titles    # We will construct this list while going through our loop
    global translation_table
    processed_database = {}
    article_id = 0

    for title in database :

        splited_title = title.split()
        processed_words = [text.translate(translation_table).lower() for text in splited_title]
        processed_database[article_id] = processed_words
        list_of_words_in_titles += processed_words
        article_id += 1

    return processed_database

pre_processed_database = pre_processing(articles_title)
#print(pre_processed_database)

# -- Creating the list of all the single words present in titles
# -- Also creating the matrix of tf_idf values for the corpus, dimension : (number of titles * number of unique words)

list_of_unique_words_in_titles = set(list_of_words_in_titles)
number_of_times_a_unique_word_appears = dict.fromkeys(list_of_unique_words_in_titles, 0)    #  Creating a dict {word1 : number of time it appears in corpus, word2 : ...}
tf_idf_values_for_matrix = []                       #  This list will contain the list of tf_idf_values for each title

for word in list_of_unique_words_in_titles : 

    number_of_times_a_unique_word_appears[word] = list_of_words_in_titles.count(word)
    word_tf_idf_values = []

    for title in pre_processed_database.values(): 

        word_tf_idf_values.append((title.count(word)/len(title)) * math.log10(len(pre_processed_database)/number_of_times_a_unique_word_appears[word]))
    
    tf_idf_values_for_matrix.append(word_tf_idf_values)

tf_idf_matrix = np.column_stack(tf_idf_values_for_matrix)     #  We create the matrix by stacking horinzontaly every list of tf_idf values in "tf_idf_values_for_matrix" 
    
print("\n-----------------------------")
print("\nNombre de mots au total : " + str(len(list_of_words_in_titles)))
print("Nombre total de mots uniques : " + str(len(list_of_unique_words_in_titles))+"\n")
#print(len(number_of_times_a_unique_word_appears))
#print(np.sum(tf_idf_matrix, axis=1))#Sum on the lines

#Collecting a query from the user and pre-processing it
print('-----------------------------')
original_query = input('Entrer votre requête : ')
processed_query = [text.translate(translation_table).lower() for text in original_query.split()]
#print(processed_query)

#  Creating a one-hot array of the query

one_hot_query_array = [0]*len(list_of_unique_words_in_titles)
for word in processed_query :
    if (word in list_of_unique_words_in_titles) :
        one_hot_index = list(list_of_unique_words_in_titles).index(word)
        one_hot_query_array[one_hot_index] = 1

if (one_hot_query_array == [0]*len(one_hot_query_array)):
    print("\nSorry there is no result for your specific query... Let's try anything else !")

#   Dot product between tf_idf_matrix and one_hot_query

consistency_matrix = np.dot(tf_idf_matrix,one_hot_query_array)
print(consistency_matrix.max(axis=0))


# list_titles = [{'original_title':'', 'cleaned_title':[]}]
# list_titles[article_id]['original_title'] 