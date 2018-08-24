# -*- coding: utf-8 -*-
'''
TF_IDF.py
by Jean Savary
Date : 23/08/2018
'''

# -- Importing librairies -- #

import os
import pandas as pd
import numpy as np
import string
import math

# -------------------------- #

# -- Defining the current directory for our databases -- #

os.chdir("/Users/jeansavary/Desktop/Codes/DataBases")

# ------------------------------------------------------ #

# -- Here is all of our methods -- #
'''
return : processed string
'''
def pre_processing(text): 

    splited_text = text.split()    # To be changed !
    translation_table = str.maketrans(dict.fromkeys(string.punctuation + '“‘”🤖👶🔮🔨'))
    processed_text = [text.translate(translation_table).lower() for text in splited_text]

    list_of_elements_to_be_removed = ['–','—','–','']
    processed_text = list(set(processed_text).difference(set(list_of_elements_to_be_removed)))
    
    for word in processed_text : 
        if ("’s" in word):
            processed_text.remove(word)
            word = word.replace("’s","")
            processed_text.append(word)

    return processed_text

'''
return : processed database --> [{"original_text" : "...", "cleaned_text" : ["..","..",...,"..."]}]
'''
def database_pre_processing (database_name, column_of_interest):  # Must include ".csv"

    raw_database = pd.read_csv(database_name)[column_of_interest]
    processed_database = []
    
    for raw_text in raw_database :

        processed_text = pre_processing(raw_text)
        processed_database.append({"original_text" : raw_text, "cleaned_text" : processed_text})

    return processed_database

'''
return : list of unique words --> [word1, word2, ...]
'''
def create_a_set_of_all_word (processed_database):

    list_of_all_words = []    # List containing ALL words !
    for dictionnary in processed_database:
        list_of_all_words += dictionnary["cleaned_text"]
    
    return set(list_of_all_words)        # List of unique words

'''
return : the tf_idf_matrix of the corpus (number of articles * number of unique words)
'''
def compute_idf_matrix (database_name = 'articles.csv', column_of_interest = 'title'):
    
    processed_database = database_pre_processing(database_name, column_of_interest)
    list_of_unique_words_in_titles = create_a_set_of_all_word(processed_database)
    number_of_times_a_unique_word_appears = {word : 0 for word in list_of_unique_words_in_titles}   # Creating a dict {word1 : number of time it appears in different lines of the corpus, word2 : ...}
    words_frequencies_in_title = {word : [] for word in list_of_unique_words_in_titles}      # Creating a dict {word1 : [frequency of word1 in line1], ...
    tf_idf_values_for_matrix = []

    for word in list_of_unique_words_in_titles :

        for dictionnary in processed_database :

            words_frequencies_in_title[word].append(dictionnary['cleaned_text'].count(word)/len(dictionnary['cleaned_text']))

            if (word in dictionnary['cleaned_text']) :

                number_of_times_a_unique_word_appears[word] += 1    # Each time a word appear in a title we increment this value
        
        word_tf_idf_values = [freq * math.log10(len(processed_database)/number_of_times_a_unique_word_appears[word]) for freq in words_frequencies_in_title[word]]
        tf_idf_values_for_matrix.append(word_tf_idf_values)
    
    tf_idf_matrix = np.column_stack(tf_idf_values_for_matrix)     #  We create the matrix by stacking horinzontaly every list of tf_idf values in "tf_idf_values_for_matrix" 

    return tf_idf_matrix
    

'''
return : the title (String) of the most relevant article according to the query
'''
def find_most_relevant_article(query, tf_idf_matrix, set_of_unique_words, processed_database) :

    processed_query = pre_processing(query)
    one_hot_query_array = [0] * len(set_of_unique_words)

    for word in processed_query :

        if (word in set_of_unique_words) :

            one_hot_index = list(set_of_unique_words).index(word)    #We have to convert the set into a list to apply .index()
            one_hot_query_array[one_hot_index] = 1

    if (one_hot_query_array == [0] * len(one_hot_query_array)):

        return "Sorry there is no result for your specific query... Let's try anything else !"

    else :

        consistency_matrix = np.dot(tf_idf_matrix,one_hot_query_array)
        max_consistency_score = consistency_matrix.max(axis=0)   #Checking the maximum consistency score per title
        index_of_most_relevant_article = np.argmax(consistency_matrix)

        #print(consistency_matrix[index_of_most_relevant_article] == max_consistency_score)     #Checking the validity of the previous line
        most_relevant_article = processed_database[index_of_most_relevant_article]['original_text']

        return most_relevant_article

# --------------------------------- #

# -- Main area -- #
processed_database = database_pre_processing("articles.csv","title")
set_of_all = create_a_set_of_all_word(processed_database)
matrix = compute_idf_matrix()

raw_query = input('\nEntrer votre requête :        ')
print("\n")
answer = find_most_relevant_article(raw_query, matrix, set_of_all, processed_database)

print("----------------------------------------------------------------------------")
print("Here is the most relevant article according to your request : \n")
print("         ----> " + answer)
print("----------------------------------------------------------------------------")
print("\n")

# -------------- #