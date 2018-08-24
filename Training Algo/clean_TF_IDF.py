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

# -- Defining the current directory for our databases -- #
os.chdir("/Users/jeansavary/Desktop/Codes/DataBases")

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

    print(processed_text)
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


processed_database = database_pre_processing("articles.csv","title")
