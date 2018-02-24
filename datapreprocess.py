# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:06:11 2017

@author: Thrishna
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
lemmatizer = WordNetLemmatizer()

def data_preproccess(example_sent):
    ''' lemmatize, stem, tokenize, remove puctuations and remove stopwords from a sentence
    @param: sentence to preprocess
    @return: list of tokenized words'''
    example_sent = example_sent.lower()
    exclude = set(string.punctuation)
    example_sent = ''.join(ch for ch in example_sent if ch not in exclude)
    #print(example_sent)
    operators = set(('where', 'when', 'what','who'))
    stop_words = set(stopwords.words("english"))- operators
    words_tokens = word_tokenize(example_sent)
    filtered_sentence = []
    for w in words_tokens:
        if w not in stop_words:
            
            w = lemmatizer.lemmatize(w)
            w = SnowballStemmer("english").stem(w)
            filtered_sentence.append(w)
    #print(filtered_sentence)
    return filtered_sentence

def create_unique_words(sentences):
    '''Create unique words from a list of sentences
    @param: list of sentences
    @return: list of unique words and length of the list'''
    
    words = [] 
    for filtered_sentence in sentences:
        for i in filtered_sentence:
            if not i in words:
                words.append(i)
    return words, len(words)

def create_bag_of_words(sentences,uniqueWords):
    '''Creates bagofwords
    @param: list of sentences and list of unique words
    @return: list containing bag of words'''       
    bag = []  
    uw_c = 0
    n_c = 0
    flag = 0
    for filtered_sentence in sentences:
        bag_vec = [0] * len(uniqueWords)
        #count to check fallback case
        n_c = 0
        uw_c = 0
        for i in filtered_sentence:
            n_c = n_c + 1
            if(i in uniqueWords):
                uw_c = uw_c + 1
                bag_vec[uniqueWords.index(i)] = 1
        bag.append(bag_vec)
    #comparing the counts for identifying fallback
    if(n_c == uw_c):
        flag = 1
    return bag,flag
    
