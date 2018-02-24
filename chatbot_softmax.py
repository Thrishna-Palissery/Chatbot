# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:19:02 2017

@author: Thrishna
"""


import os
import tensorflow as tf
import random
import numpy as np
import response as rp
import datapreprocess as dp
from sklearn.model_selection import train_test_split
import math

directory = 'dataset'
logs_path = '.\logs'

def create_dataset():
    '''Creates the dataset required for training the data
    returns datasets'''
    
    
    label_array = []
    count = 0
    sentences = []
    global uwLength
    global uniqueWords
    for filename in os.listdir(directory):
       filename = os.path.join(".\dataset", filename)
       with open(filename,"r") as file:
            print(filename)
            for example_sent in file:
                label = [0] * 10
                example_sent = example_sent.lower()
                label[count] = 1
                filtered_sentence = dp.data_preproccess(example_sent)
                sentences.append(filtered_sentence)
                label_array.append(label)
            count = count+1           
    uniqueWords,uwLength = dp.create_unique_words(sentences)
    #print(uniqueWords)                
    bagwords,_ = dp.create_bag_of_words(sentences,uniqueWords)
    bagwords = np.array(bagwords)
    label = np.array(label_array)
    return bagwords,label_array



def train_model():
    '''Create model and train the model using dataset bagwords and label
    @return: returns the updated weights and biases'''
    biases = []
    weights = []
    graph1 = tf.Graph()
    with graph1.as_default():   
        bagwords,label = create_dataset()
        X_train, X_test, y_train, y_test = train_test_split(bagwords, label, test_size=0.20, random_state=42)
        with tf.variable_scope('Input'):
            X = tf.placeholder(tf.float32, [None, uwLength], name='feature')
            Y = tf.placeholder(tf.float32, [None,10], name='label')
        with tf.variable_scope('Hidden_Layer1'):
            w1 = tf.get_variable("w1", shape=[uwLength,10],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
            b1 = tf.Variable(tf.random_normal([10]), name='b1')
            layer1 = tf.nn.relu(tf.matmul(X, tf.cast(w1,tf.float32)) + b1)
        with tf.variable_scope('Hidden_Layer2'):
            w2 = tf.get_variable("w2", shape=[10,10],initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            b2 = tf.Variable(tf.random_normal([10]), name='b2')
            hypothesis = tf.nn.softmax(tf.matmul(layer1, w2) + b2)
        # Test model 
        is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1)) 
        # Calculate accuracy 
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        #Calculate coste
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
           
        #Train data
        with tf.Session(graph=graph1) as sess:
            summary_op =  tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(logs_path, sess.graph)
            writer.add_graph(sess.graph)
            acc_t = 0
            for step in range(1000):
                cost_val, _,we1,be1,we2,be2,acc = sess.run([cost, optimizer,w1,b1,w2,b2,accuracy], feed_dict={X: X_train, Y: y_train})
                acc_t += acc
            print("Accuracy with training data",acc_t/1000)  
            weights.append(we1)
            weights.append(we2)
            biases.append(be1)
            biases.append(be2)
            test_acc = sess.run([accuracy],feed_dict={X: X_test, Y: y_test})
        print("Accuracy with test data:",(test_acc))
        print(tf.__version__)
        return weights,biases
 
       
def get_respone(command,weights,biases):   
    '''Predict the response for a given command
    @return: return the predicted response''' 
    
    
    input_sentence = []
    course_flag = 0
    #To handle fallback
    course = ['cmpe297','cmpe 297-11','deep learning','cmpe 297', 'cmpe297-11','computer engineering 297','computer engineering 297-11','Simon Shim']
    for i in course:
        if(i in command):
            course_flag = 1
    #Pre-process the sentence
    filtered_input = dp.data_preproccess(command)
    input_sentence.append(filtered_input)
    #Create-bag of words
    input_bag,flag = dp.create_bag_of_words(input_sentence,uniqueWords)
    
    #predict the response
    with tf.Session() as sess:
        #Check whether if the user question is a fallback case
        if flag != 0 or course_flag != 0:
            input_bag = np.array(input_bag)
            X = tf.placeholder(tf.float32, [None, uwLength], name='feature')
            layer1 = tf.nn.relu(tf.matmul(X, weights[0]) + biases[0])
            logits = tf.nn.softmax(tf.matmul(layer1, weights[1]) + biases[1])
            predict = tf.arg_max(logits, 1)
            test_predict = sess.run(predict, feed_dict={X: input_bag})
            for i in test_predict:
                pred = i
            response = rp.get_response(pred)
            return random.choice(response)
        #Call seq2seq to handle fallback case
        else:
            with open('temp_user_input.txt', 'w') as fb:
                fb.write(command)  # Save user input to file for chatbot
            os.system('python chatbot.py')
            with open('temp.txt') as fb:
                response = fb.read()
            open('temp.txt', 'w').close()
            #print(response)
            return response
    
    
    

                

