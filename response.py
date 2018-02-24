# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:02:19 2017

@author: Thrishna
"""

out = []
output0 = ["good bye","see you later","Bye bye"]
output1 = ['Deep learning laboratory meeting is on Tuesday from 4:00-5:45pm. There are two labs for the class. Lab1 is based on ANN/DNN ,Lab2 is based on CNN/RNN', 'cmpe297 laboratory is on 4:00-5:45pm on Tuesdays . There are two labs for the class. Lab1 is based on ANN/DNN ,Lab2 is based on CNN/RNN']
output2 = ["Deep learning lecture is on every tuesday at 3:00-5:45 pm. Lecture room number is 407 of health care building","CMPE 297 class is on every tuesday at 3:00-5:45 pm. Lecture room number is 407 of health care building"]
output3 = ["There will be surprise quizzes and two exams. Midterm exam is on October 17 2017 and covers topic related to ANN.  Deep Learning final exam is a comprehensive exam which is is planned for December 14,2017 from 14:45 – 17:00. Good luck"]
output4 = ["whats up","Hello","Hey there"]
output5 = ["You are welcome","welcome"]
output6 = ["cmpe 297 is deep learning course. It covers techniques involved in programming CNN,ANN and RNN. Greensheet and other details including the required textbooks for the course are available on SJSU CANVAS ","Deep learning course covers techniques involved in programming CNN,ANN and RNN.Greensheet and other details  including the required textbooks for the course are available on SJSU CANVAS"]
output7 = ["Prof.Shim's and teaching assisstants office hours  are on Monday from 2:30pm - 4:00 pm"]
output8 = ["Dr.Simon Shim is the Professor for deep learning course.Abhi Ram Reddy Salammagari and Srivatsa Mulpuri are the Teaching assistants. Here is the Professor's profile http://www.sjsu.edu/people/simon.shim/","Dr.Simon Shim is the Professor for cmpe 297-11 course.Abhi Ram Reddy Salammagari and Srivatsa Mulpuri are the Teaching assistants"]
output9 = ["Deep learning project topic is building chatbot. It's due on December 5,2017. Rubrics and other details are available on SJSU CANVAS","CMPE 297_11 project topic is building chatbot. It's due on December 5,2017. Rubrics and other details are available on SJSU CANVAS"]

def get_response(index):
    out.append(output0)
    out.append(output1)
    out.append(output2)
    out.append(output3)
    out.append(output4)
    out.append(output5)
    out.append(output6)
    out.append(output7)
    out.append(output8)
    out.append(output9)
    return out[index]