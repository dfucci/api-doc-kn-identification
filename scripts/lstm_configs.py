#!/usr/bin/env python3
# -*- coding: utf-8 -*-

MAX_SEQUENCE_LENGTH = 800
MAX_NB_WORDS = 20000

EMBEDDING_DIM = 50 #300
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.10
K=10 #10
EPOCHES=100 #100
OVER_SAMPLE_NUM = 800
UNDER_SAMPLE_NUM = 500

ACTIVATION_M = 'sigmoid' 
LOSS_M = 'binary_crossentropy'
OPTIMIZER_M = 'adam'

ACTIVATION_S = 'softmax' 
LOSS_S = 'categorical_crossentropy'
OPTIMIZER_S = 'rmsprop'

head_short=['Func.','Conc.', 'Dire.', 'Purp.', 'Qual.', 'Cont.', 'Stru.', 'Patt.', 'Exam.', 'Envi.', 'Refe.', 'Non-Inf.']
head_medica = ['Class-0-593_70,Class-1-079_99,Class-2-786_09,Class-3-759_89,Class-4-753_0,Class-5-786_2,Class-6-V72_5,Class-7-511_9,Class-8-596_8,Class-9-599_0,Class-10-518_0,Class-11-593_5,Class-12-V13_09,Class-13-791_0,Class-14-789_00,Class-15-593_1,Class-16-462,Class-17-592_0,Class-18-786_59,Class-19-785_6,Class-20-V67_09,Class-21-795_5,Class-22-789_09,Class-23-786_50,Class-24-596_54,Class-25-787_03,Class-26-V42_0,Class-27-786_05,Class-28-753_21,Class-29-783_0,Class-30-277_00,Class-31-780_6,Class-32-486,Class-33-788_41,Class-34-V13_02,Class-35-493_90,Class-36-788_30,Class-37-753_3,Class-38-593_89,Class-39-758_6,Class-40-741_90,Class-41-591,Class-42-599_7,Class-43-279_12,Class-44-786_07']

