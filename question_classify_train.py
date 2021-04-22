#!/usr/bin/env python3
# coding: utf-8
# File: question_classify.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10
import sys
#sys.path.append("/home/songhan/miniconda3/envs/py36_for_CrimeClassify/lib/python3.6/site-packages")
import os
import random
import numpy as np
import jieba.posseg as pseg
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from keras import optimizers
class QuestionClassify(object):
    def __init__(self):
        self.label_dict = {
            0:"婚姻家庭",
            1:"劳动纠纷",
#            2:"交通事故",
#            3:"债权债务",
#            4:"刑事辩护",
#            5:"合同纠纷",
#            6:"房产纠纷",
#            7:"侵权",
#            8:"公司法",
#            9:"医疗纠纷",
#            10:"拆迁安置",
#           11:"行政诉讼",
#            12:"建设工程"
            }
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_file = os.path.join(cur, 'data/1w_question_train.txt')
        self.embedding_path = os.path.join(cur, 'embedding/word_vec_300.bin')
        self.embdding_dict = self.load_embedding(self.embedding_path)
        self.max_length = 60
        self.embedding_size = 300
        self.dense=10
        self.lstm_modelpath = 'model/lstm_question_classify.h5'
        self.cnn_modelpath = 'model/cnn_question_classify.h5'
        return

    '''加载词向量'''
    def load_embedding(self, embedding_path):
        embedding_dict = {}
        count = 0
        for line in open(embedding_path,encoding="utf-8"):
            line = line.strip().split(' ')
            if len(line) < 300:
                continue
            wd = line[0]
            vector = np.array([float(i) for i in line[1:]])
            embedding_dict[wd] = vector
            count += 1
            if count%10000 == 0:
                print(count, 'loaded')
        print('loaded %s word embedding, finished'%count, )
        return embedding_dict

    '''对文本进行分词处理'''
    def seg_sent(self, s):
        wds = [i.word for i in pseg.cut(s) if i.flag[0] not in ['w', 'x']]
        #print(s)
        #print(wds)
        return wds

    '''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
    def rep_sentencevector(self, sentence):
        word_list = self.seg_sent(sentence)[:self.max_length]
        embedding_matrix = np.zeros((self.max_length, self.embedding_size))
        for index, wd in enumerate(word_list):
            if wd in self.embdding_dict:
                embedding_matrix[index] = self.embdding_dict.get(wd)
            else:
                continue
        len_sent = len(word_list)
        embedding_matrix = self.modify_sentencevector(embedding_matrix, len_sent)
        #print(embedding_matrix[:2])
        return embedding_matrix

    '''对于OOV词,通过左右词的词向量作平均,作为词向量表示'''
    def modify_sentencevector(self, embedding_matrix, len_sent):
        context_window = 2
        for indx, vec in enumerate(embedding_matrix):
            left = indx-context_window
            right = indx+context_window
            if left < 0:
                left = 0
            if right > len(embedding_matrix)-1:
                right = -2
            context = embedding_matrix[left:right+1]
            if vec.tolist() == [0]*300 and indx < len_sent:
                context_vector = context.mean(axis=0)
                embedding_matrix[indx] = context_vector

        return embedding_matrix

    '''对数据进行onehot映射操作'''
    def label_onehot(self, label):
        one_hot = [0]*len(self.label_dict)
        one_hot[int(label)] = 1
        return one_hot

    '''加载数据集'''
    def load_traindata(self):
        train_X = []
        train_Y = []
        count = 0
        for line in open(self.train_file,encoding="utf-8"):

            line = line.strip().strip().split('##')
            if len(line) < 2:
                continue
            count += 1
            sent = line[0]
            label = line[1]
            sent_vector = self.rep_sentencevector(sent)
            label_vector = self.label_onehot(label)
            train_X.append(sent_vector)
            train_Y.append(label_vector)

            if count % 10000 == 0:
                print('loaded %s lines'%count)
        return np.array(train_X), np.array(train_Y)

    '''构造CNN网络模型'''
    def build_cnn_model(self):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(self.max_length, self.embedding_size)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(self.dense, activation='softmax'))
        opt=optimizers.rmsprop(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      #optimizer='rmsprop',
                      optimizer=opt,
                      metrics=['accuracy'])
        model.summary()
        return model

    # 构造LSTM网络
    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(self.max_length, self.embedding_size)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        model.add(Dropout(0.5))
        model.add(Dense(self.dense, activation='softmax'))
        #opt=optimizers.rmsprop(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        model.summary()
        return model

    '''训练CNN模型'''
    def train_cnn(self):
        X_train, Y_train, X_test, Y_test = self.split_trainset()
        model = self.build_cnn_model()
        history =model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_data=(X_test, Y_test))
        model.save(self.cnn_modelpath)
        self.draw_pic(history,'CNN')

    '''训练lstm模型'''
    # batch_size:每次使用的样本数
    # epochs:训练整个样本集的数量
    def train_lstm(self):
        X_train, Y_train, X_test, Y_test = self.split_trainset()
        model = self.build_lstm_model()
        history = model.fit(X_train, Y_train, batch_size=150, epochs=10, validation_data=(X_test, Y_test))
        model.save(self.lstm_modelpath)
        self.draw_pic(history,'LSTM')
    # 画图
    def draw_pic(self,history,model):
        acc = history.history['acc']  # 获取训练集准确性数据
        val_acc = history.history['val_acc']  # 获取验证集准确性数据
        #val_acc=list(map(lambda x:x+0.18,val_acc))
        loss = history.history['loss']  # 获取训练集错误值数据
        val_loss = history.history['val_loss']  # 获取验证集错误值数据
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Trainning acc')  # 以epochs为横坐标，以训练集准确性为纵坐标
        plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # 以epochs为横坐标，以验证集准确性为纵坐标
        plt.legend()  # 绘制图例，即标明图中的线段代表何种含义
        plt.savefig("{}_{}_{}.png".format(self.train_file[:-4],model,"acc"))
        plt.figure()  # 创建一个新的图表
        plt.plot(epochs, loss, 'bo', label='Trainning loss')
        plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
        plt.legend()  ##绘制图例，即标明图中的线段代表何种含义
        plt.savefig("{}_{}_{}.png".format(self.train_file[:-4],model,"loss"))
        print("acc=",acc)
        print("val_acc=",val_acc)
        print("loss=",loss)
        print("val_loss=",val_loss)
        #plt.show()  # 显示所有图表

    '''划分数据集,按一定比例划分训练集和测试集'''
    def split_trainset(self):
        X, Y = self.load_traindata()
        X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=100)
        return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':

    handler = QuestionClassify()
    handler.train_file = "data/2fenlei16000_question_train.txt"
    handler.dense=2
    #handler.train_cnn()
    handler.train_lstm()
