from keras.models import Model
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import LSTM,Masking
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Input,Dense,Dropout,Lambda,merge,concatenate
from keras import initializers
from keras.layers.core import  Activation,Flatten, Reshape
from keras import backend as K
import helper
from keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import optimizers
import gc


nflod=7



## experiment list

## (1)impact of polarity and hydroph and positin

## (2)full connect VS LSTM
#   (2.1) test of full connect with different window size
#   (2.2) test of BLSTM with differejt window size
## (3)test of GL-BLSTM


def fullConnect(winSize,infoOfAA):
    model = Sequential()
    model.add(Dense(100, input_shape=(winSize*infoOfAA,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def LocalBLSTM(winSize, infoOfAA):

    input = Input((winSize, infoOfAA))
    mask = (Masking(mask_value=0, input_shape=(winSize, infoOfAA)))(input)
    hiddhen = Bidirectional(LSTM(40, activation='relu', dropout=0.2, recurrent_regularizer=regularizers.l2(0.005),
                                 kernel_regularizer=regularizers.l2(0.005),
                                 bias_regularizer=regularizers.l2(0.005), recurrent_dropout=0.2))(mask)
    output = Dense(2, activation='softmax')(hiddhen)
    model = Model(input=input, output=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def glblstm():
    InfoOfCys = 24
    hiddenUnit = 30
    wins = 7
    CysNum = 25
    inputOfSeq = Input((CysNum, wins, InfoOfCys))
    listOfCysRegion = helper.crop()(inputOfSeq)

    maskingLayerForLocalRegion = (Masking(mask_value=0, input_shape=(wins, InfoOfCys)))

    localBLSTM = Bidirectional(LSTM(hiddenUnit, activation='relu', kernel_initializer=initializers.glorot_normal()
                                    , recurrent_initializer=initializers.glorot_normal()))

    listOfCysRegionAfterMASK = list()
    for i in range(len(listOfCysRegion)):
        listOfCysRegionAfterMASK.append(maskingLayerForLocalRegion(listOfCysRegion[i]))

    hiddens = list()
    for i in range(len(listOfCysRegionAfterMASK)):
        hiddens.append(Reshape(target_shape=(1, hiddenUnit * 2))(localBLSTM(listOfCysRegionAfterMASK[i])))


    inputForGlobal = concatenate(hiddens, axis=1)

    inputForGlobalAfterMask = (Masking(mask_value=0, input_shape=(CysNum, hiddenUnit * 2)))(inputForGlobal)
    inputForGlobalAfterMaskAfterBn=BatchNormalization()(inputForGlobalAfterMask)

    globalBLSTM = Bidirectional(
        LSTM(hiddenUnit,activation='relu',return_sequences=1, dropout=0.2, recurrent_regularizer=regularizers.l2(0.005),
             kernel_regularizer=regularizers.l2(0.005),
             bias_regularizer=regularizers.l2(0.005), recurrent_dropout=0.2))(inputForGlobalAfterMaskAfterBn)

    output = TimeDistributed(Dense(2, activation='softmax'))(globalBLSTM)
    model = Model(input=[inputOfSeq], output=[output])

    model.compile(optimizer='adam',
                  loss={ 'time_distributed_1': 'binary_crossentropy'},
                  metrics=['acc',helper.acc_on_protein])
    return model





##this block test impact of polarity and hydroph,and Postion information
##results will be wroted into resource/testOnresidueDim
if 0:
    windowsize=51
    halfWin=windowsize//2
    X = np.load('../resource/GLBlstmTrain/X.data.npy')
    Ti = np.load('../resource/GLBlstmTrain/T.data.npy')

    ## pssm + hydro
    modelpath = '../MODEL/keras/pssm_hydro'
    Xi = np.zeros((X.shape[0], X.shape[1], windowsize, 21), np.float32)
    Xi[:, :, :, 0:20] = X[:, :, 50 - halfWin:51 + halfWin, 0:20]
    Xi[:, :, :, 20] = X[:, :, 50 - halfWin:51 + halfWin, 22]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        if j != nflod - 1:

            Xtest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = Ti[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((Ti.shape[0] - oneSplit, Ti.shape[1], Ti.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
        else:
            Xtest = Xi[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]
        model = LocalBLSTM(windowsize, 21)
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        ##train
        Xtrain, Ttrain = helper.convertTrainingData(Xtrain, Ttrain)
        callbacks_list = [checkpoint]
        model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                  , callbacks=callbacks_list, verbose=2)
        model.load_weights(modelpath)
        pr_t = model.predict(Xtest.reshape((Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2], Xtest.shape[3])))
        pr_t = pr_t.reshape((Ttest.shape[0], Ttest.shape[1], Ttest.shape[2]))
        result = helper.evaluateResult(Ttest, pr_t)
        print('pssm hydro  flod: ' + str(j))
        print(result)

        f = open('../resource/testOnresidueDim', 'a')
        f.writelines('pssm hydro split ' + str(j) + str(result) + '\n')
        f.close()

    ## pssm +position
    modelpath='../MODEL/keras/pssm_position'
    Xi = X[:, :, 50 - halfWin:51 + halfWin, 0:22]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        if j != nflod - 1:

            Xtest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = Ti[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((Ti.shape[0] - oneSplit, Ti.shape[1], Ti.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
        else:
            Xtest = Xi[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]
        model = LocalBLSTM(windowsize, 22)
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        ##train
        Xtrain, Ttrain = helper.convertTrainingData(Xtrain, Ttrain)
        callbacks_list = [checkpoint]
        model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                  , callbacks=callbacks_list, verbose=2)
        model.load_weights(modelpath)
        pr_t = model.predict(Xtest.reshape((Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2], Xtest.shape[3])))
        pr_t = pr_t.reshape((Ttest.shape[0], Ttest.shape[1], Ttest.shape[2]))
        result = helper.evaluateResult(Ttest, pr_t)
        print('pssm+postion:  flod: ' + str(j))
        print(result)

        f = open('../resource/testOnresidueDim', 'a')
        f.writelines('pssm postion split ' + str(j) + str(result) + '\n')
        f.close()


    ## only pssm
    modelpath='../MODEL/keras/pssm'
    Xi = X[:, :, 50 - halfWin:51 + halfWin, 0:20]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        if j != nflod - 1:

            Xtest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = Ti[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((Ti.shape[0] - oneSplit, Ti.shape[1], Ti.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
        else:
            Xtest = Xi[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]
        model = LocalBLSTM(windowsize, 20)
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        ##train
        Xtrain, Ttrain = helper.convertTrainingData(Xtrain, Ttrain)
        callbacks_list = [checkpoint]
        model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                  , callbacks=callbacks_list, verbose=2)
        model.load_weights(modelpath)
        pr_t = model.predict(Xtest.reshape((Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2], Xtest.shape[3])))
        pr_t = pr_t.reshape((Ttest.shape[0], Ttest.shape[1], Ttest.shape[2]))
        result = helper.evaluateResult(Ttest, pr_t)
        print('use pssm ' + ', flod: ' + str(j))
        print(result)

        f = open('../resource/testOnresidueDim', 'a')
        f.writelines('pssm: split ' + str(j) + str(result) + '\n')
        f.close()

    ## pssm +polarity
    modelpath='../MODEL/keras/pssm_polarity'

    Xi=np.zeros((X.shape[0],X.shape[1],windowsize,21),np.float32)
    Xi[:,:,:,0:20]=X[:, :, 50 - halfWin:51 + halfWin, 0:20]
    Xi[:,:,:,20]=X[:, :, 50 - halfWin:51 + halfWin, 23]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        if j != nflod - 1:

            Xtest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = Ti[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((Ti.shape[0] - oneSplit, Ti.shape[1], Ti.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
        else:
            Xtest = Xi[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]
        model = LocalBLSTM(windowsize, 21)
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        ##train
        Xtrain, Ttrain = helper.convertTrainingData(Xtrain, Ttrain)
        callbacks_list = [checkpoint]
        model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                  , callbacks=callbacks_list, verbose=2)
        model.load_weights(modelpath)
        pr_t = model.predict(Xtest.reshape((Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2], Xtest.shape[3])))
        pr_t = pr_t.reshape((Ttest.shape[0], Ttest.shape[1], Ttest.shape[2]))
        result = helper.evaluateResult(Ttest, pr_t)
        print('pssm polarity flod: ' + str(j))
        print(result)

        f = open('../resource/testOnresidueDim', 'a')
        f.writelines('pssm polarity split ' + str(j) + str(result) + '\n')
        f.close()



    ## pssm+polarity+hydro+position
    modelpath='../MODEL/keras/pssm_polarity_hydro_position'
    Xi = X[:, :, 50 - halfWin:51 + halfWin, :]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        if j != nflod - 1:

            Xtest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = Ti[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((Ti.shape[0] - oneSplit, Ti.shape[1], Ti.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
        else:
            Xtest = Xi[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]
        model = LocalBLSTM(windowsize, 24)
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        ##train
        Xtrain, Ttrain = helper.convertTrainingData(Xtrain, Ttrain)
        callbacks_list = [checkpoint]
        model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                  , callbacks=callbacks_list, verbose=2)
        model.load_weights(modelpath)
        pr_t = model.predict(Xtest.reshape((Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2], Xtest.shape[3])))
        pr_t = pr_t.reshape((Ttest.shape[0], Ttest.shape[1], Ttest.shape[2]))
        result = helper.evaluateResult(Ttest, pr_t)
        print('all flod: ' + str(j))
        print(result)

        f = open('../resource/testOnresidueDim', 'a')
        f.writelines('all split ' + str(j) + str(result) + '\n')
        f.close()


## the following two block test full connect and LSTM


#### this block find the best window size of full connect(tranditional ANN)
## window size automatically varies from 5 to 71
## results will be wroted into resource/resultOfBestWindowFC
if 0:
    X = np.load('../resource/GLBlstmTrain/X.data.npy')
    T = np.load('../resource/GLBlstmTrain/T.data.npy')
    for iWinSize in range(34):
        gc.collect()
        windowsize=2*iWinSize+51
        halfWin=windowsize//2
        Xi=X[:,:,50-halfWin:51+halfWin,:]
        Ti=T
        modelpath='../MODEL/keras/bestWindSFC_'+str(windowsize)
        for j in range(nflod):

            oneSplit=Xi.shape[0]//nflod


            Xtest=Xi[j*oneSplit:(j+1)*oneSplit]
            Ttest=Ti[j*oneSplit:(j+1)*oneSplit]
            Xtrain=np.zeros((Xi.shape[0]-oneSplit,Xi.shape[1],Xi.shape[2],Xi.shape[3]),np.float32)
            Ttrain=np.zeros((Ti.shape[0]-oneSplit,Ti.shape[1],Ti.shape[2]),np.float32)
            Xtrain[:j*oneSplit]=Xi[:j*oneSplit]
            Xtrain[(j)*oneSplit:]=Xi[(j+1)*oneSplit:]
            Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]

            model=fullConnect(windowsize,24)
            checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            ##train
            Xtrain,Ttrain=helper.convertTrainingData(Xtrain,Ttrain)
            XtrainFlat=Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2]))
            callbacks_list=[checkpoint]
            model.fit(XtrainFlat, Ttrain, 128, validation_split=0.2, epochs=50
                      , callbacks=callbacks_list,verbose=1)
            model.load_weights(modelpath)
            pr_t=model.predict(Xtest.reshape((Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2]*Xtest.shape[3])))
            pr_t=pr_t.reshape((Ttest.shape[0],Ttest.shape[1],Ttest.shape[2]))
            result=helper.evaluateResult(Ttest,pr_t)
            print('bestWSfc: '+str(windowsize)+', flod: '+str(j))
            print(result)


            f=open('../resource/resultOfBestWindowFC','a')
            f.writelines(str(windowsize)+' split '+str(j)+str(result)+'\n')
            f.close()

## this block find the best window size of local LSTM
## window size automatically varies from 5 to 71
## results will be wroted into resource/resultOfFindBestWindowsize
if 0:
    X = np.load('../resource/GLBlstmTrain/X.data.npy')
    T = np.load('../resource/GLBlstmTrain/T.data.npy')
    for iWinSize in range(34):
        windowsize=6*iWinSize+5
        halfWin=windowsize//2
        Xi=X[:,:,50-halfWin:51+halfWin,:]
        Ti=T
        modelpath='../MODEL/keras/bestWindSLBLSTM_'+str(windowsize)
        for j in range(nflod):
            oneSplit=Xi.shape[0]//nflod
            if j!=nflod-1:

                Xtest=Xi[j*oneSplit:(j+1)*oneSplit]
                Ttest=Ti[j*oneSplit:(j+1)*oneSplit]
                Xtrain=np.zeros((Xi.shape[0]-oneSplit,Xi.shape[1],Xi.shape[2],Xi.shape[3]),np.float32)
                Ttrain=np.zeros((Ti.shape[0]-oneSplit,Ti.shape[1],Ti.shape[2]),np.float32)
                Xtrain[:j*oneSplit]=Xi[:j*oneSplit]
                Xtrain[(j)*oneSplit:]=Xi[(j+1)*oneSplit:]
                Ttrain[:j * oneSplit] = Ti[:j * oneSplit]
                Ttrain[(j) * oneSplit:] = Ti[(j + 1) * oneSplit:]
            else:
                Xtest=Xi[(j)*oneSplit:]
                Ttest=Ti[(j)*oneSplit:]
                Xtrain=Xi[:(j)*oneSplit]
                Ttrain=Ti[:(j)*oneSplit]
            model=LocalBLSTM(windowsize,24)
            checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            ##train
            Xtrain,Ttrain=helper.convertTrainingData(Xtrain,Ttrain)
            callbacks_list=[checkpoint]
            model.fit(Xtrain, Ttrain, 128, validation_split=0.2, epochs=40
                      , callbacks=callbacks_list,verbose=2)
            model.load_weights(modelpath)
            pr_t=model.predict(Xtest.reshape((Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2],Xtest.shape[3])))
            pr_t=pr_t.reshape((Ttest.shape[0],Ttest.shape[1],Ttest.shape[2]))
            result=helper.evaluateResult(Ttest,pr_t)
            print('windowSize: '+str(windowsize)+', flod: '+str(j))
            print(result)


            f=open('../resource/resultOfFindBestWindowsizeBLSTM','a')
            f.writelines(str(windowsize)+' split '+str(j)+str(result)+'\n')
            f.close()

## this block test the GL-LSTM network
if 0:
    winSize = 7
    halfWin = winSize // 2
    nflod = 7
    X = np.load('../resource/GLBlstmTrain/X.data.npy')
    T = np.load('../resource/GLBlstmTrain/T.data.npy')
    X = X[:, :, 50 - halfWin:51 + halfWin, :]
    ##X=X[:,:,50-halfWin:51+halfWin,:]
    for j in range(nflod):

        modelpath = '../MODEL/keras/globalONEHOT' + str(j)

        oneSplit = X.shape[0] // nflod
        Xtest = X[j * oneSplit:(j + 1) * oneSplit]
        Ttest = T[j * oneSplit:(j + 1) * oneSplit]

        Xtrain = np.zeros((X.shape[0] - oneSplit, X.shape[1], X.shape[2], X.shape[3]), np.float32)
        Ttrain = np.zeros((T.shape[0] - oneSplit, T.shape[1], T.shape[2]), np.float32)

        Xtrain[:j * oneSplit] = X[:j * oneSplit]
        Xtrain[(j) * oneSplit:] = X[(j + 1) * oneSplit:]
        Ttrain[:j * oneSplit] = T[:j * oneSplit]
        Ttrain[(j) * oneSplit:] = T[(j + 1) * oneSplit:]
        K.clear_session()

        model = glblstm()

        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc_on_protein', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        if 1:
            ##model.load_weights( modelpath)

            model.fit([Xtrain], [Ttrain], 128, validation_split=0.2, epochs=50
                      , verbose=2,
                      callbacks=callbacks_list)  ##, callbacks=[EarlyStopping(patience=3)])callbacks=callbacks_list,
        ##crightratio,rbondrightratio=test()##Xitrain Titrain
        ## test part
        model.load_weights(modelpath)
        pr_t = model.predict([Xtest])  ##Xitest pr_ti


        if 1:
            result = (helper.evaluateResult(Ttest, pr_t))
            print('GL-LSTM one hot: ' + str(result))
            f = open('../resource/glonehot', 'a')
            f.writelines("GL onehot: split: " + str(j) + str(result) + '\n')
            result = helper.adjustEvaluateResult(Ttest, pr_t)
            f.writelines("GL onehot: split: " + str(j) + str(result) + '\n')
            f.close()
            ##assert 0

