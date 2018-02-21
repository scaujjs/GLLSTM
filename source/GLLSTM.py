from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import LSTM,Masking
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Input,Conv2D, MaxPooling2D,UpSampling2D,Dense, Dropout,Lambda,merge,concatenate
from keras import initializers
from keras.layers.core import  Activation,  Flatten, Reshape
from keras import backend as K
import helper
import numpy as np
np.set_printoptions(threshold=np.nan)
import random
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import optimizers
np.set_printoptions(threshold=1000)
import gc


nflod=7

## experiment list



## impact of polarity and hydroph and postion
## find best window size for full-connect netowork
## find best window size for BiLSTM network
## find best window size for GL-BiLSTM network




##0.8857
## this function return GLB lstm
def GLBLSTM(CysNum,wins,CysInfo):
    hiddenUnit=30
    inputOfIndexOfCys = Input((CysNum,1,1))
    indexslist = helper.crop()(inputOfIndexOfCys)

    maskOfSeq = (Masking(mask_value=0, input_shape=(wins, CysInfo)))
    hiddhenSeq = Bidirectional(LSTM(hiddenUnit,activation='relu',kernel_initializer=initializers.glorot_normal()
                                    ,recurrent_initializer=initializers.glorot_normal()))

    inputOfSeq=Input((CysNum,wins,CysInfo))
    inputOfSeqList=helper.crop()(inputOfSeq)

    maskOfseqList=list()
    for i in range(len(inputOfSeqList)):
        maskOfseqList.append(maskOfSeq(inputOfSeqList[i]))

    hiddens=list()
    for i in range(len(maskOfseqList)):
        hiddens.append(Reshape(target_shape=(1,hiddenUnit*2))(hiddhenSeq(maskOfseqList[i])))

    for i in range(CysNum):
        hiddens[i]=concatenate([hiddens[i],indexslist[i]],axis=2)
    newInput=concatenate(hiddens,axis=1,name='midden_input')

    maskOfseg=(Masking(mask_value=0, input_shape=(CysNum, hiddenUnit*2+1),name='outputla'))(newInput)
    hiddenOfSEG=Bidirectional(LSTM(hiddenUnit,return_sequences=1, dropout=0.2, recurrent_regularizer=regularizers.l2(0.005),
								 kernel_regularizer=regularizers.l2(0.005),
								 bias_regularizer=regularizers.l2(0.005), recurrent_dropout=0.2 ))(maskOfseg)
    dense2=TimeDistributed(Dense(2,activation='softmax'))(hiddenOfSEG)
    model=Model(input=[inputOfSeq,inputOfIndexOfCys],output=dense2)

    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


## standard LSTM,
def LocalBLSTM(winSize, infoOfAA):

    input = Input((winSize, infoOfAA))
    mask = (Masking(mask_value=0, input_shape=(winSize, infoOfAA)))(input)
    hiddhen = Bidirectional(LSTM(40, activation='relu', dropout=0.2, recurrent_regularizer=regularizers.l2(0.005),
                                 kernel_regularizer=regularizers.l2(0.005),
                                 bias_regularizer=regularizers.l2(0.005), recurrent_dropout=0.2))(mask)
    output = Dense(2, activation='softmax')(hiddhen)
    model = Model(input=input, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def fullConnect(winSize,infoOfAA):
    model = Sequential()
    model.add(Dense(100, input_shape=(winSize*infoOfAA,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


## this block find the best windows size for full connect network
if 0:
    Xold = np.load('../resource/GLBlstmTrain/101/X.data.npy')
    T = np.load('../resource/GLBlstmTrain/101/T.data.npy')
    X=np.zeros((Xold.shape[0],Xold.shape[1],Xold.shape[2],Xold.shape[3]-1),np.float)
    X[:,:,:,0:22]=Xold[:,:,:,0:22]
    X[:,:,:,22]=Xold[:,:,:,23]
    for iWinSize in range(35):
        gc.collect()
        windowsize=2*iWinSize+35
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

            model=fullConnect(windowsize,23)
            checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            ##train
            Xtrain,Ttrain=helper.convertTrainingData(Xtrain,Ttrain)
            XtrainFlat=Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2]))
            callbacks_list=[checkpoint]
            model.fit(XtrainFlat, Ttrain, 128, validation_split=0.2, epochs=50
                      , callbacks=callbacks_list,verbose=2)
            model.load_weights(modelpath)
            pr_t=model.predict(Xtest.reshape((Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2]*Xtest.shape[3])))
            pr_t=pr_t.reshape((Ttest.shape[0],Ttest.shape[1],Ttest.shape[2]))
            result=helper.evaluateResult(Ttest,pr_t)
            print('bestWSfc: '+str(windowsize)+', flod: '+str(j))
            print(result)


            f=open('../resource/resultOfBestWindowFC','a')
            f.writelines(str(windowsize)+' split '+str(j)+str(result)+'\n')
            f.close()

##this block test impact of polarity and hydroph and position
if 0:
    windowsize=51
    halfWin=windowsize//2
    X = np.load('../resource/GLBlstmTrain/101/X.data.npy')
    Ti = np.load('../resource/GLBlstmTrain/101/T.data.npy')

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

    if 0:






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


## this block find the best window size of local LSTM
if 0:
    X = np.load('../resource/GLBlstmTrain/101/X.data.npy')
    T = np.load('../resource/GLBlstmTrain/101/T.data.npy')
    for iWinSize in range(7):
        windowsize=6*iWinSize+55
        halfWin=windowsize//2
        Xi=X[:,:,50-halfWin:51+halfWin,:]
        Ti=T
        modelpath='../MODEL/keras/bestWindSLBLSTM_'+str(windowsize)
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


            f=open('../resource/resultOfFindBestWindowsize','a')
            f.writelines(str(windowsize)+' split '+str(j)+str(result)+'\n')
            f.close()


## this block find the best window size of global LSTM
if 0:
    Xold = np.load('../resource/GLBlstmTrain/101/X.data.npy')
    X=np.zeros((Xold.shape[0],Xold.shape[1],Xold.shape[2],Xold.shape[3]-1),np.float)
    X[:,:,:,0:22]=Xold[:,:,:,0:22]
    X[:,:,:,22]=Xold[:,:,:,23]
    Xi = np.load('../resource/GLBlstmTrain/101/iX.data.npy')
    T = np.load('../resource/GLBlstmTrain/101/T.data.npy')
    for iWinSize in range(35):
        windowsize = 2 * iWinSize + 5
        halfWin = windowsize // 2
        Xwin = X[:, :, 50 - halfWin:51 + halfWin, :]
        Ti = T
        modelpath = '../MODEL/keras/bestWindSGlobal_BLSTM_' + str(windowsize)
        for j in range(nflod):


            oneSplit = Xi.shape[0] // nflod

            oneSplit = Xi.shape[0] // nflod
            Xtest = Xwin[j * oneSplit:(j + 1) * oneSplit]
            Xitest = Xi[j * oneSplit:(j + 1) * oneSplit]
            Ttest = T[j * oneSplit:(j + 1) * oneSplit]
            Xtrain = np.zeros((Xwin.shape[0] - oneSplit, Xwin.shape[1], Xwin.shape[2], Xwin.shape[3]), np.float32)
            Xitrain = np.zeros((Xi.shape[0] - oneSplit, Xi.shape[1], Xi.shape[2], Xi.shape[3]), np.float32)
            Ttrain = np.zeros((T.shape[0] - oneSplit, T.shape[1], T.shape[2]), np.float32)
            Xtrain[:j * oneSplit] = Xwin[:j * oneSplit]
            Xtrain[(j) * oneSplit:] = Xwin[(j + 1) * oneSplit:]
            Xitrain[:j * oneSplit] = Xi[:j * oneSplit]
            Xitrain[(j) * oneSplit:] = Xi[(j + 1) * oneSplit:]
            Ttrain[:j * oneSplit] = T[:j * oneSplit]
            Ttrain[(j) * oneSplit:] = T[(j + 1) * oneSplit:]

            model = GLBLSTM(25,windowsize, 23)
            checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            ##train
            callbacks_list = [checkpoint]
            model.fit([Xtrain, Xitrain], Ttrain, 128, validation_split=0.2, epochs=50
                      , callbacks=callbacks_list, verbose=2)
            model.load_weights(modelpath)
            pr_t = model.predict([Xtest, Xitest])
            result = helper.evaluateResult(Ttest, pr_t)
            print('windowSize: ' + str(windowsize) + ', flod: ' + str(j))
            print(result)

            f = open('../resource/bestWsGlobal', 'a')
            f.writelines(str(windowsize) + ' split ' + str(j) + str(result) + '\n')
            f.close()


## this block test GL-LSTM when window size( ), info of each amino( )
if 0:#
    winSize=53
    halfWin=winSize//2
    modelpath ='../MODEL/keras/gLSTM'

    X = np.load('../resource/GLBlstmTrain/101/X.data.npy')
    Xi = np.load('../resource/GLBlstmTrain/101/iX.data.npy')
    T = np.load('../resource/GLBlstmTrain/101/T.data.npy')
    X=X[:,:,50-halfWin:51+halfWin,:]
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        Xtest = X[j * oneSplit:(j + 1) * oneSplit]
        Xitest=Xi[j * oneSplit:(j + 1) * oneSplit]
        Ttest = T[j * oneSplit:(j + 1) * oneSplit]
        Xtrain = np.zeros((X.shape[0] - oneSplit, X.shape[1], X.shape[2], X.shape[3]), np.float32)
        Xitrain=np.zeros((Xi.shape[0]-oneSplit,Xi.shape[1],Xi.shape[2],Xi.shape[3]),np.float32)
        Ttrain = np.zeros((T.shape[0] - oneSplit, T.shape[1], T.shape[2]), np.float32)
        Xtrain[:j * oneSplit] = X[:j * oneSplit]
        Xtrain[(j) * oneSplit:] = X[(j + 1) * oneSplit:]
        Xitrain[:j * oneSplit]=Xi[:j * oneSplit]
        Xitrain[(j) * oneSplit:]=Xi[(j + 1) * oneSplit:]
        Ttrain[:j * oneSplit] = T[:j * oneSplit]
        Ttrain[(j) * oneSplit:] = T[(j + 1) * oneSplit:]


        model = GLBLSTM(25,winSize,24)
        ##plot_model(model,'./finalModel.png',show_shapes=1)

        ## train part
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        ##model.load_weights(helper.pOOD + 'MODEL/keras/' + modelName)

        model.fit([Xtrain,Xitrain], Ttrain, 128,validation_split=0.2 ,epochs=100
                  , callbacks=callbacks_list,verbose=2)##, callbacks=[EarlyStopping(patience=3)])
        ##crightratio,rbondrightratio=test()
        ## test part
        model.load_weights(modelpath)
        pr_t=model.predict([Xtest,Xitest])
        result=(helper.evaluateResult(Ttest,pr_t))
        print('GL-LSTM at batchsize 128: ' + str(result))
        f=open('../resource/gl_lstm','a')
        f.writelines("GL-LSTM at batchsize 128: "+str(result)+'\n')
        f.close()
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        Xtest = X[j * oneSplit:(j + 1) * oneSplit]
        Xitest=Xi[j * oneSplit:(j + 1) * oneSplit]
        Ttest = T[j * oneSplit:(j + 1) * oneSplit]
        Xtrain = np.zeros((X.shape[0] - oneSplit, X.shape[1], X.shape[2], X.shape[3]), np.float32)
        Xitrain=np.zeros((Xi.shape[0]-oneSplit,Xi.shape[1],Xi.shape[2],Xi.shape[3]),np.float32)
        Ttrain = np.zeros((T.shape[0] - oneSplit, T.shape[1], T.shape[2]), np.float32)
        Xtrain[:j * oneSplit] = X[:j * oneSplit]
        Xtrain[(j) * oneSplit:] = X[(j + 1) * oneSplit:]
        Xitrain[:j * oneSplit]=Xi[:j * oneSplit]
        Xitrain[(j) * oneSplit:]=Xi[(j + 1) * oneSplit:]
        Ttrain[:j * oneSplit] = T[:j * oneSplit]
        Ttrain[(j) * oneSplit:] = T[(j + 1) * oneSplit:]
        if 0:
            Xtest = X[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]

        model = GLBLSTM(25,winSize,24)
        ##plot_model(model,'./finalModel.png',show_shapes=1)

        ## train part
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        ##model.load_weights(helper.pOOD + 'MODEL/keras/' + modelName)

        model.fit([Xtrain,Xitrain], Ttrain, 64,validation_split=0.2 ,epochs=100
                  , callbacks=callbacks_list,verbose=2)##, callbacks=[EarlyStopping(patience=3)])
        ##crightratio,rbondrightratio=test()
        ## test part
        model.load_weights(modelpath)
        pr_t=model.predict([Xtest,Xitest])
        result=(helper.evaluateResult(Ttest,pr_t))
        print('GL-LSTM at batch size 64: ' + str(result))
        f=open('../resource/gl_lstm','a')
        f.writelines("GL-LSTM at batch size 64: "+str(result)+'\n')
        f.close()
    for j in range(nflod):
        oneSplit = Xi.shape[0] // nflod
        Xtest = X[j * oneSplit:(j + 1) * oneSplit]
        Xitest=Xi[j * oneSplit:(j + 1) * oneSplit]
        Ttest = T[j * oneSplit:(j + 1) * oneSplit]
        Xtrain = np.zeros((X.shape[0] - oneSplit, X.shape[1], X.shape[2], X.shape[3]), np.float32)
        Xitrain=np.zeros((Xi.shape[0]-oneSplit,Xi.shape[1],Xi.shape[2],Xi.shape[3]),np.float32)
        Ttrain = np.zeros((T.shape[0] - oneSplit, T.shape[1], T.shape[2]), np.float32)
        Xtrain[:j * oneSplit] = X[:j * oneSplit]
        Xtrain[(j) * oneSplit:] = X[(j + 1) * oneSplit:]
        Xitrain[:j * oneSplit]=Xi[:j * oneSplit]
        Xitrain[(j) * oneSplit:]=Xi[(j + 1) * oneSplit:]
        Ttrain[:j * oneSplit] = T[:j * oneSplit]
        Ttrain[(j) * oneSplit:] = T[(j + 1) * oneSplit:]
        if 0:
            Xtest = X[(j) * oneSplit:]
            Ttest = Ti[(j) * oneSplit:]
            Xtrain = Xi[:(j) * oneSplit]
            Ttrain = Ti[:(j) * oneSplit]

        model = GLBLSTM(25,winSize,24)
        ##plot_model(model,'./finalModel.png',show_shapes=1)

        ## train part
        checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        ##model.load_weights(helper.pOOD + 'MODEL/keras/' + modelName)

        model.fit([Xtrain,Xitrain], Ttrain, 32,validation_split=0.2 ,epochs=100
                  , callbacks=callbacks_list,verbose=2)##, callbacks=[EarlyStopping(patience=3)])
        ##crightratio,rbondrightratio=test()
        ## test part
        model.load_weights(modelpath)
        pr_t=model.predict([Xtest,Xitest])
        result=(helper.evaluateResult(Ttest,pr_t))
        print('GL-LSTM at batch size 32: ' + str(result))
        f=open('../resource/gl_lstm','a')
        f.writelines("GL-LSTM at batch size 32: "+str(result)+'\n')
        f.close()

